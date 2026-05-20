import warnings
import os
import pickle
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from gensim.models import FastText, LdaMulticore
from gensim import corpora
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

warnings.filterwarnings("ignore")

RESULTS_DIR      = "results"
SAVED_MODELS_DIR = "saved_models"
os.makedirs(RESULTS_DIR,      exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED   = 42
    BATCH_SIZE  = 32
    EPOCHS      = 30
    EMB_DIM     = 100
    MAX_LEN     = 30
    HIDDEN_DIM  = 128
    LR_LSTM     = 0.0005
    FOLDS       = 5
    PATIENCE    = 5
    LR_FACTOR   = 0.5
    LR_PATIENCE = 2
    FT_WINDOW    = 5
    FT_MIN_COUNT = 2
    FT_EPOCHS    = 10
    TRANSFORMER_MODEL     = "xlm-roberta-base"
    UKR_ROBERTA_MODEL     = "youscan/ukr-roberta-base"
    TRANSFORMER_EPOCHS    = 3
    TRANSFORMER_MAX_LEN   = 128
    TRANSFORMER_BATCH     = 16
    LR_TRANSFORMER        = 2e-5
    SLAVIC_LANGS = {"uk", "ru"}

cfg = Config()
torch.manual_seed(cfg.SEED)

class Vocabulary:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, freq_threshold: int = 2, max_size: int = 10_000):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size       = max_size

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    def build(self, sentences: list[str]):
        freq: dict[str, int] = {}
        for s in sentences:
            for w in self.tokenize(s):
                freq[w] = freq.get(w, 0) + 1
        idx = 4
        for word, cnt in sorted(freq.items(), key=lambda x: -x[1]):
            if cnt < self.freq_threshold or idx >= self.max_size:
                break
            self.stoi[word] = idx
            self.itos[idx]  = word
            idx += 1

    def numericalize(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.UNK) for w in self.tokenize(text)]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = (
            [self.vocab.SOS]
            + self.vocab.numericalize(self.texts[idx])
            + [self.vocab.EOS]
        )
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids += [self.vocab.PAD] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, output_dim)
        self.dropout   = nn.Dropout(0.3)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        emb         = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        return self.sigmoid(self.fc(h_n.squeeze(0)))

class BiLSTMFastText(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim,
                 pad_idx=0, pretrained_weights=None):
        super().__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm    = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1     = nn.Linear(2 * hidden_dim, 128)
        self.fc2     = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb    = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        out    = self.dropout(out[:, -1, :])
        out    = F.relu(self.fc1(out))
        return self.sigmoid(self.fc2(out))

class MetricHistoryCallback(TrainerCallback):
    def __init__(self):
        self.epochs   = []
        self.val_f1   = []
        self.val_loss = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.epochs.append(metrics.get("epoch", len(self.epochs) + 1))
            self.val_f1.append(metrics.get("eval_f1", 0.0))
            self.val_loss.append(metrics.get("eval_loss", 0.0))

def train_lstm(model, train_loader, val_loader, epochs, lr=0.001, patience=None):
    if patience is None:
        patience = cfg.PATIENCE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE,
    )
    model.to(cfg.DEVICE)

    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_loss  = float("inf")
    patience_count = 0
    best_weights   = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, leave=False, desc=f"  Epoch {epoch+1:02d}/{epochs}"):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            optimizer.zero_grad()
            out  = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += ((out > 0.5).float() == y).sum().item()
            tr_total   += len(y)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                out  = model(x).squeeze(-1)
                loss = criterion(out, y)
                v_loss    += loss.item() * len(y)
                preds      = (out > 0.5).float()
                v_correct += (preds == y).sum().item()
                v_total   += len(y)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        tr_l = tr_loss / tr_total;  v_l = v_loss / v_total
        tr_a = tr_correct / tr_total * 100
        v_a  = v_correct  / v_total  * 100

        train_losses.append(tr_l); val_losses.append(v_l)
        train_accs.append(tr_a);   val_accs.append(v_a)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(v_l)

        print(f"  Epoch {epoch+1:02d} | Loss {tr_l:.4f}/{v_l:.4f} | "
              f"Acc {tr_a:.1f}%/{v_a:.1f}% | LR {current_lr:.2e}")

        if v_l < best_val_loss:
            best_val_loss  = v_l
            patience_count = 0
            best_weights   = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping на епосі {epoch+1}")
                break

    model.load_state_dict(best_weights)
    print(f"  Завантажено ваги з найкращою val_loss: {best_val_loss:.4f}")
    return train_accs, val_accs, train_losses, val_losses, all_preds, all_labels

def _savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_training_curves(train_accs, val_accs, train_losses, val_losses,
                         model_name, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_accs) + 1)
    ax1.plot(epochs, train_accs,  "b-",  label="Train",      linewidth=2)
    ax1.plot(epochs, val_accs,    "r--", label="Validation",  linewidth=2)
    ax1.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, train_losses, "b-",  label="Train",      linewidth=2)
    ax2.plot(epochs, val_losses,   "r--", label="Validation",  linewidth=2)
    ax2.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("BCE Loss")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(filename)

def plot_transformer_history(callback, filename, title):
    if not callback.val_f1:
        return
    epochs = callback.epochs or list(range(1, len(callback.val_f1) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, callback.val_f1,   "g-o", linewidth=2, markersize=8)
    ax1.set_title(f"{title} — Val F1",  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted F1")
    ax1.set_ylim(0, 1); ax1.grid(alpha=0.3)
    ax2.plot(epochs, callback.val_loss, "r-o", linewidth=2, markersize=8)
    ax2.set_title(f"{title} — Val Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(filename)

def plot_confusion_matrices(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, preds, labels) in zip(axes, all_results):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"], linewidths=0.5)
        f1 = f1_score(labels, preds, average="weighted")
        ax.set_title(f"{name}\nF1={f1:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — All Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig("confusion_matrices.png")

def plot_f1_comparison(results):
    names  = list(results.keys())
    scores = [results[n]["f1"]          for n in names]
    stds   = [results[n].get("std", 0)  for n in names]

    color_map = {
        "Simple LSTM":        "#E87B4C",
        "Bi-LSTM + FastText": "#4C9BE8",
        "XLM-RoBERTa":        "#5CB85C",
        "ukr-RoBERTa":        "#9B59B6",
        "Ensemble Routing":   "#F39C12",
    }
    colors = [color_map.get(n, "#95A5A6") for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(names, scores, xerr=stds, color=colors, height=0.45,
                   edgecolor="white", capsize=5, error_kw={"elinewidth": 2})
    for bar, score, std in zip(bars, scores, stds):
        label = f"{score:.4f}" + (f" ±{std:.4f}" if std > 0 else "")
        ax.text(score + max(stds or [0]) + 0.01,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.20)
    ax.set_xlabel("Weighted F1-score", fontsize=12)
    ax.set_title(
        "Model Comparison — Weighted F1\n"
        "(LSTM: 5-fold CV ± std  |  Transformer: 80/20 split)",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(x=0.80, color="gray", linestyle="--", alpha=0.4, label="F1=0.80")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig("comparison_f1_bar.png")

def plot_classification_report_heatmap(all_results):
    records = []
    for name, preds, labels in all_results:
        report = classification_report(
            labels, preds,
            target_names=["Negative", "Positive"], output_dict=True,
        )
        for cls in ["Negative", "Positive"]:
            records.append({
                "Model":     name,
                "Class":     cls,
                "Precision": report[cls]["precision"],
                "Recall":    report[cls]["recall"],
                "F1-score":  report[cls]["f1-score"],
            })
    df_r = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(16, max(3, len(all_results) * 1.2 + 1)))
    for ax, metric in zip(axes, ["Precision", "Recall", "F1-score"]):
        pivot = df_r.pivot(index="Model", columns="Class", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel("")
    plt.suptitle("Classification Report — Per Model & Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("classification_report_heatmap.png")

def plot_lda_topics(lda_model, num_topics, top_n=8):
    cols       = min(num_topics, 3)
    rows_count = (num_topics + cols - 1) // cols
    fig, axes  = plt.subplots(rows_count, cols,
                               figsize=(6 * cols, 4 * rows_count))
    axes = np.array(axes).flatten()
    for i in range(num_topics):
        topic_terms = lda_model.show_topic(i, topn=top_n)
        words  = [t[0] for t in topic_terms][::-1]
        scores = [t[1] for t in topic_terms][::-1]
        axes[i].barh(words, scores, color=f"C{i}", alpha=0.8)
        axes[i].set_title(f"Тема {i + 1}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Вага слова")
        axes[i].grid(axis="x", alpha=0.3)
    for j in range(num_topics, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("LDA — Топ-слова по темах", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("lda_topics.png")

def plot_language_routing_results(routing_results: dict):
    langs   = list(routing_results.keys())
    xlm_f1  = [routing_results[l]["xlm_f1"]      for l in langs]
    ukr_f1  = [routing_results[l]["ukr_f1"]      for l in langs]
    ens_f1  = [routing_results[l]["ensemble_f1"] for l in langs]

    x     = np.arange(len(langs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, xlm_f1, width, label="XLM-RoBERTa",    color="#5CB85C")
    b2 = ax.bar(x,          ukr_f1, width, label="ukr-RoBERTa",    color="#9B59B6")
    b3 = ax.bar(x + width,  ens_f1, width, label="Ensemble Routing", color="#F39C12")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Weighted F1", fontsize=12)
    ax.set_title(
        "F1 по мовах: XLM-RoBERTa vs ukr-RoBERTa vs Ensemble Routing\n"
        "Ensemble = ukr-RoBERTa для uk/ru, XLM-RoBERTa для решти",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig("language_routing_results.png")

def plot_context_demo(rows):
    col_labels = ["Текст (скорочено)", "Тип", "Очікується",
                  "Simple LSTM", "Bi-LSTM", "Transformer"]

    def _fmt(label, conf):
        return f"{label}\n({conf:.0%})"

    def _color(predicted, expected):
        return "#A8D5A2" if predicted == expected else "#F4A0A0"

    cell_text, cell_colors = [], []
    for r in rows:
        text_short = (r["text"][:42] + "…") if len(r["text"]) > 42 else r["text"]
        exp        = r["expected"]
        cell_text.append([
            text_short, r["note"], exp,
            _fmt(r["simple_lstm_label"], r["simple_lstm_conf"]),
            _fmt(r["bilstm_label"],       r["bilstm_conf"]),
            _fmt(r["transformer_label"],  r["transformer_conf"]),
        ])
        cell_colors.append([
            "#F5F5F5", "#EEF0FF", "#DDEEFF",
            _color(r["simple_lstm_label"], exp),
            _color(r["bilstm_label"],       exp),
            _color(r["transformer_label"],  exp),
        ])

    fig, ax = plt.subplots(figsize=(18, max(4, len(rows) * 1.1 + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(
        "Демо: Сарказм та Заперечення — Simple LSTM vs Transformer\n"
        "Зелений = правильно   |   Червоний = помилково",
        fontsize=13, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    _savefig("sarcasm_demo.png")

class ModelBenchmark:

    def __init__(self, df, text_col="clean_text", label_col="Label"):
        self.df       = df.copy().reset_index(drop=True)
        self.text_col = text_col

        if "y" not in self.df.columns:
            if label_col in self.df.columns:
                self.df["y"] = (
                    self.df[label_col].astype(str).str.lower() == "positive"
                ).astype(int)

        self.train_df, self.val_df = train_test_split(
            self.df, test_size=0.2, random_state=cfg.SEED, stratify=self.df["y"],
        )
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df   = self.val_df.reset_index(drop=True)

        for _df in (self.train_df, self.val_df):
            _df[self.text_col] = _df[self.text_col].fillna("").astype(str)

        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)}")
        print(self.val_df["y"].value_counts().to_string())

        self.results: dict = {}

        self._lstm_simple  = None
        self._lstm_bilstm  = None
        self._transformer  = None
        self._tf_tokenizer = None
        self._ukr_model    = None
        self._ukr_tokenizer = None

    def _build_vocab(self):
        texts = self.train_df[self.text_col].fillna("").astype(str).tolist()
        vocab = Vocabulary(freq_threshold=2)
        vocab.build(texts)
        return vocab

    def _make_loaders(self, tr_texts, tr_labels, vl_texts, vl_labels, vocab):
        tr_ds = SentimentDataset(tr_texts, tr_labels, vocab, cfg.MAX_LEN)
        vl_ds = SentimentDataset(vl_texts, vl_labels, vocab, cfg.MAX_LEN)
        return (DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True),
                DataLoader(vl_ds, batch_size=cfg.BATCH_SIZE, shuffle=False))

    def _fine_tune_transformer(self, model_name, train_df, val_df, save_path, title):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_batch(batch):
            return tokenizer(
                batch[self.text_col],
                truncation=True, padding="max_length",
                max_length=cfg.TRANSFORMER_MAX_LEN,
            )

        def make_hf(df):
            hf = HFDataset.from_pandas(
                df[[self.text_col, "y"]]
                .assign(**{self.text_col: df[self.text_col].fillna("").astype(str)})
                .rename(columns={"y": "labels"})
                .reset_index(drop=True)
            )
            hf = hf.map(tokenize_batch, batched=True).remove_columns([self.text_col])
            hf.set_format("torch")
            return hf

        train_hf = make_hf(train_df)
        val_hf   = make_hf(val_df)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )
        history_cb = MetricHistoryCallback()

        training_args = TrainingArguments(
            output_dir                  = f"./{save_path}_results",
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            learning_rate               = cfg.LR_TRANSFORMER,
            per_device_train_batch_size = cfg.TRANSFORMER_BATCH,
            per_device_eval_batch_size  = cfg.TRANSFORMER_BATCH,
            num_train_epochs            = cfg.TRANSFORMER_EPOCHS,
            weight_decay                = 0.01,
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            logging_steps               = 20,
            fp16                        = torch.cuda.is_available(),
            report_to                   = "none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=1)
            return {"f1": f1_score(labels, preds, average="weighted"),
                    "accuracy": float((preds == labels).mean())}

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_hf, eval_dataset=val_hf,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[history_cb, EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

        pred_out = trainer.predict(val_hf)
        preds    = pred_out.predictions.argmax(axis=1).tolist()
        labels   = pred_out.label_ids.tolist()
        f1       = f1_score(labels, preds, average="weighted")

        print(f"\n{title} — Weighted F1: {f1:.4f}")
        print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

        trainer.save_model(f"{SAVED_MODELS_DIR}/{save_path}")
        tokenizer.save_pretrained(f"{SAVED_MODELS_DIR}/{save_path}")
        print(f"Збережено: {SAVED_MODELS_DIR}/{save_path}/")

        return f1, preds, labels, history_cb, trainer.model, tokenizer

    def run_lda(self, num_topics=5, no_below=5, no_above=0.5,
                iterations=400, top_words=8):
        print("\n" + "=" * 60)
        print(f"LDA — ТЕМАТИЧНЕ МОДЕЛЮВАННЯ ({num_topics} тем)")
        print("=" * 60)

        tokens  = (self.train_df[self.text_col].fillna("").astype(str)
                   .apply(str.split).tolist())
        id2word = corpora.Dictionary(tokens)
        id2word.filter_extremes(no_below=no_below, no_above=no_above)
        corpus  = [id2word.doc2bow(t) for t in tokens]

        lda = LdaMulticore(corpus=corpus, id2word=id2word,
                           num_topics=num_topics, iterations=iterations)
        for idx, topic in lda.print_topics(-1):
            print(f"  Тема {idx+1}: {topic}\n")

        plot_lda_topics(lda, num_topics, top_n=top_words)

        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
            vis  = gensimvis.prepare(lda, corpus, id2word)
            html = os.path.join(RESULTS_DIR, "lda_visualization.html")
            pyLDAvis.save_html(vis, html)
            print(f"Saved: {html}")
        except ImportError:
            print("pyLDAvis не встановлено.")

        self._lda_model = lda
        return lda

    def run_simple_lstm(self):
        print("\n" + "=" * 60)
        print("MODEL 1: Simple LSTM")
        print("=" * 60)

        all_texts  = self.train_df[self.text_col].fillna("").astype(str).tolist()
        all_labels = self.train_df["y"].tolist()
        vocab      = self._build_vocab()

        skf      = StratifiedKFold(n_splits=cfg.FOLDS, shuffle=True, random_state=cfg.SEED)
        fold_f1s = []

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(all_texts, all_labels)):
            print(f"\n  ── Fold {fold+1}/{cfg.FOLDS} ──")
            tr_texts = [all_texts[i] for i in tr_idx]
            tr_lbls  = [all_labels[i] for i in tr_idx]
            vl_texts = [all_texts[i] for i in vl_idx]
            vl_lbls  = [all_labels[i] for i in vl_idx]
            tr_ld, vl_ld = self._make_loaders(tr_texts, tr_lbls, vl_texts, vl_lbls, vocab)
            fold_model = SimpleLSTM(len(vocab), cfg.EMB_DIM, cfg.HIDDEN_DIM, 1, vocab.PAD)
            _, _, _, _, f_preds, f_labels = train_lstm(fold_model, tr_ld, vl_ld, cfg.EPOCHS, cfg.LR_LSTM)
            fold_f1 = f1_score(f_labels, f_preds, average="weighted")
            fold_f1s.append(fold_f1)
            print(f"  Fold {fold+1} F1: {fold_f1:.4f}")

        mean_f1 = float(np.mean(fold_f1s))
        std_f1  = float(np.std(fold_f1s))
        print(f"\nSimple LSTM — CV Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

        val_texts  = self.val_df[self.text_col].fillna("").astype(str).tolist()
        val_labels = self.val_df["y"].tolist()
        final_tr_ld, final_vl_ld = self._make_loaders(
            all_texts, all_labels, val_texts, val_labels, vocab)
        model = SimpleLSTM(len(vocab), cfg.EMB_DIM, cfg.HIDDEN_DIM, 1, vocab.PAD)
        tr_accs, v_accs, tr_losses, v_losses, preds, labels = train_lstm(
            model, final_tr_ld, final_vl_ld, cfg.EPOCHS, cfg.LR_LSTM)
        print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
        plot_training_curves(tr_accs, v_accs, tr_losses, v_losses,
                             "Simple LSTM", "training_curves_simple_lstm.png")

        torch.save(model.state_dict(), f"{SAVED_MODELS_DIR}/simple_lstm.pt")
        with open(f"{SAVED_MODELS_DIR}/simple_lstm_vocab.pkl", "wb") as fh:
            pickle.dump(vocab, fh)
        with open(f"{SAVED_MODELS_DIR}/simple_lstm_config.pkl", "wb") as fh:
            pickle.dump({"vocab_size": len(vocab), "emb_dim": cfg.EMB_DIM,
                         "hidden_dim": cfg.HIDDEN_DIM, "output_dim": 1,
                         "pad_idx": vocab.PAD}, fh)

        self._lstm_simple = (model, vocab)
        self.results["Simple LSTM"] = {
            "f1": mean_f1, "std": std_f1, "preds": preds,
            "labels": labels, "eval_method": f"{cfg.FOLDS}-fold CV",
        }
        return mean_f1

    def run_bilstm_fasttext(self):
        print("\n" + "=" * 60)
        print("MODEL 2: Bi-LSTM + FastText")
        print("=" * 60)

        all_texts  = self.train_df[self.text_col].fillna("").astype(str).tolist()
        all_labels = self.train_df["y"].tolist()

        print("  Навчання FastText...")
        ft_model = FastText(
            sentences=[t.split() for t in all_texts],
            vector_size=cfg.EMB_DIM, window=cfg.FT_WINDOW,
            min_count=cfg.FT_MIN_COUNT, workers=4, sg=1, epochs=cfg.FT_EPOCHS,
        )
        vocab   = self._build_vocab()
        weights = np.zeros((len(vocab), cfg.EMB_DIM), dtype=np.float32)
        for word, idx in vocab.stoi.items():
            if word in ft_model.wv:
                weights[idx] = ft_model.wv[word]
        weights_tensor = torch.tensor(weights)

        skf      = StratifiedKFold(n_splits=cfg.FOLDS, shuffle=True, random_state=cfg.SEED)
        fold_f1s = []

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(all_texts, all_labels)):
            print(f"\n  ── Fold {fold+1}/{cfg.FOLDS} ──")
            tr_texts = [all_texts[i] for i in tr_idx]
            tr_lbls  = [all_labels[i] for i in tr_idx]
            vl_texts = [all_texts[i] for i in vl_idx]
            vl_lbls  = [all_labels[i] for i in vl_idx]
            tr_ld, vl_ld = self._make_loaders(tr_texts, tr_lbls, vl_texts, vl_lbls, vocab)
            fold_model = BiLSTMFastText(len(vocab), cfg.EMB_DIM, cfg.HIDDEN_DIM, 1,
                                        vocab.PAD, weights_tensor)
            _, _, _, _, f_preds, f_labels = train_lstm(fold_model, tr_ld, vl_ld, cfg.EPOCHS, cfg.LR_LSTM)
            fold_f1 = f1_score(f_labels, f_preds, average="weighted")
            fold_f1s.append(fold_f1)
            print(f"  Fold {fold+1} F1: {fold_f1:.4f}")

        mean_f1 = float(np.mean(fold_f1s))
        std_f1  = float(np.std(fold_f1s))
        print(f"\nBi-LSTM + FastText — CV Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

        val_texts  = self.val_df[self.text_col].fillna("").astype(str).tolist()
        val_labels = self.val_df["y"].tolist()
        final_tr_ld, final_vl_ld = self._make_loaders(
            all_texts, all_labels, val_texts, val_labels, vocab)
        model = BiLSTMFastText(len(vocab), cfg.EMB_DIM, cfg.HIDDEN_DIM, 1,
                               vocab.PAD, weights_tensor)
        tr_accs, v_accs, tr_losses, v_losses, preds, labels = train_lstm(
            model, final_tr_ld, final_vl_ld, cfg.EPOCHS, cfg.LR_LSTM)
        print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
        plot_training_curves(tr_accs, v_accs, tr_losses, v_losses,
                             "Bi-LSTM + FastText", "training_curves_bilstm_fasttext.png")

        torch.save(model.state_dict(), f"{SAVED_MODELS_DIR}/bilstm_fasttext.pt")
        with open(f"{SAVED_MODELS_DIR}/bilstm_vocab.pkl", "wb") as fh:
            pickle.dump(vocab, fh)
        ft_model.save(f"{SAVED_MODELS_DIR}/fasttext.model")
        with open(f"{SAVED_MODELS_DIR}/bilstm_config.pkl", "wb") as fh:
            pickle.dump({"vocab_size": len(vocab), "emb_dim": cfg.EMB_DIM,
                         "hidden_dim": cfg.HIDDEN_DIM, "output_dim": 1,
                         "pad_idx": vocab.PAD}, fh)

        self._lstm_bilstm = (model, vocab)
        self.results["Bi-LSTM + FastText"] = {
            "f1": mean_f1, "std": std_f1, "preds": preds,
            "labels": labels, "eval_method": f"{cfg.FOLDS}-fold CV",
        }
        return mean_f1

    def run_transformer(self, model_name=None):
        model_name = model_name or cfg.TRANSFORMER_MODEL
        print("\n" + "=" * 60)
        print(f"MODEL 3: XLM-RoBERTa ({model_name})")
        print("  Мультимовний — en/uk/ru/zh/es та ін.")
        print("=" * 60)

        f1, preds, labels, history_cb, model, tokenizer = self._fine_tune_transformer(
            model_name, self.train_df, self.val_df,
            save_path="transformer", title="XLM-RoBERTa",
        )
        plot_transformer_history(history_cb, "training_curves_transformer.png",
                                 "XLM-RoBERTa")

        self._transformer   = model
        self._tf_tokenizer  = tokenizer
        self._xlm_f1 = f1   
        print(f"XLM-RoBERTa F1 (внутрішній): {f1:.4f}")
        return f1

    def run_ukr_roberta(self):
        print("\n" + "=" * 60)
        print("MODEL 4: ukr-RoBERTa (youscan/ukr-roberta-base)")
        print("  Навчена на Ukrainian/Russian корпусах.")
        print("  Тренується на uk/ru підмножині датасету.")
        print("=" * 60)

        if "language" not in self.train_df.columns:
            print("  [WARN] Колонка 'language' відсутня у датасеті.")
            print("  Додай 'language' до processed_reviews.csv у clean_text.py")
            print("  Тренуємо на всьому датасеті як fallback...")
            train_sub = self.train_df
            val_sub   = self.val_df
        else:
            train_sub = self.train_df[
                self.train_df["language"].isin(cfg.SLAVIC_LANGS)
            ].reset_index(drop=True)
            val_sub = self.val_df[
                self.val_df["language"].isin(cfg.SLAVIC_LANGS)
            ].reset_index(drop=True)

            print(f"  uk/ru train: {len(train_sub)} | val: {len(val_sub)}")

            if len(train_sub) < 100:
                print("  [WARN] Замало uk/ru даних. Тренуємо на всьому датасеті.")
                train_sub = self.train_df
                val_sub   = self.val_df

        f1, preds, labels, history_cb, model, tokenizer = self._fine_tune_transformer(
            cfg.UKR_ROBERTA_MODEL, train_sub, val_sub,
            save_path="ukr_roberta", title="ukr-RoBERTa",
        )
        plot_transformer_history(history_cb, "training_curves_ukr_roberta.png",
                                 "ukr-RoBERTa")

        self._ukr_model     = model
        self._ukr_tokenizer = tokenizer
        self._ukr_f1 = f1  
        print(f"ukr-RoBERTa F1 (внутрішній): {f1:.4f}")
        return f1

    def run_ensemble_routing(self):
        print("\n" + "=" * 60)
        print("MODEL 5: Ensemble Routing")
        print("  uk/ru → ukr-RoBERTa | решта → XLM-RoBERTa")
        print("=" * 60)

        if self._transformer is None or self._ukr_model is None:
            print("[ERROR] Спочатку запусти run_transformer() і run_ukr_roberta()")
            return 0.0

        val_df = self.val_df.copy()

        has_lang = "language" in val_df.columns

        all_preds  = []
        all_labels = val_df["y"].tolist()

        self._transformer.eval().to(cfg.DEVICE)
        self._ukr_model.eval().to(cfg.DEVICE)

        lang_results: dict[str, dict] = {}

        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc="Ensemble inference"):
            text = str(row[self.text_col])
            lang = str(row.get("language", "en")) if has_lang else "en"

            if lang in cfg.SLAVIC_LANGS and self._ukr_model is not None:
                model     = self._ukr_model
                tokenizer = self._ukr_tokenizer
            else:
                model     = self._transformer
                tokenizer = self._tf_tokenizer

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=cfg.TRANSFORMER_MAX_LEN,
            )
            enc = {k: v.to(cfg.DEVICE) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits

            pred = int(logits.argmax(dim=1).item())
            all_preds.append(pred)

            if has_lang:
                if lang not in lang_results:
                    lang_results[lang] = {"preds": [], "labels": []}
                lang_results[lang]["preds"].append(pred)
                lang_results[lang]["labels"].append(int(row["y"]))

        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"\nEnsemble Routing — Weighted F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds,
                                    target_names=["Negative", "Positive"]))

        self.results["Transformer Ensemble"] = {
            "f1": f1, "std": 0.0,
            "preds": all_preds, "labels": all_labels,
            "eval_method": "80/20 split (XLM + ukr routing)",
        }

        xlm_f1 = getattr(self, "_xlm_f1", None)
        ukr_f1 = getattr(self, "_ukr_f1", None)
        if xlm_f1 and ukr_f1:
            print(f"\n  [Внутрішні метрики Ensemble]")
            print(f"  XLM-RoBERTa  : {xlm_f1:.4f}")
            print(f"  ukr-RoBERTa  : {ukr_f1:.4f}")
            print(f"  Ensemble     : {f1:.4f}")

        if has_lang and lang_results:
            self._plot_routing_by_language(lang_results)

        return f1

    def _plot_routing_by_language(self, lang_results: dict):
        routing_data = {}

        self._transformer.eval().to(cfg.DEVICE)
        if self._ukr_model:
            self._ukr_model.eval().to(cfg.DEVICE)

        for lang, data in lang_results.items():
            if len(data["labels"]) < 10:
                continue

            ens_f1 = f1_score(data["labels"], data["preds"], average="weighted")

            xlm_f1 = self.results.get("XLM-RoBERTa", {}).get("f1", 0.0)
            ukr_f1 = self.results.get("ukr-RoBERTa",  {}).get("f1", 0.0)

            routing_data[lang] = {
                "xlm_f1":      xlm_f1,
                "ukr_f1":      ukr_f1,
                "ensemble_f1": ens_f1,
            }

        if routing_data:
            plot_language_routing_results(routing_data)

    def _predict_lstm(self, model, vocab, text):
        ids = [vocab.SOS] + vocab.numericalize(text.lower()) + [vocab.EOS]
        if len(ids) > cfg.MAX_LEN:
            ids = ids[:cfg.MAX_LEN]
        else:
            ids += [vocab.PAD] * (cfg.MAX_LEN - len(ids))
        tensor = torch.tensor([ids], dtype=torch.long).to(cfg.DEVICE)
        model.eval().to(cfg.DEVICE)
        with torch.no_grad():
            score = float(model(tensor).squeeze().cpu().item())
        label = "positive" if score > 0.5 else "negative"
        conf  = score if score > 0.5 else 1.0 - score
        return label, conf

    def _predict_transformer(self, text, language="en"):
        if (language in cfg.SLAVIC_LANGS
                and self._ukr_model is not None
                and self._ukr_tokenizer is not None):
            model     = self._ukr_model
            tokenizer = self._ukr_tokenizer
        else:
            model     = self._transformer
            tokenizer = self._tf_tokenizer

        model.eval().to(cfg.DEVICE)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=cfg.TRANSFORMER_MAX_LEN,
        )
        inputs = {k: v.to(cfg.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs    = torch.softmax(logits, dim=1)[0]
        label_id = int(logits.argmax(dim=1).item())
        return {0: "negative", 1: "positive"}[label_id], float(probs[label_id].cpu())

    _DEMO_EXAMPLES = [
        ("This course is absolutely amazing, best investment ever!",   "positive", "Явно позитивний"),
        ("Terrible platform, crashes every session, complete waste.",   "negative", "Явно негативний"),
        ("Oh sure, support responds super fast — only waited 3 weeks!", "negative", "Сарказм (підтримка)"),
        ("The content is not bad at all, actually quite useful.",       "positive", "Заперечення 'not bad'"),
        ("Incredible instructor — explains once, never answers questions.", "negative", "Іронія"),
        ("Course is fine but the price is absolutely not worth it.",   "negative", "Заперечення + mixed"),
        ("Курс чудовий, але платформа постійно гальмує.",              "negative", "Мішаний (укр.)"),
        ("Викладач просто неймовірний — відповідає раз на місяць.",   "negative", "Сарказм (укр.)"),
    ]

    def demo_context(self):
        if self._lstm_simple is None or self._transformer is None:
            print("Спочатку запусти run_simple_lstm() та run_transformer()")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("ДЕМО: КОНТЕКСТ ТА САРКАЗМ")
        print("=" * 60)

        rows = []
        for text, expected, note in self._DEMO_EXAMPLES:
            sl_lbl, sl_conf = self._predict_lstm(
                self._lstm_simple[0], self._lstm_simple[1], text)
            bl_lbl, bl_conf = (self._predict_lstm(
                self._lstm_bilstm[0], self._lstm_bilstm[1], text)
                if self._lstm_bilstm else ("—", 0.0))

            lang = "uk" if any("\u0400" <= c <= "\u04ff" for c in text) else "en"
            tf_lbl, tf_conf = self._predict_transformer(text, language=lang)

            print(f"\n  [{note}]\n  Text: {text}")
            print(f"  Simple LSTM: {sl_lbl} ({sl_conf:.0%})  "
                  f"{'✓' if sl_lbl==expected else '✗'}")
            print(f"  Bi-LSTM    : {bl_lbl} ({bl_conf:.0%})")
            print(f"  Transformer: {tf_lbl} ({tf_conf:.0%})  "
                  f"({'✓' if tf_lbl==expected else '✗'}, "
                  f"{'ukr-RoBERTa' if lang in cfg.SLAVIC_LANGS and self._ukr_model else 'XLM-RoBERTa'})")

            rows.append({
                "text": text, "note": note, "expected": expected,
                "simple_lstm_label": sl_lbl, "simple_lstm_conf": sl_conf,
                "bilstm_label": bl_lbl, "bilstm_conf": bl_conf,
                "transformer_label": tf_lbl, "transformer_conf": tf_conf,
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, "sarcasm_demo_results.csv"), index=False)
        plot_context_demo(rows)
        return df
    
    def report(self):
        print("\n" + "=" * 60)
        print("ПІДСУМКОВЕ ПОРІВНЯННЯ")
        print("=" * 60)

        rows = [{"Model": name, "F1 (weighted)": round(data["f1"], 4),
                 "Std": round(data.get("std", 0.0), 4),
                 "Eval method": data.get("eval_method", "—")}
                for name, data in self.results.items()]
        comp_df = (pd.DataFrame(rows)
                   .sort_values("F1 (weighted)", ascending=False)
                   .reset_index(drop=True))
        comp_df.index += 1
        print(comp_df.to_string())

        csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
        comp_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        all_results = [(name, data["preds"], data["labels"])
                       for name, data in self.results.items()]

        plot_f1_comparison(self.results)
        plot_confusion_matrices(all_results)
        plot_classification_report_heatmap(all_results)

        return comp_df

if __name__ == "__main__":
    df = pd.read_csv("processed_reviews.csv")
    assert "clean_text" in df.columns, "Потрібна колонка 'clean_text'"

    bench = ModelBenchmark(df=df, text_col="clean_text", label_col="Label")
    bench.run_lda(num_topics=3)
    bench.run_simple_lstm()
    bench.run_bilstm_fasttext()
    bench.run_transformer("xlm-roberta-base")
    bench.run_ukr_roberta()
    bench.run_ensemble_routing()

    comparison = bench.report()
    bench.demo_context()