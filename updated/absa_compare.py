                                                              
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

BASE_DIR         = Path(__file__).parent
DATA_PATH        = BASE_DIR / "processed_reviews_labeled.csv"
RESULTS_DIR      = BASE_DIR / "results_4_3_compare"
MODELS_ATTENTION = BASE_DIR / "saved_models_4_3"
MODELS_BERT      = BASE_DIR / "saved_models_4_3_bert" / "bert_absa"

RESULTS_DIR.mkdir(exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
ASPECTS    = ["content_quality", "clarity", "difficulty"]
VAL_SAMPLE_SIZE = 2000 if DEVICE == "cuda" else 800

EMB_DIM    = 100
HIDDEN_DIM = 128
MAX_LEN    = 50
DROPOUT    = 0.3

ASPECT_DESCRIPTIONS: dict[str, str] = {
    "content_quality": "quality and depth of course content and materials",
    "clarity":         "clarity of explanations and teaching style",
    "difficulty":      "difficulty level, pace and workload of the course",
}

DEMO_REVIEWS: list[dict] = [
    {
        "text": "The course material is comprehensive, but the instructor explains it in a confusing way and the workload is too heavy.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "negative"},
    },
    {
        "text": "The videos are short and clear, although the assignments are quite demanding.",
        "expected": {"content_quality": "positive", "clarity": "positive", "difficulty": "negative"},
    },
    {
        "text": "The topic is interesting, but the course lacks practical examples and the pace is too fast for beginners.",
        "expected": {"content_quality": "negative", "clarity": "positive", "difficulty": "negative"},
    },
    {
        "text": "Excellent explanations and useful examples, but the final project is much harder than the lectures.",
        "expected": {"content_quality": "positive", "clarity": "positive", "difficulty": "negative"},
    },
    {
        "text": "The content feels outdated, but the teacher explains every concept clearly and the workload is reasonable.",
        "expected": {"content_quality": "negative", "clarity": "positive", "difficulty": "positive"},
    },
    {
        "text": "Good structure and relevant readings, but the lecturer jumps between topics and the quizzes are confusing.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "negative"},
    },
    {
        "text": "Чудовий матеріал і корисні приклади, але викладач пояснює нечітко і занадто швидко.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "negative"},
    },
    {
        "text": "Завдання складні, але пояснення зрозумілі, а матеріали курсу добре структуровані.",
        "expected": {"content_quality": "positive", "clarity": "positive", "difficulty": "negative"},
    },
    {
        "text": "The course is easy to follow and not too difficult, but many lectures repeat the same basic content.",
        "expected": {"content_quality": "negative", "clarity": "positive", "difficulty": "positive"},
    },
    {
        "text": "The instructor is monotone, yet the readings are relevant and the assignments are fair.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "positive"},
    },
]

print(f"Device: {DEVICE}")

class Vocabulary:
    def __init__(self, freq_threshold: int = 3, max_size: int = 15000):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size       = max_size

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.lower().strip() for t in str(text).split() if t.strip()]

    def numericalize(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.stoi["<UNK>"]) for w in self.tokenize(text)]

class AspectAttention(nn.Module):
    """Адитивна (Bahdanau) увага — v3."""
    TEMPERATURE = 2.0

    def __init__(self, hidden_dim: int, aspect_emb_dim: int):
        super().__init__()
        self.W_h = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.W_a = nn.Linear(aspect_emb_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, aspect_vec):
        h_proj  = self.W_h(lstm_out)
        a_proj  = self.W_a(aspect_vec).unsqueeze(0)
        score   = self.v(torch.tanh(h_proj + a_proj)).squeeze(-1)
        alpha   = F.softmax(score / self.TEMPERATURE, dim=0).transpose(0, 1)
        context = torch.bmm(alpha.unsqueeze(1), lstm_out.permute(1, 0, 2)).squeeze(1)
        return context, alpha

class ModelWithAttention(nn.Module):
    """Bi-LSTM + separate attention/gate/head for every aspect."""
    GATE_DIM = 64

    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        emb_layer,
        aspect_emb_dim,
        n_aspects,
        dropout,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.n_aspects   = n_aspects
        self.embedding   = emb_layer
        self.lstm        = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=False)
        self.dropout     = nn.Dropout(dropout)
        self.attentions  = nn.ModuleList([
            AspectAttention(hidden_dim, aspect_emb_dim) for _ in range(n_aspects)
        ])
        self.gates       = nn.ModuleList([
            nn.Linear(aspect_emb_dim, self.GATE_DIM) for _ in range(n_aspects)
        ])
        self.fc1s        = nn.ModuleList([
            nn.Linear(2 * hidden_dim + self.GATE_DIM, 64) for _ in range(n_aspects)
        ])
        self.fc2s        = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(n_aspects)
        ])

    def encode(self, text):
        _, N = text.shape
        h0 = torch.zeros(2, N, self.hidden_dim, device=text.device)
        c0 = torch.zeros(2, N, self.hidden_dim, device=text.device)
        emb = self.embedding(text)
        lstm_out, (h_n, _) = self.lstm(emb, (h0, c0))
        last_h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return lstm_out, last_h

    def forward_aspect(self, lstm_out, last_h, aspect_vec, asp_idx):
        context, alpha = self.attentions[asp_idx](lstm_out, aspect_vec)
        repr_ = context + last_h
        gate = F.relu(self.gates[asp_idx](aspect_vec))
        combined = self.dropout(torch.cat([repr_, gate], dim=-1))
        y = F.relu(self.fc1s[asp_idx](combined))
        y = torch.sigmoid(self.fc2s[asp_idx](y)).squeeze(-1)
        return y, alpha

    def forward(self, text, aspect_vecs):
        lstm_out, last_h = self.encode(text)
        outputs, alphas = [], []
        for i, av in enumerate(aspect_vecs):
            out, alpha = self.forward_aspect(lstm_out, last_h, av, i)
            outputs.append(out)
            alphas.append(alpha)
        return outputs, alphas

class AttentionLoader:
    """Завантажує збережену ABSA-Attention модель без перенавчання."""

    def __init__(self):
        self.model: ModelWithAttention | None = None
        self.vocab: Vocabulary | None = None
        self.ft:    FastText | None   = None

    def load(self, model_dir: Path = MODELS_ATTENTION) -> bool:
        ft_path    = model_dir / "absa_attention_ft.model"
        vocab_path = model_dir / "absa_attention_vocab.pkl"
        pt_path    = model_dir / "absa_attention.pt"

        for p in (ft_path, vocab_path, pt_path):
            if not p.exists():
                print(f"[WARN] Не знайдено: {p}")
                print("       Спочатку запусти task_4_3_absa_attention.py")
                return False

        print(f"Завантаження FastText: {ft_path}")
        self.ft = FastText.load(str(ft_path))

        print(f"Завантаження словника: {vocab_path}")
        with open(vocab_path, "rb") as f:
            vdata = pickle.load(f)
        self.vocab = Vocabulary()
        self.vocab.stoi = vdata["stoi"]
        self.vocab.itos = vdata["itos"]

        W = np.zeros((len(self.vocab), EMB_DIM), dtype=np.float32)
        for w, i in self.vocab.stoi.items():
            try:
                W[i] = self.ft.wv[w]
            except KeyError:
                pass
        emb_layer = nn.Embedding.from_pretrained(
            torch.tensor(W), freeze=False,
            padding_idx=self.vocab.stoi.get("<PAD>", 0),
        )

        self.model = ModelWithAttention(
            vocab_size=len(self.vocab),
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            emb_layer=emb_layer,
            aspect_emb_dim=EMB_DIM,
            n_aspects=len(ASPECTS),
            dropout=DROPOUT,
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        self.model.eval()
        print("ABSA-Attention завантажено.")
        return True

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """[MAX_LEN, 1] тензор для інференсу."""
        ids = [self.vocab.stoi["<SOS>"]]
        ids += self.vocab.numericalize(text)
        ids.append(self.vocab.stoi["<EOS>"])
        if len(ids) < MAX_LEN:
            ids += [self.vocab.stoi["<PAD>"]] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(1).to(DEVICE)

    def _aspect_vec(self, aspect: str) -> torch.Tensor:
        try:
            v = self.ft.wv[aspect]
        except KeyError:
            v = np.zeros(EMB_DIM, dtype=np.float32)
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, dict]:
        """Per-aspect інференс для одного тексту."""
        tensor  = self._text_to_tensor(text)
        results = {}
        self.model.eval()
        lstm_out, last_h = self.model.encode(tensor)
        for asp_idx, asp in enumerate(ASPECTS):
            av          = self._aspect_vec(asp)
            out, alpha  = self.model.forward_aspect(lstm_out, last_h, av, asp_idx)
            score       = float(out.squeeze().cpu().item())
            attn        = alpha.squeeze(0).cpu().tolist()

            tokens    = text.lower().split()
            tok_attn  = [
                (tok, attn[i + 1])
                for i, tok in enumerate(tokens[: MAX_LEN - 2])
                if i + 1 < len(attn)
            ]
            tok_attn.sort(key=lambda x: x[1], reverse=True)

            results[asp] = {
                "label":      "positive" if score > 0.5 else "negative",
                "score":      round(score, 4),
                "top_tokens": tok_attn[:5],
            }
        return results

    @torch.no_grad()
    def eval_per_aspect(
        self,
        texts_by_aspect: dict[str, list[str]],
        labels_by_aspect: dict[str, list[int]],
    ) -> dict[str, dict]:
        """Per-aspect F1/Accuracy на наборі текстів."""
        self.model.eval()
        per: dict[str, dict] = {}
        for asp_idx, asp in enumerate(ASPECTS):
            av_cpu = self._aspect_vec(asp)
            texts = texts_by_aspect[asp]
            labels = labels_by_aspect[asp]
            preds  = []
            for text in texts:
                tensor = self._text_to_tensor(text)
                lstm_out, last_h = self.model.encode(tensor)
                out, _ = self.model.forward_aspect(lstm_out, last_h, av_cpu, asp_idx)
                preds.append(int(float(out.squeeze()) > 0.5))

            per[asp] = {
                "f1":       round(f1_score(labels, preds, average="weighted"), 4),
                "accuracy": round(accuracy_score(labels, preds), 4),
                "pos_rate": round(sum(preds) / len(preds), 4),
                "preds":    preds,
                "labels":   labels,
                "n_eval":   len(labels),
            }
            print(f"  [Attention] {asp:<20} "
                  f"F1={per[asp]['f1']:.4f}  Acc={per[asp]['accuracy']:.4f}  "
                  f"N={per[asp]['n_eval']}")
        return per

class BERTLoader:
    """Завантажує збережену BERT-ABSA модель без перенавчання."""

    def __init__(self):
        self.model     = None
        self.tokenizer = None

    def load(self, model_dir: Path = MODELS_BERT) -> bool:
        if not model_dir.exists():
            print(f"[WARN] Не знайдено: {model_dir}")
            print("       Спочатку запусти task_4_3_bert_absa.py")
            return False

        print(f"Завантаження BERT-ABSA з {model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).to(DEVICE)
        self.model.eval()
        print("BERT-ABSA завантажено.")
        return True

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, dict]:
        """Per-aspect інференс для одного тексту."""
        results = {}
        self.model.eval()
        for asp in ASPECTS:
            enc = self.tokenizer(
                text,
                ASPECT_DESCRIPTIONS[asp],
                return_tensors = "pt",
                truncation     = True,
                padding        = "max_length",
                max_length     = 128,
            )
            enc    = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
            lid    = int(logits.argmax(dim=1).item())
            results[asp] = {
                "label":      "positive" if lid == 1 else "negative",
                "score":      round(probs[lid], 4),
                "pos_prob":   round(probs[1], 4),
            }
        return results

    @torch.no_grad()
    def eval_per_aspect(
        self,
        texts_by_aspect: dict[str, list[str]],
        labels_by_aspect: dict[str, list[int]],
        batch_size: int = 32,
    ) -> dict[str, dict]:
        """Per-aspect F1/Accuracy — батчевий інференс."""
        self.model.eval()
        per: dict[str, dict] = {}
        for asp in ASPECTS:
            asp_desc = ASPECT_DESCRIPTIONS[asp]
            texts = texts_by_aspect[asp]
            labels = labels_by_aspect[asp]
            preds    = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    [asp_desc] * len(batch_texts),
                    return_tensors = "pt",
                    truncation     = True,
                    padding        = True,
                    max_length     = 128,
                )
                enc    = {k: v.to(DEVICE) for k, v in enc.items()}
                logits = self.model(**enc).logits
                preds.extend(logits.argmax(dim=1).cpu().tolist())

            per[asp] = {
                "f1":       round(f1_score(labels, preds, average="weighted"), 4),
                "accuracy": round(accuracy_score(labels, preds), 4),
                "pos_rate": round(sum(preds) / len(preds), 4),
                "preds":    preds,
                "labels":   labels,
                "n_eval":   len(labels),
            }
            print(f"  [BERT-ABSA] {asp:<20} "
                  f"F1={per[asp]['f1']:.4f}  Acc={per[asp]['accuracy']:.4f}  "
                  f"N={per[asp]['n_eval']}")
        return per

def find_disagreements(
    texts:      list[str],
    true_labels: list[int],
    attn_preds:  list[int],                                                   
    bert_preds:  list[int],                                              
    n: int = 6,
) -> dict[str, list[dict]]:
    """Знаходить приклади де моделі дають різні відповіді."""
    lmap = {0: "negative", 1: "positive"}

    def _row(i):
        return {
            "text":      texts[i][:100] + ("…" if len(texts[i]) > 100 else ""),
            "true":      lmap[true_labels[i]],
            "attn":      lmap[attn_preds[i]],
            "attn_ok":   attn_preds[i] == true_labels[i],
            "bert":      lmap[bert_preds[i]],
            "bert_ok":   bert_preds[i] == true_labels[i],
        }

    attn_wins = [
        _row(i) for i in range(len(true_labels))
        if attn_preds[i] == true_labels[i] and bert_preds[i] != true_labels[i]
    ][:n]

    bert_wins = [
        _row(i) for i in range(len(true_labels))
        if bert_preds[i] == true_labels[i] and attn_preds[i] != true_labels[i]
    ][:n]

    both_wrong = [
        _row(i) for i in range(len(true_labels))
        if attn_preds[i] != true_labels[i] and bert_preds[i] != true_labels[i]
    ][:n]

    print(f"  Attention wins={len(attn_wins)} | BERT wins={len(bert_wins)} | "
          f"Both wrong={len(both_wrong)}")
    return {
        "attention_wins": attn_wins,
        "bert_wins":      bert_wins,
        "both_wrong":     both_wrong,
    }

IMAGE_PATHS: list[str] = []
COLORS = {
    "attention": "#4C9BE8",
    "bert":      "#E87B4C",
}

def _savefig(name: str):
    p = RESULTS_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    IMAGE_PATHS.append(str(p))
    print(f"Збережено: {p}")

def plot_per_aspect_f1(
    attn_per: dict[str, dict],
    bert_per: dict[str, dict],
):
    """Згрупований bar-chart: Attention vs BERT-ABSA для кожного аспекту."""
    x     = np.arange(len(ASPECTS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "f1",       "Weighted F1 по аспектах"),
        (axes[1], "accuracy", "Accuracy по аспектах"),
    ]:
        a_vals = [attn_per[asp][metric] for asp in ASPECTS]
        b_vals = [bert_per[asp][metric] for asp in ASPECTS]

        bars_a = ax.bar(x - width / 2, a_vals, width, color=COLORS["attention"],
                        label="ABSA-Attention (v3)", edgecolor="white")
        bars_b = ax.bar(x + width / 2, b_vals, width, color=COLORS["bert"],
                        label="BERT-ABSA",           edgecolor="white")

        ax.bar_label(bars_a, fmt="%.3f", padding=2, fontsize=9)
        ax.bar_label(bars_b, fmt="%.3f", padding=2, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(ASPECTS, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)

    plt.suptitle(
        "Порівняння ABSA-Attention vs BERT-ABSA по аспектах",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _savefig("per_aspect_f1.png")

def plot_overall_metrics(
    attn_per: dict[str, dict],
    bert_per: dict[str, dict],
):
    """Radar / bar-chart загальних метрик."""
    a_f1   = round(float(np.mean([v["f1"]       for v in attn_per.values()])), 4)
    b_f1   = round(float(np.mean([v["f1"]       for v in bert_per.values()])), 4)
    a_acc  = round(float(np.mean([v["accuracy"] for v in attn_per.values()])), 4)
    b_acc  = round(float(np.mean([v["accuracy"] for v in bert_per.values()])), 4)

    metrics = ["F1 (mean)", "Accuracy (mean)"]
    a_vals  = [a_f1, a_acc]
    b_vals  = [b_f1, b_acc]
    x       = np.arange(len(metrics))
    width   = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a  = ax.bar(x - width / 2, a_vals, width, color=COLORS["attention"],
                     label="ABSA-Attention (v3)", edgecolor="white")
    bars_b  = ax.bar(x + width / 2, b_vals, width, color=COLORS["bert"],
                     label="BERT-ABSA",           edgecolor="white")

    ax.bar_label(bars_a, fmt="%.4f", padding=3, fontsize=12, fontweight="bold")
    ax.bar_label(bars_b, fmt="%.4f", padding=3, fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Загальне порівняння: ABSA-Attention vs BERT-ABSA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4, label="80% baseline")

    winner = "BERT-ABSA" if b_f1 > a_f1 else "ABSA-Attention"
    delta  = abs(b_f1 - a_f1)
    ax.set_xlabel(f"Переможець за F1: {winner}  (Δ={delta:.4f})",
                  fontsize=11, color="#333", labelpad=10)
    plt.tight_layout()
    _savefig("overall_metrics.png")

    return a_f1, b_f1

def plot_confusion_matrices(
    attn_per: dict[str, dict],
    bert_per: dict[str, dict],
):
    """Матриці помилок для primary аспекту (content_quality) кожної моделі."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, per, title, cmap in [
        (axes[0], attn_per, "ABSA-Attention [content_quality]", "Blues"),
        (axes[1], bert_per, "BERT-ABSA [content_quality]",      "Oranges"),
    ]:
        r  = per["content_quality"]
        cm = confusion_matrix(r["labels"], r["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Neg", "Pos"],
                    yticklabels=["Neg", "Pos"],
                    linewidths=0.5)
        ax.set_title(f"{title}\nF1={r['f1']:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.suptitle("Матриці помилок (content_quality аспект)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("confusion_matrices.png")

def plot_demo_comparison(demo_rows: list[dict]):
    labels = []
    attn_scores = []
    bert_scores = []
    notes = []

    for i, r in enumerate(demo_rows, start=1):
        text = str(r["text"]).replace("\n", " ")
        short = text[:74] + "..." if len(text) > 74 else text
        labels.append(f"{i}. {short}")

        attn_ok = [r[f"attn_{asp}_label"] == r[f"expected_{asp}"] for asp in ASPECTS]
        bert_ok = [r[f"bert_{asp}_label"] == r[f"expected_{asp}"] for asp in ASPECTS]
        attn_scores.append(sum(attn_ok))
        bert_scores.append(sum(bert_ok))

        attn_marks = "".join("+" if ok else "-" for ok in attn_ok)
        bert_marks = "".join("+" if ok else "-" for ok in bert_ok)
        notes.append(f"Attn {attn_marks} | BERT {bert_marks}")

    y = np.arange(len(demo_rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(13, max(7, len(demo_rows) * 0.62)))

    bars_a = ax.barh(
        y + height / 2,
        attn_scores,
        height,
        color=COLORS["attention"],
        edgecolor="white",
        label="ABSA-Attention",
    )
    bars_b = ax.barh(
        y - height / 2,
        bert_scores,
        height,
        color=COLORS["bert"],
        edgecolor="white",
        label="BERT-ABSA",
    )

    for bars, scores in [(bars_a, attn_scores), (bars_b, bert_scores)]:
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{score}/3",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    for i, note in enumerate(notes):
        ax.text(3.35, i, note, va="center", fontsize=8, color="#555")

    attn_total = int(sum(attn_scores))
    bert_total = int(sum(bert_scores))
    total = len(demo_rows) * len(ASPECTS)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.35)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Кількість правильно визначених аспектів з 3")
    ax.set_title(
        "Демо-коментарі: скільки аспектів модель визначила правильно\n"
        f"ABSA-Attention: {attn_total}/{total} | BERT-ABSA: {bert_total}/{total} "
        "(+ = правильно, - = помилка для content/clarity/difficulty)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    _savefig("demo_comparison.png")

def plot_disagreements(disagreements: dict[str, list[dict]]):
    """Bar-chart кількості розходжень по категоріях."""
    cats   = ["Attention wins\n(Attn✓, BERT✗)",
              "BERT wins\n(BERT✓, Attn✗)",
              "Обидві\nпомиляються"]
    counts = [
        len(disagreements["attention_wins"]),
        len(disagreements["bert_wins"]),
        len(disagreements["both_wrong"]),
    ]
    colors = [COLORS["attention"], COLORS["bert"], "#999"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cats, counts, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.35 + 1)
    ax.set_ylabel("Кількість прикладів з val set")
    ax.set_title("Аналіз розходжень (content_quality аспект)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig("disagreements.png")

def _ok_span(ok: bool, text: str) -> str:
    c = "#27ae60" if ok else "#e74c3c"
    i = "✓" if ok else "✗"
    return f'<span style="color:{c};font-weight:bold">{i} {text}</span>'

def _disagreement_html(title: str, rows: list[dict], note: str) -> str:
    if not rows:
        return ""
    trs = "".join(
        f"""<tr>
          <td style="font-style:italic;font-size:13px">"{r['text']}"</td>
          <td style="text-align:center;font-weight:bold">{r['true']}</td>
          <td style="text-align:center">{_ok_span(r['attn_ok'], r['attn'])}</td>
          <td style="text-align:center">{_ok_span(r['bert_ok'], r['bert'])}</td>
        </tr>"""
        for r in rows
    )
    return f"""
    <h3>{title}</h3>
    <p style="color:#555;font-size:13px">{note}</p>
    <table>
      <tr><th>Відгук</th><th>Правильно</th>
          <th>ABSA-Attention</th><th>BERT-ABSA</th></tr>
      {trs}
    </table>"""

def build_html_report(
    attn_per:      dict[str, dict],
    bert_per:      dict[str, dict],
    attn_f1_mean:  float,
    bert_f1_mean:  float,
    demo_rows:     list[dict],
    disagreements: dict[str, list[dict]],
):
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    winner   = "BERT-ABSA" if bert_f1_mean > attn_f1_mean else "ABSA-Attention (v3)"
    delta    = bert_f1_mean - attn_f1_mean
    d_color  = "#27ae60" if delta >= 0 else "#e74c3c"
    d_sign   = "+" if delta >= 0 else ""

    asp_rows = ""
    for asp in ASPECTS:
        a = attn_per[asp]
        b = bert_per[asp]
        diff = b["f1"] - a["f1"]
        better = "BERT" if diff > 0.001 else ("Attention" if diff < -0.001 else "Рівно")
        b_color = "#27ae60" if diff > 0.001 else ("#e74c3c" if diff < -0.001 else "#888")
        asp_rows += f"""
        <tr>
          <td><b>{asp}</b></td>
          <td>{a['f1']:.4f}</td><td>{a['accuracy']:.4f}</td>
          <td>{b['f1']:.4f}</td><td>{b['accuracy']:.4f}</td>
          <td style="color:{b_color};font-weight:bold">{d_sign if diff>=0 else ''}{diff:.4f} → {better}</td>
        </tr>"""

    demo_cards = ""
    for r in demo_rows:
        demo_cards += f"""
        <div class="card">
          <div class="review">&ldquo;{r['text']}&rdquo;</div>
          <span class="badge pos">Очікування по аспектах:
            content={r['expected_content_quality']},
            clarity={r['expected_clarity']},
            difficulty={r['expected_difficulty']}
          </span>
          <table class="inner">
            <tr>
              <th>Модель / Аспект</th><th>Очікується</th><th>Прогноз</th><th>Score</th>
            </tr>"""
        for asp in ASPECTS:
            exp = r[f"expected_{asp}"]
            al = r[f"attn_{asp}_label"]
            ac = r[f"attn_{asp}_score"]
            bg = "#c8f7c5" if al == exp else "#f7c5c5"
            demo_cards += f"""
            <tr style="background:{bg}">
              <td>Attention [{asp}]</td>
              <td>{exp}</td>
              <td>{'✓' if al == exp else '✗'} {al}</td>
              <td>{ac:.3f}</td>
            </tr>"""
        for asp in ASPECTS:
            exp = r[f"expected_{asp}"]
            bl = r[f"bert_{asp}_label"]
            bc = r[f"bert_{asp}_score"]
            bg = "#c8f7c5" if bl == exp else "#f7c5c5"
            demo_cards += f"""
            <tr style="background:{bg}">
              <td>BERT-ABSA [{asp}]</td>
              <td>{exp}</td>
              <td>{'✓' if bl == exp else '✗'} {bl}</td>
              <td>{bc:.3f}</td>
            </tr>"""
        demo_cards += "</table></div>"

    imgs_html = "".join(
        f'<div class="img-block"><img src="{Path(p).name}"><p>{Path(p).stem.replace("_"," ").title()}</p></div>'
        for p in IMAGE_PATHS
    )

    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>Порівняння ABSA-Attention vs BERT-ABSA</title>
  <style>
    body    {{font-family:"Segoe UI",Arial,sans-serif;max-width:1200px;
              margin:0 auto;padding:24px;color:#222;background:#f7f9fc}}
    h1      {{color:#1a3a6b;border-bottom:3px solid #1a3a6b;padding-bottom:8px}}
    h2      {{color:#2c5f9e;margin-top:32px}}
    h3      {{color:#34607a;margin-top:20px}}
    table   {{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
    th,td   {{border:1px solid #ccc;padding:8px 12px;text-align:left;vertical-align:top}}
    th      {{background:#2c5f9e;color:#fff}}
    tr:nth-child(even) {{background:#eef2f8}}
    .delta  {{font-size:22px;font-weight:bold;color:{d_color}}}
    .gallery{{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}}
    .img-block{{text-align:center}}
    .img-block img{{max-width:540px;border:1px solid #ccc;border-radius:6px}}
    .img-block p{{font-size:12px;color:#666;margin:4px 0}}
    .card   {{background:white;border-radius:10px;padding:20px;margin:15px 0;
              box-shadow:0 2px 8px rgba(0,0,0,.1)}}
    .review {{font-style:italic;font-size:1.05em;margin-bottom:10px;
              border-left:4px solid #3498db;padding-left:12px}}
    .badge  {{display:inline-block;padding:4px 12px;border-radius:20px;
              font-weight:bold;margin-bottom:10px}}
    .pos    {{background:#27ae60;color:white}}
    .neg    {{background:#e74c3c;color:white}}
    .inner  {{margin-top:8px;font-size:12px}}
    .inner th{{background:#7f8c8d;font-size:.85em}}
    .win-box{{display:inline-block;background:white;border-radius:8px;
              padding:15px 25px;margin:8px;box-shadow:0 2px 6px rgba(0,0,0,.1);
              text-align:center}}
    .win-val{{font-size:2em;font-weight:bold;color:#3498db}}
    footer  {{margin-top:48px;font-size:12px;color:#888;
              border-top:1px solid #ccc;padding-top:10px}}
  </style>
</head>
<body>
<h1>Порівняння ABSA-Attention (v3) vs BERT-ABSA</h1>
<p>Дата: {now} | Val set: 80/20 split (seed=42, stratified)</p>

<h2>Загальний результат</h2>
<div>
  <div class="win-box">
    <div class="win-val">{attn_f1_mean:.4f}</div>
    <div>ABSA-Attention (v3)<br><small>Bi-LSTM + Additive Att + DK</small></div>
  </div>
  <div class="win-box">
    <div class="win-val">{bert_f1_mean:.4f}</div>
    <div>BERT-ABSA<br><small>XLM-RoBERTa sentence-pair</small></div>
  </div>
  <div class="win-box">
    <div class="win-val delta">{d_sign}{delta:.4f}</div>
    <div>Δ F1 (BERT − Attention)<br><small>Переможець: <b>{winner}</b></small></div>
  </div>
</div>

<h2>Per-aspect метрики</h2>
<table>
  <tr>
    <th>Аспект</th>
    <th>Attention F1</th><th>Attention Acc</th>
    <th>BERT-ABSA F1</th><th>BERT-ABSA Acc</th>
    <th>Δ F1 (BERT − Att)</th>
  </tr>
  {asp_rows}
</table>
<img src="per_aspect_f1.png" style="max-width:100%;border-radius:6px;margin:12px 0">

<h2>Загальні метрики (усереднені по аспектах)</h2>
<img src="overall_metrics.png" style="max-width:600px;border-radius:6px">

<h2>Матриці помилок (content_quality)</h2>
<img src="confusion_matrices.png" style="max-width:100%;border-radius:6px">

<h2>Аналіз розходжень на val set</h2>
<p>Ситуації де моделі дають різні відповіді для аспекту <b>content_quality</b>
(обраний як primary — найбільш загальний аспект).</p>
<img src="disagreements.png" style="max-width:600px;border-radius:6px">
{_disagreement_html(
    "ABSA-Attention правильно, BERT-ABSA помиляється",
    disagreements["attention_wins"],
    "Приклади де FastText Domain Knowledge + аспектна Gate дає перевагу."
)}
{_disagreement_html(
    "BERT-ABSA правильно, ABSA-Attention помиляється",
    disagreements["bert_wins"],
    "Приклади де глибоке розуміння контексту RoBERTa виграє над LSTM+DK."
)}
{_disagreement_html(
    "Обидві моделі помиляються",
    disagreements["both_wrong"],
    "Найважчі приклади — неоднозначний або прихований сентимент."
)}

<h2>Демо-відгуки (8 прикладів)</h2>
<img src="demo_comparison.png" style="max-width:100%;border-radius:6px;margin-bottom:16px">
{demo_cards}

<h2>Графіки</h2>
<div class="gallery">{imgs_html}</div>

<h2>Порівняльний висновок</h2>
<ul>
  <li><b>ABSA-Attention (v3)</b> — Bi-LSTM + Additive Attention + Residual + Aspect Gate +
  Domain Knowledge Ontology. Переваги: пояснювані ваги уваги (top tokens),
  ієрархічний domain knowledge, легкий і швидкий у навчанні.
  Обмеження: якість FastText-ембедінгів для рідкісних мов (UA/RU),
  фіксований розмір словника, чутливість до довжини тексту.</li>

  <li><b>BERT-ABSA</b> — XLM-RoBERTa sentence-pair classification.
  Переваги: мультилінгвальний токенайзер (понад 100 мов),
  self-attention покриває довгий контекст, аспект у вхідному рядку —
  найбільш пряма ABSA-постановка. Обмеження: повільне навчання,
  без явних ваг уваги по словах, потребує більше пам'яті.</li>
</ul>

<footer>Згенеровано: task_4_3_compare.py | {now}</footer>
</body>
</html>"""

    p = RESULTS_DIR / "comparison_report.html"
    p.write_text(html, encoding="utf-8")
    print(f"HTML-звіт збережено: {p}")
    return str(p)

def main():
    print("=" * 60)
    print("Порівняння ABSA-Attention (v3) vs BERT-ABSA")
    print("=" * 60)

    print(f"\nЗавантаження: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    if "y" not in df.columns:
        df["y"] = (df["Label"].astype(str).str.lower() == "positive").astype(int)

    _, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
    val_df    = val_df.reset_index(drop=True)
    if VAL_SAMPLE_SIZE and len(val_df) > VAL_SAMPLE_SIZE:
        val_df = val_df.sample(n=VAL_SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
        print(f"Для швидкого порівняння використано sample val: {VAL_SAMPLE_SIZE} рядків")
    val_texts  = val_df["clean_text"].tolist()
    val_labels = val_df["y"].tolist()
    print(f"Val: {len(val_df)} рядків | "
          f"Pos={sum(val_labels)} | Neg={len(val_labels)-sum(val_labels)}")

    texts_by_aspect: dict[str, list[str]] = {}
    labels_by_aspect: dict[str, list[int]] = {}
    for asp in ASPECTS:
        col = f"label_{asp}"
        if col not in val_df.columns:
            raise ValueError(
                f"У датасеті немає колонки {col}. "
                "Для чесної ABSA-оцінки потрібен processed_reviews_labeled.csv."
            )
        asp_df = val_df[val_df[col].astype(str).str.lower().isin(["positive", "negative"])].copy()
        texts_by_aspect[asp] = asp_df["clean_text"].tolist()
        labels_by_aspect[asp] = (
            asp_df[col].astype(str).str.lower().eq("positive").astype(int).tolist()
        )
        pos = sum(labels_by_aspect[asp])
        total = len(labels_by_aspect[asp])
        print(f"  {asp:<20}: N={total} | Pos={pos} | Neg={total - pos}")

    print("\n" + "=" * 60)
    attn_loader = AttentionLoader()
    has_attn    = attn_loader.load()

    bert_loader = BERTLoader()
    has_bert    = bert_loader.load()

    if not has_attn and not has_bert:
        print("\n[ERROR] Жодна модель не знайдена. Спочатку навчіть моделі:")
        print("  python task_4_3_absa_attention.py")
        print("  python task_4_3_bert_absa.py")
        return

    attn_per: dict[str, dict] = {}
    bert_per: dict[str, dict] = {}

    if has_attn:
        print("\n========== ABSA-Attention: per-aspect ==========")
        attn_per = attn_loader.eval_per_aspect(texts_by_aspect, labels_by_aspect)

    if has_bert:
        print("\n========== BERT-ABSA: per-aspect ==========")
        bert_per = bert_loader.eval_per_aspect(texts_by_aspect, labels_by_aspect)

    records = []
    for asp in ASPECTS:
        if asp in attn_per:
            records.append({"model": "ABSA-Attention", "aspect": asp,
                            **{k: v for k, v in attn_per[asp].items()
                               if k not in ("preds", "labels")}})
        if asp in bert_per:
            records.append({"model": "BERT-ABSA", "aspect": asp,
                            **{k: v for k, v in bert_per[asp].items()
                               if k not in ("preds", "labels")}})
    pd.DataFrame(records).to_csv(RESULTS_DIR / "comparison_results.csv", index=False)
    print(f"\nМетрики збережено: {RESULTS_DIR / 'comparison_results.csv'}")

    print("\n========== ДЕМО-ВІДГУКИ ==========")
    demo_rows = []
    for item in DEMO_REVIEWS:
        text = item["text"]
        expected_by_aspect = item["expected"]
        print(f'\n"{text[:70]}…"' if len(text) > 70 else f'\n"{text}"')
        row = {"text": text}
        for asp in ASPECTS:
            row[f"expected_{asp}"] = expected_by_aspect[asp]

        if has_attn:
            attn_res = attn_loader.predict(text)
            for asp, vals in attn_res.items():
                expected = expected_by_aspect[asp]
                ok = "✓" if vals["label"] == expected else "✗"
                print(f"  Attention [{asp:<20}] {vals['label']:<8} "
                      f"expected={expected:<8} score={vals['score']:.3f}  {ok}")
                row[f"attn_{asp}_label"] = vals["label"]
                row[f"attn_{asp}_score"] = vals["score"]

        if has_bert:
            bert_res = bert_loader.predict(text)
            for asp, vals in bert_res.items():
                expected = expected_by_aspect[asp]
                ok = "✓" if vals["label"] == expected else "✗"
                print(f"  BERT-ABSA  [{asp:<20}] {vals['label']:<8} "
                      f"expected={expected:<8} score={vals['score']:.3f}  {ok}")
                row[f"bert_{asp}_label"] = vals["label"]
                row[f"bert_{asp}_score"] = vals["score"]

        demo_rows.append(row)

    demo_csv = RESULTS_DIR / "demo_predictions.csv"
    pd.DataFrame(demo_rows).to_csv(demo_csv, index=False)
    print(f"\nДемо-прогнози збережено: {demo_csv}")

    disagreements: dict[str, list[dict]] = {
        "attention_wins": [], "bert_wins": [], "both_wrong": []
    }
    if has_attn and has_bert:
        print("\n========== АНАЛІЗ РОЗХОДЖЕНЬ (content_quality) ==========")
        disagreements = find_disagreements(
            texts_by_aspect["content_quality"],
            labels_by_aspect["content_quality"],
            attn_per["content_quality"]["preds"],
            bert_per["content_quality"]["preds"],
        )

    print("\nГенерую графіки...")
    if has_attn and has_bert:
        plot_per_aspect_f1(attn_per, bert_per)
        attn_f1_mean, bert_f1_mean = plot_overall_metrics(attn_per, bert_per)
        plot_confusion_matrices(attn_per, bert_per)
        plot_disagreements(disagreements)
    elif has_attn:
        attn_f1_mean = float(np.mean([v["f1"] for v in attn_per.values()]))
        bert_f1_mean = 0.0
    else:
        attn_f1_mean = 0.0
        bert_f1_mean = float(np.mean([v["f1"] for v in bert_per.values()]))

    if demo_rows and has_attn and has_bert:
        plot_demo_comparison(demo_rows)

    print(f"\n{'═'*60}")
    if has_attn:
        print(f"  ABSA-Attention (mean F1)  : {attn_f1_mean:.4f}")
    if has_bert:
        print(f"  BERT-ABSA      (mean F1)  : {bert_f1_mean:.4f}")
    if has_attn and has_bert:
        delta  = bert_f1_mean - attn_f1_mean
        winner = "BERT-ABSA" if delta > 0 else "ABSA-Attention"
        print(f"  Δ F1 (BERT − Attn)        : {'+' if delta>=0 else ''}{delta:.4f}")
        print(f"  Рекомендується             : {winner}")
    print(f"  Результати                : {RESULTS_DIR}")
    print(f"{'═'*60}")

    if has_attn and has_bert and demo_rows:
        print("\nГенерую HTML-звіт...")
        build_html_report(
            attn_per, bert_per,
            attn_f1_mean, bert_f1_mean,
            demo_rows, disagreements,
        )
    print(f"\nВсі результати у: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
