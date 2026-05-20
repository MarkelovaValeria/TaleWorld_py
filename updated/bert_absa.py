import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

BASE_DIR        = Path("")
DATA_PATH       = BASE_DIR / "processed_reviews_labeled.csv"
RESULTS_DIR     = BASE_DIR / "results_4_3_bert"
MODELS_DIR      = BASE_DIR / "saved_models_4_3_bert"
SAVED_4_2_DIR   = BASE_DIR / "saved_models" / "transformer"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class Config:
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    SEED         = 42
    MODEL_NAME   = "xlm-roberta-base"
    MAX_LEN      = 128
    BATCH_SIZE   = 16
    EPOCHS       = 5
    LR           = 2e-5
    WEIGHT_DECAY = 0.01
    PATIENCE     = 2
    ASPECTS      = ["content_quality", "clarity", "difficulty"]

cfg = Config()
print(f"Device: {cfg.DEVICE}")

ASPECT_DESCRIPTIONS: dict[str, str] = {
    "content_quality": "quality and depth of course content and materials",
    "clarity":         "clarity of explanations and teaching style",
    "difficulty":      "difficulty level, pace and workload of the course",
}

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

class BERTABSATrainer:

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.history   = MetricHistoryCallback()
        self.best_f1   = None
        self._val_df   = None

    def _expand_df(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            for asp in cfg.ASPECTS:
                lbl_col = f"label_{asp}"
                if lbl_col in df.columns:
                    asp_lbl = str(row[lbl_col]).lower().strip()
                    if asp_lbl == "neutral":
                        continue
                    y = 1 if asp_lbl == "positive" else 0
                else:
                    y = int(row["y"])

                rows.append({
                    "text":        str(row["clean_text"]),
                    "aspect_desc": ASPECT_DESCRIPTIONS[asp],
                    "labels":      y,
                })
        return pd.DataFrame(rows)

    def _tokenize(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            batch["aspect_desc"],
            truncation  = True,
            padding     = "max_length",
            max_length  = cfg.MAX_LEN,
        )

    def _to_hf(self, df: pd.DataFrame) -> HFDataset:
        ds = HFDataset.from_pandas(df[["text", "aspect_desc", "labels"]])
        ds = ds.map(self._tokenize, batched=True)
        ds = ds.remove_columns(["text", "aspect_desc"])
        ds.set_format("torch")
        return ds

    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "f1":       f1_score(labels, preds, average="weighted"),
            "accuracy": float((preds == labels).mean()),
        }

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
        self._val_df = val_df.copy()

        print("Завантаження токенайзера та моделі...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODEL_NAME,
            num_labels = 2,
            id2label   = {0: "negative", 1: "positive"},
            label2id   = {"negative": 0, "positive": 1},
        )

        train_exp = self._expand_df(train_df)
        val_exp   = self._expand_df(val_df)

        print(f"Train: {len(train_df)} відгуків → {len(train_exp)} прикладів "
              f"(neutral відфільтровано)")
        print(f"Val  : {len(val_df)} відгуків → {len(val_exp)} прикладів")

        pos = (train_exp["labels"] == 1).sum()
        neg = (train_exp["labels"] == 0).sum()
        print(f"Train label dist: positive={pos} negative={neg}")

        train_hf = self._to_hf(train_exp)
        val_hf   = self._to_hf(val_exp)

        training_args = TrainingArguments(
            output_dir                  = str(MODELS_DIR / "checkpoints"),
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            learning_rate               = cfg.LR,
            per_device_train_batch_size = cfg.BATCH_SIZE,
            per_device_eval_batch_size  = cfg.BATCH_SIZE,
            num_train_epochs            = cfg.EPOCHS,
            weight_decay                = cfg.WEIGHT_DECAY,
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            greater_is_better           = True,
            logging_steps               = 50,
            fp16                        = torch.cuda.is_available(),
            dataloader_num_workers      = 0,
            dataloader_pin_memory       = False,
            report_to                   = "none",
        )

        self.history = MetricHistoryCallback()

        trainer = Trainer(
            model            = self.model,
            args             = training_args,
            train_dataset    = train_hf,
            eval_dataset     = val_hf,
            processing_class = self.tokenizer,
            compute_metrics  = self._compute_metrics,
            callbacks        = [
                self.history,
                EarlyStoppingCallback(early_stopping_patience=cfg.PATIENCE),
            ],
        )

        print("\nНавчання BERT-ABSA...")
        trainer.train()

        pred_out = trainer.predict(val_hf)
        preds    = pred_out.predictions.argmax(axis=1).tolist()
        labels   = pred_out.label_ids.tolist()
        self.best_f1 = f1_score(labels, preds, average="weighted")

        print(f"\nBERT-ABSA — Weighted F1: {self.best_f1:.4f}")
        print(classification_report(labels, preds,
                                    target_names=["Negative", "Positive"]))

        save_path = MODELS_DIR / "bert_absa"
        trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"Модель збережена: {save_path}")

        self.model = trainer.model
        return self.best_f1

    def eval_per_aspect(self) -> dict[str, dict]:
        assert self._val_df is not None, "Спочатку запусти fit()"
        self.model.eval()
        self.model.to(cfg.DEVICE)

        results = {}
        for asp in cfg.ASPECTS:
            asp_desc = ASPECT_DESCRIPTIONS[asp]
            lbl_col  = f"label_{asp}"

            if lbl_col in self._val_df.columns:
                df_asp = self._val_df[
                    self._val_df[lbl_col].str.lower() != "neutral"
                ].copy()
                df_asp["labels"] = (
                    df_asp[lbl_col].str.lower() == "positive"
                ).astype(int)
            else:
                df_asp = self._val_df.copy()
                df_asp["labels"] = df_asp["y"].astype(int)

            if len(df_asp) == 0:
                print(f"  {asp}: немає non-neutral рядків — пропускаємо")
                continue

            df_asp["text"]        = df_asp["clean_text"].astype(str)
            df_asp["aspect_desc"] = asp_desc

            hf = self._to_hf(df_asp[["text", "aspect_desc", "labels"]])

            all_preds, all_labels = [], []
            with torch.no_grad():
                for i in range(0, len(hf), cfg.BATCH_SIZE * 2):
                    batch          = hf[i: i + cfg.BATCH_SIZE * 2]
                    input_ids      = torch.tensor(batch["input_ids"]).to(cfg.DEVICE)
                    attention_mask = torch.tensor(batch["attention_mask"]).to(cfg.DEVICE)
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                    preds  = logits.argmax(dim=1).cpu().tolist()
                    lbls   = (batch["labels"].tolist()
                              if hasattr(batch["labels"], "tolist")
                              else list(batch["labels"]))
                    all_preds.extend(preds)
                    all_labels.extend(lbls)

            f1  = f1_score(all_labels, all_preds, average="weighted")
            acc = accuracy_score(all_labels, all_preds)
            pos = sum(all_preds) / len(all_preds) if all_preds else 0

            results[asp] = {
                "f1":            round(f1,  4),
                "accuracy":      round(acc, 4),
                "positive_rate": round(pos, 4),
                "preds":         all_preds,
                "labels":        all_labels,
            }
            print(f"  {asp:<20} F1={f1:.4f}  Acc={acc:.4f}  "
                  f"Pos%={pos:.1%}  n={len(all_labels)}")

        return results

    def analyze_review(self, text: str) -> dict[str, dict]:
        self.model.eval()
        self.model.to(cfg.DEVICE)

        results = {}
        for asp in cfg.ASPECTS:
            enc = self.tokenizer(
                text,
                ASPECT_DESCRIPTIONS[asp],
                return_tensors = "pt",
                truncation     = True,
                padding        = "max_length",
                max_length     = cfg.MAX_LEN,
            )
            enc = {k: v.to(cfg.DEVICE) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits

            probs    = torch.softmax(logits, dim=1)[0].cpu().tolist()
            label_id = int(logits.argmax(dim=1).item())
            label    = "positive" if label_id == 1 else "negative"
            conf     = probs[label_id]

            results[asp] = {
                "label":      label,
                "confidence": round(conf, 3),
                "pos_prob":   round(probs[1], 3),
                "neg_prob":   round(probs[0], 3),
            }
        return results

class RegularBERTEvaluator:

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.loaded    = False

    def load(self) -> bool:
        if not SAVED_4_2_DIR.exists():
            print(f"[WARN] Модель 4.2 не знайдена: {SAVED_4_2_DIR}")
            return False
        print(f"Завантаження загального RoBERTa з {SAVED_4_2_DIR} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(SAVED_4_2_DIR))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(SAVED_4_2_DIR)
        ).to(cfg.DEVICE)
        self.model.eval()
        self.loaded = True
        print("Завантажено.")
        return True

    def predict(self, text: str) -> tuple[str, float]:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128,
        )
        enc = {k: v.to(cfg.DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs    = torch.softmax(logits, dim=1)[0].cpu().tolist()
        label_id = int(logits.argmax(dim=1).item())
        return ("positive" if label_id == 1 else "negative"), round(probs[label_id], 3)

    def eval_on_val(self, val_df: pd.DataFrame) -> tuple[float, list, list]:
        preds, labels = [], []
        for _, row in val_df.iterrows():
            lbl, _ = self.predict(str(row["clean_text"]))
            preds.append(1 if lbl == "positive" else 0)
            labels.append(int(row["y"]))
        f1 = f1_score(labels, preds, average="weighted")
        return f1, preds, labels

def _savefig(name: str):
    path = RESULTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Збережено: {path}")

def plot_training_curves(history: MetricHistoryCallback):
    if not history.val_f1:
        return
    epochs = history.epochs or list(range(1, len(history.val_f1) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history.val_f1,   "g-o", linewidth=2, markersize=8)
    ax1.set_title("BERT-ABSA — Val F1",  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Епоха"); ax1.set_ylabel("Weighted F1")
    ax1.set_ylim(0, 1); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history.val_loss, "r-o", linewidth=2, markersize=8)
    ax2.set_title("BERT-ABSA — Val Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Епоха"); ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    plt.suptitle("Криві навчання XLM-RoBERTa ABSA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("training_curves.png")

def plot_per_aspect_f1(per_aspect_bert: dict, regular_f1: float | None):
    aspects = list(per_aspect_bert.keys())
    f1s     = [per_aspect_bert[a]["f1"] for a in aspects]
    x = np.arange(len(aspects))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, f1s, width=0.5, color="#4C9BE8",
                  label="BERT-ABSA (per-aspect)", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11, fontweight="bold")
    if regular_f1 is not None:
        ax.axhline(regular_f1, color="#E87B4C", linewidth=2.5, linestyle="--",
                   label=f"RoBERTa загальний (4.2) F1={regular_f1:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Weighted F1", fontsize=12)
    ax.set_title("F1 по аспектах: загальний RoBERTa vs BERT-ABSA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig("per_aspect_f1.png")

def plot_confusion_matrices(bert_absa_results: dict, regular_preds, regular_labels):
    n_total = len(cfg.ASPECTS) + (1 if regular_preds else 0)
    fig, axes = plt.subplots(1, n_total, figsize=(6 * n_total, 5))
    if n_total == 1:
        axes = [axes]
    col = 0
    if regular_preds:
        cm = confusion_matrix(regular_labels, regular_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes[col],
                    xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"], linewidths=0.5)
        f1 = f1_score(regular_labels, regular_preds, average="weighted")
        axes[col].set_title(f"RoBERTa загальний\nF1={f1:.4f}", fontsize=11, fontweight="bold")
        col += 1
    for asp in cfg.ASPECTS:
        if asp not in bert_absa_results:
            col += 1
            continue
        r  = bert_absa_results[asp]
        cm = confusion_matrix(r["labels"], r["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[col],
                    xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"], linewidths=0.5)
        axes[col].set_title(f"BERT-ABSA [{asp}]\nF1={r['f1']:.4f}",
                            fontsize=11, fontweight="bold")
        col += 1
    plt.suptitle("Матриці помилок — RoBERTa загальний vs BERT-ABSA",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("confusion_matrices.png")

def plot_demo_table(rows: list[dict]):
    has_regular = any(r.get("regular_label") for r in rows)
    col_labels  = ["Текст (скорочено)", "Очікується"]
    if has_regular:
        col_labels.append("RoBERTa (4.2)")
    for a in cfg.ASPECTS:
        col_labels.append(f"BERT-ABSA\n[{a}]")

    def _color(pred, expected):
        return "#A8D5A2" if pred == expected else "#F4A0A0"
    def _fmt(label, conf):
        return f"{label}\n({conf:.0%})"

    cell_text, cell_colors = [], []
    for r in rows:
        short = (r["text"][:38] + "…") if len(r["text"]) > 38 else r["text"]
        exp   = r["expected"]
        row_t = [short, exp]
        row_c = ["#F5F5F5", "#DDEEFF"]
        if has_regular:
            rl = r.get("regular_label", "—")
            rc = r.get("regular_conf", 0.0)
            row_t.append(_fmt(rl, rc))
            row_c.append(_color(rl, exp))
        for a in cfg.ASPECTS:
            al = r.get(f"bert_{a}_label", "—")
            ac = r.get(f"bert_{a}_conf",  0.0)
            row_t.append(_fmt(al, ac))
            row_c.append(_color(al, exp))
        cell_text.append(row_t)
        cell_colors.append(row_c)

    fig, ax = plt.subplots(figsize=(4 * len(col_labels), max(4, len(rows) * 1.2 + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Демо: RoBERTa загальний (4.2) vs BERT-ABSA (4.3)\n"
                 "Зелений = правильно  |  Червоний = помилково",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    _savefig("demo_table.png")

def build_html_report(per_aspect, bert_f1, regular_f1, demo_rows, history):
    asp_rows = ""
    for asp, m in per_aspect.items():
        delta = ""
        if regular_f1 is not None:
            d = m["f1"] - regular_f1
            sign  = "+" if d >= 0 else ""
            color = "#27ae60" if d >= 0 else "#e74c3c"
            delta = f'<span style="color:{color};font-weight:bold">{sign}{d:.4f}</span>'
        asp_rows += f"""
        <tr>
          <td><b>{asp}</b></td><td>{m['f1']:.4f}</td>
          <td>{m['accuracy']:.4f}</td><td>{m['positive_rate']:.1%}</td>
          <td>{delta}</td>
        </tr>"""

    history_rows = "".join(
        f"<tr><td>{ep}</td><td>{f1:.4f}</td><td>{loss:.4f}</td></tr>"
        for ep, f1, loss in zip(history.epochs, history.val_f1, history.val_loss)
    )

    demo_cards = ""
    for r in demo_rows:
        demo_cards += f'''
        <div class="card">
          <div class="review-text">"{r['text']}"</div>
          <span class="badge {'pos' if r['expected']=='positive' else 'neg'}">
            Очікується: {r['expected']}
          </span>
          <table class="asp-table">
            <tr><th>Модель / Аспект</th><th>Прогноз</th><th>Впевненість</th></tr>'''
        if r.get("regular_label"):
            bg = "#c8f7c5" if r["regular_label"] == r["expected"] else "#f7c5c5"
            demo_cards += f'''
            <tr style="background:{bg}">
              <td><i>RoBERTa загальний (4.2)</i></td>
              <td>{r['regular_label']}</td><td>{r['regular_conf']:.1%}</td>
            </tr>'''
        for asp in cfg.ASPECTS:
            lbl  = r.get(f"bert_{asp}_label", "—")
            conf = r.get(f"bert_{asp}_conf",  0.0)
            bg   = "#c8f7c5" if lbl == r["expected"] else "#f7c5c5"
            demo_cards += f'''
            <tr style="background:{bg}">
              <td>BERT-ABSA [{asp}]</td>
              <td>{lbl}</td><td>{conf:.1%}</td>
            </tr>'''
        demo_cards += "</table></div>"

    reg_row = (f'''<tr><td>XLM-RoBERTa загальний (4.2)</td>
               <td>{regular_f1:.4f}</td><td>Без урахування аспектів</td></tr>''')\
              if regular_f1 else ""

    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>BERT-ABSA — Звіт</title>
  <style>
    body {{font-family:sans-serif;max-width:1100px;margin:40px auto;background:#f8f9fa;color:#333;padding:20px}}
    h1 {{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
    h2 {{color:#34495e;margin-top:30px}}
    table {{border-collapse:collapse;width:100%;margin:15px 0}}
    th,td {{border:1px solid #ddd;padding:10px;text-align:center}}
    th {{background:#2c3e50;color:white}}
    tr:nth-child(even) {{background:#f9f9f9}}
    .card {{background:white;border-radius:10px;padding:20px;margin:15px 0;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
    .review-text {{font-style:italic;font-size:1.05em;margin-bottom:10px;border-left:4px solid #3498db;padding-left:12px}}
    .badge {{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:bold;margin-bottom:10px}}
    .pos {{background:#27ae60;color:white}} .neg {{background:#e74c3c;color:white}}
    .asp-table th {{background:#7f8c8d;font-size:.9em}}
    .metric-box {{display:inline-block;background:white;border-radius:8px;padding:15px 25px;margin:8px;box-shadow:0 2px 6px rgba(0,0,0,.1);text-align:center}}
    .metric-val {{font-size:2em;font-weight:bold;color:#3498db}}
    img {{max-width:100%;border-radius:8px;margin:10px 0}}
  </style>
</head>
<body>
  <h1>BERT-ABSA — Звіт (XLM-RoBERTa Sentence-Pair)</h1>
  <p>Згенеровано: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  <h2>Підхід</h2>
  <p>Кожен відгук + опис аспекту подається як sentence-pair.
  Тренувальні мітки — per-aspect pseudo-labels з auto_label.py
  (sentence-level pseudo-labeling через претреновану мультимовну модель).
  Рядки з міткою <b>neutral</b> виключаються з навчання.</p>
  <h2>Загальні метрики</h2>
  <div>
    <div class="metric-box"><div class="metric-val">{bert_f1:.4f}</div><div>BERT-ABSA F1</div></div>
    {'<div class="metric-box"><div class="metric-val">' + f"{regular_f1:.4f}" + '</div><div>RoBERTa загальний F1</div></div>' if regular_f1 else ""}
  </div>
  <h2>Порівняння моделей</h2>
  <table>
    <tr><th>Модель</th><th>F1 (weighted)</th><th>Примітка</th></tr>
    {reg_row}
    <tr><td><b>BERT-ABSA (sentence-pair)</b></td><td><b>{bert_f1:.4f}</b></td><td>Per-aspect мітки</td></tr>
  </table>
  <h2>F1 по аспектах (BERT-ABSA)</h2>
  <table>
    <tr><th>Аспект</th><th>F1</th><th>Accuracy</th><th>Pos%</th><th>Δ vs загальний</th></tr>
    {asp_rows}
  </table>
  <img src="per_aspect_f1.png">
  <h2>Криві навчання</h2>
  <img src="training_curves.png">
  <h2>Матриці помилок</h2>
  <img src="confusion_matrices.png">
  <h2>Таблиця навчання</h2>
  <table><tr><th>Епоха</th><th>Val F1</th><th>Val Loss</th></tr>{history_rows}</table>
  <h2>Демо-відгуки</h2>
  <img src="demo_table.png">
  {demo_cards}
</body></html>"""

    path = RESULTS_DIR / "report_bert_absa.html"
    path.write_text(html, encoding="utf-8")
    print(f"HTML-звіт збережено: {path}")
    return str(path)

DEMO_REVIEWS = [
    ("The course material is very comprehensive and well-structured with great examples.", "positive"),
    ("The instructor explains concepts in a very confusing and monotone way.",              "negative"),
    ("The pace is overwhelming, assignments pile up and there is no time to understand.",  "negative"),
    ("Excellent depth of material but way too advanced for beginners, very steep curve.",  "negative"),
    ("Very clear explanations with real-world examples, easy to follow along.",            "positive"),
    ("Workload is reasonable and the difficulty level is just right for intermediate learners.", "positive"),
    ("Чудовий матеріал, але викладач пояснює нечітко і занадто швидко.",                  "negative"),
    ("Завдання надто складні і незрозумілі, темп курсу надмірний для початківців.",       "negative"),
]

if __name__ == "__main__":

    torch.manual_seed(cfg.SEED)

    print("=" * 60)
    print("BERT-ABSA — XLM-RoBERTa Sentence-Pair + Per-Aspect Labels")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    if "y" not in df.columns:
        df["y"] = (df["Label"].astype(str).str.lower() == "positive").astype(int)

    print(f"Датасет: {len(df)} рядків | Pos: {df['y'].sum()} | Neg: {(df['y']==0).sum()}")

    asp_cols = [f"label_{a}" for a in cfg.ASPECTS]
    has_asp  = all(c in df.columns for c in asp_cols)
    if has_asp:
        print("Per-aspect мітки знайдено:", asp_cols)
    else:
        print("[WARN] Per-aspect міток немає — використовується загальний Label")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=cfg.SEED, stratify=df["y"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    print("\n" + "=" * 60)
    print("БАЗОВА ЛІНІЯ: XLM-RoBERTa загальний (task_4_2)")
    print("=" * 60)
    regular_bert   = RegularBERTEvaluator()
    has_regular    = regular_bert.load()
    regular_f1     = None
    regular_preds  = None
    regular_labels = None
    if has_regular:
        regular_f1, regular_preds, regular_labels = regular_bert.eval_on_val(val_df)
        print(f"RoBERTa загальний — Val F1: {regular_f1:.4f}")

    print("\n" + "=" * 60)
    print("BERT-ABSA: XLM-RoBERTa Sentence-Pair + Per-Aspect Labels")
    print("=" * 60)
    bert_absa = BERTABSATrainer()
    bert_f1   = bert_absa.fit(train_df, val_df)

    print("\n========== PER-ASPECT МЕТРИКИ ==========")
    per_aspect = bert_absa.eval_per_aspect()

    print("\n========== ДЕМО-ВІДГУКИ ==========")
    demo_rows = []
    for text, expected in DEMO_REVIEWS:
        print(f'\n"{text}"')
        row = {"text": text, "expected": expected}
        if has_regular:
            reg_lbl, reg_conf = regular_bert.predict(text)
            ok = "✓" if reg_lbl == expected else "✗"
            print(f"  RoBERTa (4.2) | {reg_lbl:<8} | conf={reg_conf:.3f}  {ok}")
            row["regular_label"] = reg_lbl
            row["regular_conf"]  = reg_conf
        absa_res = bert_absa.analyze_review(text)
        for asp, vals in absa_res.items():
            ok = "✓" if vals["label"] == expected else "✗"
            print(f"  BERT-ABSA [{asp:<16}] | {vals['label']:<8} | conf={vals['confidence']:.3f}  {ok}")
            row[f"bert_{asp}_label"] = vals["label"]
            row[f"bert_{asp}_conf"]  = vals["confidence"]
        demo_rows.append(row)

    records = []
    for asp, m in per_aspect.items():
        records.append({"model":"BERT-ABSA","aspect":asp,"f1":m["f1"],
                        "accuracy":m["accuracy"],"pos_rate":m["positive_rate"]})
    if regular_f1:
        for asp in cfg.ASPECTS:
            records.append({"model":"RoBERTa-general","aspect":asp,
                            "f1":round(regular_f1,4),"accuracy":round(regular_f1,4),"pos_rate":None})
    pd.DataFrame(records).to_csv(RESULTS_DIR/"comparison_results.csv", index=False)
    print(f"\nМетрики збережено: {RESULTS_DIR}/comparison_results.csv")

    print("\nГенерую графіки...")
    plot_training_curves(bert_absa.history)
    plot_per_aspect_f1(per_aspect, regular_f1)
    plot_confusion_matrices(per_aspect, regular_preds, regular_labels)
    plot_demo_table(demo_rows)

    print("\nГенерую HTML-звіт...")
    build_html_report(per_aspect, bert_f1, regular_f1, demo_rows, bert_absa.history)

    print("\n" + "═"*55)
    if regular_f1:
        print(f"  RoBERTa загальний (4.2) F1 : {regular_f1:.4f}")
    print(f"  BERT-ABSA F1               : {bert_f1:.4f}")
    print(f"  Результати                 : {RESULTS_DIR}")
    print(f"  Модель                     : {MODELS_DIR}/bert_absa")
    print("═"*55)
