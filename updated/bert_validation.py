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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_reviews_labeled.csv"
MODEL_DIR = BASE_DIR / "saved_models_4_3_bert" / "bert_absa"
RESULTS_DIR = BASE_DIR / "results_4_4_bert_validation"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 32
VAL_SAMPLE_SIZE = 2500 if DEVICE == "cuda" else 1000
ASPECTS = ["content_quality", "clarity", "difficulty"]

ASPECT_DESCRIPTIONS = {
    "content_quality": "quality and depth of course content and materials",
    "clarity": "clarity of explanations and teaching style",
    "difficulty": "difficulty level, pace and workload of the course",
}

IMAGE_PATHS = []

def savefig(name: str):
    path = RESULTS_DIR / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    IMAGE_PATHS.append(path)
    print(f"Збережено: {path}")

class BertABSAEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str], aspect: str) -> tuple[list[int], list[float]]:
        preds = []
        probs = []
        aspect_desc = ASPECT_DESCRIPTIONS[aspect]
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            enc = self.tokenizer(
                batch_texts,
                [aspect_desc] * len(batch_texts),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.model(**enc).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            batch_preds = logits.argmax(dim=1).detach().cpu().numpy()
            probs.extend(batch_probs.tolist())
            preds.extend(batch_preds.tolist())
        return preds, probs

def load_validation_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["y"] = df["Label"].astype(str).str.lower().eq("positive").astype(int)
    _, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
    val_df = val_df.reset_index(drop=True)
    if VAL_SAMPLE_SIZE and len(val_df) > VAL_SAMPLE_SIZE:
        val_df = val_df.sample(n=VAL_SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    return val_df

def aspect_frame(val_df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    col = f"label_{aspect}"
    data = val_df[val_df[col].astype(str).str.lower().isin(["positive", "negative"])].copy()
    data["target"] = data[col].astype(str).str.lower().eq("positive").astype(int)
    return data

def evaluate(model: BertABSAEvaluator, val_df: pd.DataFrame):
    rows = []
    reports = {}
    predictions = {}
    for aspect in ASPECTS:
        data = aspect_frame(val_df, aspect)
        y_true = data["target"].tolist()
        texts = data["clean_text"].tolist()
        y_pred, pos_prob = model.predict(texts, aspect)
        predictions[aspect] = {"labels": y_true, "preds": y_pred, "probs": pos_prob, "texts": texts}
        rows.append({
            "aspect": aspect,
            "n_eval": len(y_true),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
            "positive_rate_pred": round(float(np.mean(y_pred)), 4),
            "positive_rate_true": round(float(np.mean(y_true)), 4),
        })
        report = classification_report(
            y_true,
            y_pred,
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        )
        reports[aspect] = pd.DataFrame(report).transpose()
        reports[aspect].to_csv(RESULTS_DIR / f"classification_report_{aspect}.csv", encoding="utf-8-sig")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics, reports, predictions

def plot_metrics(metrics: pd.DataFrame):
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, metrics["accuracy"], width, label="Accuracy", color="#4C9BE8")
    b2 = ax.bar(x, metrics["precision_weighted"], width, label="Precision", color="#5CB85C")
    b3 = ax.bar(x + width, metrics["f1_weighted"], width, label="F1", color="#E87B4C")
    for bars in [b1, b2, b3]:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["aspect"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("BERT-ABSA: якість класифікації по аспектах", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    savefig("bert_validation_metrics.png")

def plot_confusions(predictions: dict):
    fig, axes = plt.subplots(1, len(ASPECTS), figsize=(15, 4))
    for ax, aspect in zip(axes, ASPECTS):
        y_true = predictions[aspect]["labels"]
        y_pred = predictions[aspect]["preds"]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"],
        )
        f1 = f1_score(y_true, y_pred, average="weighted")
        ax.set_title(f"{aspect}\nF1={f1:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("BERT-ABSA: confusion matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("bert_confusion_matrices.png")

def plot_class_balance(metrics: pd.DataFrame):
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, metrics["positive_rate_true"], width, label="True positive rate", color="#7DB7E8")
    b2 = ax.bar(x + width / 2, metrics["positive_rate_pred"], width, label="Predicted positive rate", color="#F2A36B")
    ax.bar_label(b1, fmt="%.2f", padding=2)
    ax.bar_label(b2, fmt="%.2f", padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["aspect"])
    ax.set_ylim(0, 1)
    ax.set_title("Баланс класів: фактичний vs прогнозований positive rate", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    savefig("bert_class_balance.png")

def build_html(metrics: pd.DataFrame):
    rows = "".join(
        f"""
        <tr>
          <td><b>{r.aspect}</b></td>
          <td>{int(r.n_eval)}</td>
          <td>{r.accuracy:.4f}</td>
          <td>{r.precision_weighted:.4f}</td>
          <td>{r.recall_weighted:.4f}</td>
          <td><b>{r.f1_weighted:.4f}</b></td>
        </tr>
        """
        for r in metrics.itertuples()
    )
    mean_f1 = metrics["f1_weighted"].mean()
    imgs = "".join(f'<div><img src="{p.name}"><p>{p.stem}</p></div>' for p in IMAGE_PATHS)
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>4.4 Валідація BERT-ABSA</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;color:#222;background:#f7f9fc}}
    h1{{color:#1a3a6b;border-bottom:3px solid #1a3a6b;padding-bottom:8px}}
    h2{{color:#2c5f9e;margin-top:28px}}
    table{{border-collapse:collapse;width:100%;background:white;margin:14px 0}}
    th,td{{border:1px solid #ccc;padding:9px 12px;text-align:center}}
    th{{background:#2c5f9e;color:white}}
    tr:nth-child(even){{background:#eef2f8}}
    .metric{{font-size:34px;color:#3498db;font-weight:bold}}
    .gallery{{display:flex;flex-wrap:wrap;gap:16px}}
    .gallery img{{max-width:520px;border:1px solid #ccc;border-radius:6px;background:white}}
    .gallery p{{font-size:12px;color:#666;text-align:center}}
  </style>
</head>
<body>
  <h1>4.4. Валідація та оцінка якості роботи BERT-ABSA</h1>
  <p>Дата генерації: {datetime.now().strftime("%Y-%m-%d %H:%M")}. Модель завантажена з <code>{MODEL_DIR}</code>. Оцінка виконана на validation split 80/20, seed={SEED}. Для кожного аспекту neutral-відгуки виключені.</p>
  <h2>Підсумок</h2>
  <p>Середній weighted F1 по трьох аспектах:</p>
  <div class="metric">{mean_f1:.4f}</div>
  <h2>Метрики по аспектах</h2>
  <table>
    <tr><th>Аспект</th><th>N</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Weighted F1</th></tr>
    {rows}
  </table>
  <h2>Візуалізації</h2>
  <div class="gallery">{imgs}</div>
  <h2>Висновок</h2>
  <p>BERT-ABSA оцінюється окремо для кожного аспекту: якість контенту, зрозумілість і складність. Такий формат валідації коректніший за загальну мітку sentiment, бо один і той самий коментар може бути позитивним за одним аспектом і негативним за іншим.</p>
</body>
</html>"""
    path = RESULTS_DIR / "report_4_4_validation.html"
    path.write_text(html, encoding="utf-8")
    print(f"HTML-звіт: {path}")

def main():
    print(f"Device: {DEVICE}")
    print(f"Дані: {DATA_PATH}")
    print(f"Модель: {MODEL_DIR}")
    val_df = load_validation_data()
    model = BertABSAEvaluator()
    metrics, reports, predictions = evaluate(model, val_df)
    print(metrics.to_string(index=False))
    plot_metrics(metrics)
    plot_confusions(predictions)
    plot_class_balance(metrics)
    build_html(metrics)
    print(f"Усі результати: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
