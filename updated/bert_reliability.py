import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "processed_reviews_labeled.csv"
MODEL_DIR = BASE_DIR / "saved_models_4_3_bert" / "bert_absa"
RESULTS_DIR = BASE_DIR / "results_5_2_bert_reliability"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 32
ROBUSTNESS_SAMPLE_SIZE = 180 if DEVICE == "cpu" else 500
STABILITY_SAMPLE_SIZE = 300 if DEVICE == "cpu" else 800
STABILITY_RUNS = 8
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

class BertABSA:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str], aspect: str) -> list[int]:
        preds = []
        aspect_desc = ASPECT_DESCRIPTIONS[aspect]
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                [aspect_desc] * len(batch),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.model(**enc).logits
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        return preds

def load_val_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["y"] = df["Label"].astype(str).str.lower().eq("positive").astype(int)
    _, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
    return val_df.reset_index(drop=True)

def aspect_df(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    col = f"label_{aspect}"
    out = df[df[col].astype(str).str.lower().isin(["positive", "negative"])].copy()
    out["target"] = out[col].astype(str).str.lower().eq("positive").astype(int)
    return out.reset_index(drop=True)

def noise_lowercase(text: str) -> str:
    return text.lower()

def noise_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", "   ", text.strip())

def noise_punctuation(text: str) -> str:
    return text + " !!! ..."

def noise_typo(text: str) -> str:
    words = text.split()
    if not words:
        return text
    idx = min(len(words) - 1, max(0, len(words) // 2))
    word = words[idx]
    if len(word) > 4:
        words[idx] = word[:-2] + word[-1]
    else:
        words[idx] = word + "x"
    return " ".join(words)

def noise_delete_word(text: str) -> str:
    words = text.split()
    if len(words) <= 4:
        return text
    idx = len(words) // 3
    return " ".join(words[:idx] + words[idx + 1:])

NOISE_FUNCTIONS = {
    "lowercase": noise_lowercase,
    "extra_spaces": noise_extra_spaces,
    "punctuation_noise": noise_punctuation,
    "minor_typo": noise_typo,
    "delete_one_word": noise_delete_word,
}

def robustness_test(model: BertABSA, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = random.Random(SEED)
    for aspect in ASPECTS:
        data = aspect_df(val_df, aspect)
        n = min(ROBUSTNESS_SAMPLE_SIZE, len(data))
        data = data.sample(n=n, random_state=SEED).reset_index(drop=True)
        texts = data["clean_text"].tolist()
        base_preds = model.predict(texts, aspect)
        for noise_name, noise_fn in NOISE_FUNCTIONS.items():
            noisy_texts = [noise_fn(t) for t in texts]
            noisy_preds = model.predict(noisy_texts, aspect)
            same = [int(a == b) for a, b in zip(base_preds, noisy_preds)]
            rows.append({
                "aspect": aspect,
                "noise_type": noise_name,
                "n": n,
                "robustness": round(float(np.mean(same)), 4),
                "changed_predictions": int(n - sum(same)),
            })
        shuffled = texts[:]
        rng.shuffle(shuffled)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "robustness_results.csv", index=False, encoding="utf-8-sig")
    return result

def stability_test(model: BertABSA, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    summary_rows = []
    for aspect in ASPECTS:
        data = aspect_df(val_df, aspect)
        for run in range(1, STABILITY_RUNS + 1):
            n = min(STABILITY_SAMPLE_SIZE, len(data))
            sample = data.sample(n=n, random_state=SEED + run).reset_index(drop=True)
            labels = sample["target"].tolist()
            preds = model.predict(sample["clean_text"].tolist(), aspect)
            run_rows.append({
                "aspect": aspect,
                "run": run,
                "n": n,
                "accuracy": round(accuracy_score(labels, preds), 4),
                "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
            })
    runs = pd.DataFrame(run_rows)
    for aspect in ASPECTS:
        sub = runs[runs["aspect"] == aspect]
        summary_rows.append({
            "aspect": aspect,
            "runs": len(sub),
            "mean_f1": round(sub["f1_weighted"].mean(), 4),
            "std_f1": round(sub["f1_weighted"].std(ddof=0), 4),
            "min_f1": round(sub["f1_weighted"].min(), 4),
            "max_f1": round(sub["f1_weighted"].max(), 4),
            "mean_accuracy": round(sub["accuracy"].mean(), 4),
            "std_accuracy": round(sub["accuracy"].std(ddof=0), 4),
        })
    summary = pd.DataFrame(summary_rows)
    runs.to_csv(RESULTS_DIR / "stability_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "stability_summary.csv", index=False, encoding="utf-8-sig")
    return runs, summary

def plot_robustness(robustness: pd.DataFrame):
    pivot = robustness.pivot(index="noise_type", columns="aspect", values="robustness").loc[list(NOISE_FUNCTIONS.keys())]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#4C9BE8", "#5CB85C", "#E87B4C"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_title("5.2 Robustness: частка незмінних прогнозів після шуму", fontsize=13, fontweight="bold")
    ax.set_ylabel("Robustness score")
    ax.set_xlabel("Тип шуму")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Aspect")
    plt.tight_layout()
    savefig("robustness_by_noise.png")

def plot_stability(runs: pd.DataFrame, summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for aspect, color in zip(ASPECTS, ["#4C9BE8", "#5CB85C", "#E87B4C"]):
        sub = runs[runs["aspect"] == aspect]
        ax.plot(sub["run"], sub["f1_weighted"], marker="o", linewidth=2, label=aspect, color=color)
    ax.set_ylim(0, 1.05)
    ax.set_title("5.2 Stability: weighted F1 на різних validation-підвибірках", fontsize=13, fontweight="bold")
    ax.set_xlabel("Run")
    ax.set_ylabel("Weighted F1")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    savefig("stability_f1_runs.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary["aspect"], summary["std_f1"], color=["#4C9BE8", "#5CB85C", "#E87B4C"], edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title("5.2 Stability: стандартне відхилення F1", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std F1")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    savefig("stability_f1_std.png")

def build_html(robustness: pd.DataFrame, stability: pd.DataFrame):
    rob_summary = robustness.groupby("aspect")["robustness"].mean().reset_index()
    rows_rob = "".join(
        f"<tr><td>{r.aspect}</td><td>{r.robustness:.4f}</td></tr>"
        for r in rob_summary.itertuples()
    )
    rows_stab = "".join(
        f"<tr><td>{r.aspect}</td><td>{r.mean_f1:.4f}</td><td>{r.std_f1:.4f}</td><td>{r.min_f1:.4f}</td><td>{r.max_f1:.4f}</td></tr>"
        for r in stability.itertuples()
    )
    imgs = "".join(f'<div><img src="{p.name}"><p>{p.stem}</p></div>' for p in IMAGE_PATHS)
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>5.2 Reliability BERT-ABSA</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;color:#222;background:#f7f9fc}}
    h1{{color:#1a3a6b;border-bottom:3px solid #1a3a6b;padding-bottom:8px}}
    h2{{color:#2c5f9e;margin-top:28px}}
    table{{border-collapse:collapse;width:100%;background:white;margin:14px 0}}
    th,td{{border:1px solid #ccc;padding:9px 12px;text-align:center}}
    th{{background:#2c5f9e;color:white}}
    tr:nth-child(even){{background:#eef2f8}}
    .gallery{{display:flex;flex-wrap:wrap;gap:16px}}
    .gallery img{{max-width:520px;border:1px solid #ccc;border-radius:6px;background:white}}
    .gallery p{{font-size:12px;color:#666;text-align:center}}
  </style>
</head>
<body>
  <h1>5.2. Тестування надійності BERT-ABSA ML-моделі</h1>
  <p>Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}. Модель: <code>{MODEL_DIR}</code>.</p>
  <h2>Robustness</h2>
  <p>Robustness перевіряє, чи зберігається прогноз після невеликих шумових змін тексту: lowercase, зайві пробіли, пунктуація, дрібна помилка, видалення одного слова.</p>
  <table><tr><th>Аспект</th><th>Середній robustness</th></tr>{rows_rob}</table>
  <h2>Stability</h2>
  <p>Stability перевіряє, наскільки сильно змінюється F1-score на різних випадкових validation-підвибірках. Чим менше std F1, тим стабільніша модель.</p>
  <table><tr><th>Аспект</th><th>Mean F1</th><th>Std F1</th><th>Min F1</th><th>Max F1</th></tr>{rows_stab}</table>
  <h2>Візуалізації</h2>
  <div class="gallery">{imgs}</div>
  <h2>Висновок</h2>
  <p>Надійність моделі оцінювалася не лише стандартними метриками якості, а й стійкістю до шуму та стабільністю на різних підвибірках. Це дозволяє показати, чи модель працює передбачувано в умовах реальних текстових відгуків.</p>
</body>
</html>"""
    path = RESULTS_DIR / "report_5_2_reliability.html"
    path.write_text(html, encoding="utf-8")
    print(f"HTML-звіт: {path}")

def main():
    print(f"Device: {DEVICE}")
    print(f"Модель: {MODEL_DIR}")
    print(f"Дані: {DATA_PATH}")
    model = BertABSA()
    val_df = load_val_df()
    robustness = robustness_test(model, val_df)
    runs, stability = stability_test(model, val_df)
    print("\nRobustness:")
    print(robustness.to_string(index=False))
    print("\nStability summary:")
    print(stability.to_string(index=False))
    plot_robustness(robustness)
    plot_stability(runs, stability)
    build_html(robustness, stability)
    print(f"Усі результати: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
