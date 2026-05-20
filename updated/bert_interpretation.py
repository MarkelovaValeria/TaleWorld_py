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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "saved_models_4_3_bert" / "bert_absa"
RESULTS_DIR = BASE_DIR / "results_4_5_bert_interpretation"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
ASPECTS = ["content_quality", "clarity", "difficulty"]

ASPECT_DESCRIPTIONS = {
    "content_quality": "quality and depth of course content and materials",
    "clarity": "clarity of explanations and teaching style",
    "difficulty": "difficulty level, pace and workload of the course",
}

DEMO_REVIEWS = [
    {
        "text": "The course material is comprehensive, but the instructor explains it in a confusing way and the workload is too heavy.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "negative"},
        "note": "Змішаний приклад: контент позитивний, пояснення і складність негативні",
    },
    {
        "text": "The videos are short and clear, although the assignments are quite demanding.",
        "expected": {"content_quality": "positive", "clarity": "positive", "difficulty": "negative"},
        "note": "Позитивна зрозумілість, але негативна складність",
    },
    {
        "text": "The content feels outdated, but the teacher explains every concept clearly and the workload is reasonable.",
        "expected": {"content_quality": "negative", "clarity": "positive", "difficulty": "positive"},
        "note": "Негативний контент, але позитивні clarity/difficulty",
    },
    {
        "text": "Чудовий матеріал і корисні приклади, але викладач пояснює нечітко і занадто швидко.",
        "expected": {"content_quality": "positive", "clarity": "negative", "difficulty": "negative"},
        "note": "Багатомовний змішаний приклад",
    },
]

IMAGE_PATHS = []

def savefig(name: str):
    path = RESULTS_DIR / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    IMAGE_PATHS.append(path)
    print(f"Збережено: {path}")

class BertABSAInterpreter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: list[str], aspect: str) -> np.ndarray:
        probs = []
        aspect_desc = ASPECT_DESCRIPTIONS[aspect]
        for text in texts:
            enc = self.tokenizer(
                text,
                aspect_desc,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs.append(torch.softmax(logits, dim=1)[0].detach().cpu().numpy())
        return np.vstack(probs)

    def predict_one(self, text: str, aspect: str) -> dict:
        probs = self.predict_proba([text], aspect)[0]
        label_id = int(np.argmax(probs))
        return {
            "label": "positive" if label_id == 1 else "negative",
            "confidence": round(float(probs[label_id]), 4),
            "negative_prob": round(float(probs[0]), 4),
            "positive_prob": round(float(probs[1]), 4),
        }

def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def occlusion_importance(model: BertABSAInterpreter, text: str, aspect: str) -> list[tuple[str, float]]:
    tokens = tokenize_words(text)
    if not tokens:
        return []
    base_pos = float(model.predict_proba([text], aspect)[0, 1])
    rows = []
    for i, token in enumerate(tokens):
        if not token.strip() or re.fullmatch(r"[^\w]+", token):
            continue
        masked = " ".join(tokens[:i] + tokens[i + 1:])
        new_pos = float(model.predict_proba([masked], aspect)[0, 1])
        rows.append((token, base_pos - new_pos))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows[:10]

def plot_aspect_sensitivity(model: BertABSAInterpreter):
    fig, axes = plt.subplots(1, len(DEMO_REVIEWS), figsize=(5 * len(DEMO_REVIEWS), 4), sharey=True)
    if len(DEMO_REVIEWS) == 1:
        axes = [axes]
    records = []
    for ax, item in zip(axes, DEMO_REVIEWS):
        text = item["text"]
        probs = [float(model.predict_proba([text], asp)[0, 1]) for asp in ASPECTS]
        colors = ["#4C9BE8", "#5CB85C", "#E87B4C"]
        bars = ax.bar(ASPECTS, probs, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(text[:45] + "...", fontsize=9)
        ax.tick_params(axis="x", labelrotation=25)
        ax.axhline(0.5, color="#333", linestyle="--", linewidth=1, alpha=0.5)
        for asp, prob in zip(ASPECTS, probs):
            records.append({"text": text, "aspect": asp, "positive_probability": round(prob, 4)})
    plt.suptitle("BERT-ABSA: aspect sensitivity для однакових коментарів", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("aspect_sensitivity.png")
    pd.DataFrame(records).to_csv(RESULTS_DIR / "aspect_sensitivity.csv", index=False, encoding="utf-8-sig")

def plot_occlusion(model: BertABSAInterpreter):
    records = []
    for idx, item in enumerate(DEMO_REVIEWS, start=1):
        text = item["text"]
        fig, axes = plt.subplots(len(ASPECTS), 1, figsize=(10, 3.2 * len(ASPECTS)))
        if len(ASPECTS) == 1:
            axes = [axes]
        for ax, aspect in zip(axes, ASPECTS):
            weights = occlusion_importance(model, text, aspect)
            if not weights:
                ax.axis("off")
                continue
            words = [w for w, _ in weights][::-1]
            vals = [v for _, v in weights][::-1]
            colors = ["#27ae60" if v > 0 else "#e74c3c" for v in vals]
            ax.barh(words, vals, color=colors, edgecolor="white")
            ax.axvline(0, color="#333", linewidth=0.8)
            pred = model.predict_one(text, aspect)
            ax.set_title(f"{aspect}: pred={pred['label']} | P(pos)={pred['positive_prob']:.3f}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Вплив слова на P(positive): + підтримує positive, - підтримує negative")
            ax.grid(axis="x", alpha=0.25)
            for word, value in weights:
                records.append({
                    "review_id": idx,
                    "text": text,
                    "aspect": aspect,
                    "word": word,
                    "importance": round(float(value), 5),
                })
        plt.suptitle(f"Token occlusion explanation #{idx}: {item['note']}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(f"occlusion_review_{idx}.png")
    pd.DataFrame(records).to_csv(RESULTS_DIR / "token_occlusion_importance.csv", index=False, encoding="utf-8-sig")

def plot_lime(model: BertABSAInterpreter):
    if not HAS_LIME:
        print("LIME не встановлено, використано token occlusion fallback.")
        return
    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    records = []
    for idx, item in enumerate(DEMO_REVIEWS[:3], start=1):
        text = item["text"]
        fig, axes = plt.subplots(len(ASPECTS), 1, figsize=(10, 3.2 * len(ASPECTS)))
        if len(ASPECTS) == 1:
            axes = [axes]
        for ax, aspect in zip(axes, ASPECTS):
            def predictor(texts, asp=aspect):
                return model.predict_proba(list(texts), asp)
            exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=300, labels=[1])
            pairs = sorted(exp.as_list(label=1), key=lambda x: abs(x[1]), reverse=True)[:10]
            words = [w for w, _ in pairs][::-1]
            vals = [v for _, v in pairs][::-1]
            colors = ["#27ae60" if v > 0 else "#e74c3c" for v in vals]
            ax.barh(words, vals, color=colors, edgecolor="white")
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_title(f"LIME [{aspect}]", fontsize=10, fontweight="bold")
            ax.set_xlabel("LIME weight for positive class")
            ax.grid(axis="x", alpha=0.25)
            for word, value in pairs:
                records.append({
                    "review_id": idx,
                    "text": text,
                    "aspect": aspect,
                    "word": word,
                    "lime_weight_positive": round(float(value), 5),
                })
        plt.suptitle(f"LIME explanation #{idx}: {item['note']}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(f"lime_review_{idx}.png")
    pd.DataFrame(records).to_csv(RESULTS_DIR / "lime_importance.csv", index=False, encoding="utf-8-sig")

def save_predictions(model: BertABSAInterpreter):
    rows = []
    for idx, item in enumerate(DEMO_REVIEWS, start=1):
        row = {"review_id": idx, "text": item["text"], "note": item["note"]}
        for aspect in ASPECTS:
            pred = model.predict_one(item["text"], aspect)
            row[f"expected_{aspect}"] = item["expected"][aspect]
            row[f"pred_{aspect}"] = pred["label"]
            row[f"confidence_{aspect}"] = pred["confidence"]
            row[f"positive_prob_{aspect}"] = pred["positive_prob"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "interpretation_predictions.csv", index=False, encoding="utf-8-sig")
    return rows

def build_html(rows: list[dict]):
    pred_rows = ""
    for row in rows:
        pred_rows += f"<h3>#{row['review_id']}. {row['note']}</h3><p><i>{row['text']}</i></p><table><tr><th>Аспект</th><th>Очікується</th><th>Прогноз</th><th>P(pos)</th></tr>"
        for aspect in ASPECTS:
            ok = row[f"expected_{aspect}"] == row[f"pred_{aspect}"]
            bg = "#c8f7c5" if ok else "#f7c5c5"
            pred_rows += f"""
            <tr style="background:{bg}">
              <td>{aspect}</td>
              <td>{row[f'expected_{aspect}']}</td>
              <td>{'✓' if ok else '✗'} {row[f'pred_{aspect}']}</td>
              <td>{row[f'positive_prob_{aspect}']:.4f}</td>
            </tr>
            """
        pred_rows += "</table>"
    imgs = "".join(f'<div><img src="{p.name}"><p>{p.stem}</p></div>' for p in IMAGE_PATHS)
    method = "LIME + token occlusion" if HAS_LIME else "token occlusion fallback, бо LIME не встановлено"
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>4.5 Інтерпретація BERT-ABSA</title>
  <style>
    body{{font-family:Segoe UI,Arial,sans-serif;max-width:1120px;margin:30px auto;color:#222;background:#f7f9fc}}
    h1{{color:#1a3a6b;border-bottom:3px solid #1a3a6b;padding-bottom:8px}}
    h2{{color:#2c5f9e;margin-top:28px}}
    h3{{color:#34607a}}
    table{{border-collapse:collapse;width:100%;background:white;margin:10px 0 18px}}
    th,td{{border:1px solid #ccc;padding:8px 12px;text-align:left}}
    th{{background:#2c5f9e;color:white}}
    .gallery{{display:flex;flex-wrap:wrap;gap:16px}}
    .gallery img{{max-width:520px;border:1px solid #ccc;border-radius:6px;background:white}}
    .gallery p{{font-size:12px;color:#666;text-align:center}}
  </style>
</head>
<body>
  <h1>4.5. Інтерпретація результатів BERT-ABSA</h1>
  <p>Дата генерації: {datetime.now().strftime("%Y-%m-%d %H:%M")}. Модель: <code>{MODEL_DIR}</code>. Метод інтерпретації: {method}.</p>
  <h2>Aspect sensitivity</h2>
  <p>Один і той самий коментар подається з різними описами аспектів. Якщо модель справді aspect-based, імовірність positive має змінюватися залежно від аспекту.</p>
  <h2>Прогнози на змішаних коментарях</h2>
  {pred_rows}
  <h2>Графіки інтерпретації</h2>
  <div class="gallery">{imgs}</div>
  <h2>Висновок</h2>
  <p>Інтерпретація показує, які слова найбільше змінюють імовірність positive для кожного аспекту. Це дозволяє пояснити, чому один коментар може бути позитивним для content_quality, але негативним для clarity або difficulty.</p>
</body>
</html>"""
    path = RESULTS_DIR / "report_4_5_interpretation.html"
    path.write_text(html, encoding="utf-8")
    print(f"HTML-звіт: {path}")

def main():
    print(f"Device: {DEVICE}")
    print(f"LIME: {'доступний' if HAS_LIME else 'не встановлено'}")
    print(f"Модель: {MODEL_DIR}")
    model = BertABSAInterpreter()
    rows = save_predictions(model)
    plot_aspect_sensitivity(model)
    plot_occlusion(model)
    plot_lime(model)
    build_html(rows)
    print(f"Усі результати: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
