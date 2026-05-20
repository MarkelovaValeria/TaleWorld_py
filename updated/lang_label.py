import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import os

DetectorFactory.seed = 42
INPUT_PATH  = "processed_reviews.csv"
OUTPUT_PATH = "processed_reviews.csv"   
TEXT_COL    = "clean_text"             

def detect_language(text: str) -> str:
    try:
        text = str(text).strip()
        if len(text.split()) < 2:
            return "en"   

        lang = detect(text)

        if lang.startswith("zh"):
            return "zh"

        return lang

    except Exception:
        return "en"

def main():
    print("=" * 55)
    print("  ВИЗНАЧЕННЯ МОВИ ВІДГУКІВ")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Файл не знайдено: {INPUT_PATH}\n"
            f"Спочатку запусти clean_text.py"
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"Завантажено: {len(df)} рядків")
    print(f"Колонки: {df.columns.tolist()}")

    if "language" in df.columns:
        print("\nКолонка 'language' вже існує.")
        print(df["language"].value_counts().to_string())
        print("\nЯкщо хочеш перевизначити — видали колонку вручну і запусти знову.")
        return

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    tqdm.pandas(desc="Визначення мови")
    df["language"] = df[TEXT_COL].progress_apply(detect_language)

    print("\nРозподіл мов:")
    lang_counts = df["language"].value_counts()
    total = len(df)
    for lang, count in lang_counts.items():
        print(f"  {lang:<6} {count:>6} ({count/total:.1%})")

    slavic = df["language"].isin(["uk", "ru"]).sum()
    other  = total - slavic
    print(f"\n  uk/ru (→ ukr-RoBERTa) : {slavic:>6} ({slavic/total:.1%})")
    print(f"  решта (→ XLM-RoBERTa) : {other:>6} ({other/total:.1%})")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nЗбережено: {OUTPUT_PATH}")
    print(f"Нова колонка: 'language'")

if __name__ == "__main__":
    main()