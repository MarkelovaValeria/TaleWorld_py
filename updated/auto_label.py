import re
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "processed_reviews_labeled.csv"
BACKUP     = BASE_DIR / "processed_reviews_labeled_backup.csv"
STATS_PATH = BASE_DIR / "label_aspects_stats.txt"

ASPECT_KEYWORDS: dict[str, set[str]] = {

    "content_quality": {
        "content", "material", "topic", "subject", "syllabus",
        "curriculum", "relevant", "outdated", "updated", "comprehensive",
        "thorough", "shallow", "depth", "theory", "practice",
        "concept", "knowledge", "information", "coverage",
        "resource", "handout", "slide", "notebook", "transcript",
        "video", "pdf", "reading", "book", "article", "paper",
        "reference", "code", "dataset", "repository",
        "exercise", "structure", "organize", "module", "chapter",
        "section", "unit", "week", "lesson", "lecture", "sequence",
        "flow", "logical", "coherent", "scattered", "disorganized",
        "контент", "матеріал", "тема", "програма", "актуальний",
        "глибина", "теорія", "практика", "ресурс", "відео",
        "структура", "модуль", "розділ", "урок", "логічний",
        "материал", "теория", "практика",
    },

    "clarity": {
        "clear", "clarity", "explain", "explanation", "understandable",
        "confusing", "unclear", "vague", "ambiguous", "straightforward",
        "concise", "verbose", "simple", "intuitive",
        "instructor", "teacher", "professor", "mentor", "tutor",
        "lecturer", "speaker", "presenter", "engaging", "boring",
        "monotone", "enthusiastic", "passionate", "charismatic",
        "accent", "pronunciation", "articulate", "mumble",
        "language", "vocabulary", "terminology",
        "example", "illustration", "demo", "demonstration",
        "analogy", "visual", "diagram", "chart",
        "зрозумілий", "пояснення", "чіткий", "викладач", "вчитель",
        "нудний", "захоплений", "приклад", "ілюстрація",
        "понятный", "объяснение", "преподаватель", "скучный",
    },

    "difficulty": {
        "difficulty", "difficult", "easy", "hard", "challenging",
        "beginner", "intermediate", "advanced", "expert", "level",
        "appropriate", "overwhelming", "steep", "prerequisite",
        "pace", "speed", "fast", "slow", "rush", "quick", "gradual",
        "manageable", "reasonable", "pressure", "time", "duration",
        "workload", "homework", "assignment", "task", "project",
        "quiz", "exam", "test", "deadline", "effort", "repetitive",
        "складний", "легкий", "важкий", "рівень", "перевантажений",
        "темп", "швидко", "повільно", "навантаження", "завдання",
        "сложный", "лёгкий", "уровень", "перегруженный", "нагрузка",
    },
}

POSITIVE_WORDS: set[str] = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "outstanding", "superb", "brilliant", "perfect", "best", "love",
    "helpful", "useful", "valuable", "informative", "recommend",
    "interesting", "engaging", "enjoyable", "professional",
    "well", "nice", "fine", "solid", "decent", "strong",
    "clear", "organized", "structured", "comprehensive", "thorough",
    "detailed", "concise", "intuitive", "straightforward", "relevant",
    "updated", "modern", "practical", "applicable", "hands-on",
    "enthusiastic", "passionate", "knowledgeable", "experienced",
    "manageable", "reasonable", "appropriate", "balanced",
    "чудовий", "відмінний", "чіткий", "корисний", "цікавий",
    "хороший", "прекрасний", "зрозумілий", "гарний",
    "отличный", "хороший", "полезный", "понятный",
}

NEGATIVE_WORDS: set[str] = {
    "bad", "poor", "terrible", "awful", "horrible", "worst",
    "boring", "dull", "useless", "waste", "disappointing",
    "frustrating", "annoying", "confusing", "unclear", "vague",
    "shallow", "incomplete", "outdated", "irrelevant",
    "disorganized", "scattered", "inconsistent", "wrong",
    "monotone", "dry", "slow", "fast", "overwhelming", "steep",
    "difficult", "hard", "challenging", "repetitive", "tedious",
    "rushed", "dense", "cluttered", "messy", "lacking", "missing",
    "error", "mistake", "glitch", "typo", "broken",
    "expensive", "overpriced", "unprofessional",
    "поганий", "жахливий", "нудний", "незрозумілий", "важкий",
    "повільний", "застарілий", "марний",
    "плохой", "скучный", "непонятный", "устаревший", "бесполезный",
}

NEGATION_WORDS: set[str] = {
    "not", "no", "never", "barely", "hardly", "without",
    "lack", "lacks", "lacking", "nothing", "none",
    "не", "ні", "нема", "немає",
}

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯіІїЇєЄёЁ']+", text.lower())

def _is_mentioned(tokens: list[str], aspect: str) -> bool:
    kws = ASPECT_KEYWORDS[aspect]
    return any(t in kws for t in tokens)

def _aspect_sentiment(tokens: list[str], aspect: str) -> str:
    kws    = ASPECT_KEYWORDS[aspect]
    window = 5

    pos_score = 0.0
    neg_score = 0.0

    for i, tok in enumerate(tokens):
        if tok not in kws:
            continue

        start = max(0, i - window)
        end   = min(len(tokens), i + window + 1)
        ctx   = tokens[start:end]

        for j, w in enumerate(ctx):
            if w in POSITIVE_WORDS:
                neg_before = any(
                    ctx[k] in NEGATION_WORDS
                    for k in range(max(0, j - 2), j)
                )
                if neg_before:
                    neg_score += 1.0
                else:
                    pos_score += 1.0
            elif w in NEGATIVE_WORDS:
                neg_before = any(
                    ctx[k] in NEGATION_WORDS
                    for k in range(max(0, j - 2), j)
                )
                if neg_before:
                    pos_score += 0.5
                else:
                    neg_score += 1.0

    if pos_score == 0 and neg_score == 0:
        return "unknown"
    return "positive" if pos_score >= neg_score else "negative"

def label_row(row: pd.Series) -> dict:
    overall  = str(row["Label"]).strip().lower()
    text     = str(row.get("clean_text", row.get("Review", "")))
    tokens   = _tokenize(text)

    result = {}
    for asp in ("content_quality", "clarity", "difficulty"):
        mentioned = _is_mentioned(tokens, asp)
        result[f"{asp}_mentioned"] = mentioned

        if mentioned:
            sent = _aspect_sentiment(tokens, asp)
            result[f"{asp}_label"] = sent if sent != "unknown" else overall
        else:
            result[f"{asp}_label"] = "neutral"

    return result

def main():
    print("=" * 60)
    print("Rule-Based Автоматичне розмічення аспектних міток")
    print("Sliding window ±5 + negation handling")
    print("=" * 60)

    print(f"\nЧитаємо: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Розмір: {df.shape} | Label: {df['Label'].value_counts().to_dict()}")

    if not BACKUP.exists():
        shutil.copy(DATA_PATH, BACKUP)
        print(f"Backup збережено: {BACKUP}")

    print("\nРозмічення аспектів...")
    label_rows = df.apply(label_row, axis=1, result_type="expand")

    for asp in ("content_quality", "clarity", "difficulty"):
        df[f"label_{asp}"]     = label_rows[f"{asp}_label"]
        df[f"{asp}_mentioned"] = label_rows[f"{asp}_mentioned"]

    stats_lines = []
    stats_lines.append("=" * 60)
    stats_lines.append("СТАТИСТИКА РОЗМІЧЕННЯ (rule-based)")
    stats_lines.append("=" * 60)
    stats_lines.append(f"Всього відгуків: {len(df)}")

    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)

    for asp in ("content_quality", "clarity", "difficulty"):
        col           = f"label_{asp}"
        mentioned_n   = df[f"{asp}_mentioned"].sum()
        mentioned_pct = mentioned_n / len(df) * 100
        pos_n = (df[col] == "positive").sum()
        neg_n = (df[col] == "negative").sum()
        neu_n = (df[col] == "neutral").sum()

        line = (
            f"\n[{asp}]\n"
            f"  Згаданий : {mentioned_n} ({mentioned_pct:.1f}%)\n"
            f"  positive : {pos_n} ({pos_n/len(df):.1%})\n"
            f"  negative : {neg_n} ({neg_n/len(df):.1%})\n"
            f"  neutral  : {neu_n} ({neu_n/len(df):.1%})"
        )
        stats_lines.append(line)
        print(line)

    diff_mask = (
        (df["label_content_quality"] != df["label_clarity"]) |
        (df["label_clarity"]         != df["label_difficulty"])
    )
    diff_n = diff_mask.sum()
    line = f"\nВідгуки з різними мітками по аспектах: {diff_n} ({diff_n/len(df)*100:.1f}%)"
    stats_lines.append(line)
    print(line)

    rows_expanded = []
    for _, row in df.iterrows():
        for asp in ("content_quality", "clarity", "difficulty"):
            lbl = str(row[f"label_{asp}"]).lower()
            if lbl == "neutral":
                continue
            rows_expanded.append({
                "text":   str(row["clean_text"])[:60],
                "aspect": asp,
                "label":  1 if lbl == "positive" else 0,
            })
    expanded = pd.DataFrame(rows_expanded)

    if len(expanded) > 0:
        text_label_var = expanded.groupby("text")["label"].nunique()
        conflicting    = text_label_var[text_label_var > 1]
        line2 = (
            f"Тренувальних прикладів: {len(expanded)}\n"
            f"Текстів де аспекти мають РІЗНІ мітки: "
            f"{len(conflicting)} ({len(conflicting)/len(df):.1%})"
        )
        stats_lines.append(line2)
        print(f"\n{line2}")

    save_cols = [c for c in df.columns if not c.endswith("_mentioned")]
    df[save_cols].to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\nОновлений датасет збережено: {DATA_PATH}")

    STATS_PATH.write_text("\n".join(stats_lines), encoding="utf-8")
    print(f"Статистика збережена: {STATS_PATH}")

    print("\n" + "=" * 60)
    print("ПРИКЛАДИ З РІЗНИМИ МІТКАМИ ПО АСПЕКТАХ")
    print("=" * 60)

    diff_examples = df[diff_mask][
        ["clean_text", "Label",
         "label_content_quality", "label_clarity", "label_difficulty"]
    ].head(8)

    for _, row in diff_examples.iterrows():
        print(f'\n"{str(row["clean_text"])[:90]}"')
        print(f"  Overall : {row['Label']}")
        for asp in ("content_quality", "clarity", "difficulty"):
            print(f"  {asp:<22}: {row[f'label_{asp}']}")

if __name__ == "__main__":
    main()
