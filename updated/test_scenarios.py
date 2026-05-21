"""
test_scenarios.py
=================
3 сценарії для перевірки роботи BERT-ABSA моделі:
  - Сценарій A: виключно позитивні відгуки (очікується ~4.5–5/5)
  - Сценарій B: виключно негативні відгуки (очікується ~1–1.5/5)
  - Сценарій C: змішані відгуки (очікується ~2.5–3.5/5)

Результати зберігаються у:
  results_scenarios/scenario_A_positive.json
  results_scenarios/scenario_B_negative.json
  results_scenarios/scenario_C_mixed.json
  results_scenarios/summary_all_scenarios.json
  results_scenarios/report_scenarios.html
"""

import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── Конфігурація ─────────────────────────────────────────────────────────────
MODEL_PATH   = Path("saved_models_4_3_bert/bert_absa")
RESULTS_DIR  = Path("results_scenarios")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

ASPECTS = ["content_quality", "clarity", "difficulty"]

ASPECT_DESCRIPTIONS = {
    "content_quality": "quality and depth of course content and materials",
    "clarity":         "clarity of explanations and teaching style",
    "difficulty":      "difficulty level, pace and workload of the course",
}

ASPECT_NAMES_UA = {
    "content_quality": "Якість контенту",
    "clarity":         "Зрозумілість пояснень",
    "difficulty":      "Складність та темп",
}

CONFIDENCE_THRESHOLD = 0.70   # нижче → neutral


# ─── Сценарій A: ПОЗИТИВНІ відгуки ────────────────────────────────────────────
SCENARIO_A = {
    "id":          "A",
    "name":        "Позитивний курс",
    "description": "Лише захоплені відгуки — очікуємо оцінки ~4.5–5/5",
    "reviews": [
        {
            "id": 1,
            "text": (
                "Один з найкращих курсів, які я проходив! Матеріал охоплює всі "
                "важливі теми, є реальні проєкти та приклади з практики. "
                "Дуже рекомендую всім, хто хоче вивчити Python з нуля."
            ),
        },
        {
            "id": 2,
            "text": (
                "Викладач пояснює надзвичайно чітко і зрозуміло. Кожна тема "
                "розкривається крок за кроком, нічого зайвого. Після курсу "
                "я вже написав свій перший проєкт — це говорить саме за себе."
            ),
        },
        {
            "id": 3,
            "text": (
                "Чудовий баланс між теорією і практикою. Темп ідеальний — "
                "не поспішаєш, але й не нудьгуєш. Завдання цікаві та "
                "поступово ускладнюються, що дуже мотивує."
            ),
        },
        {
            "id": 4,
            "text": (
                "Absolutely fantastic course! The content is deep and well-structured, "
                "explanations are crystal clear with many real-world examples. "
                "The pace is perfect for both beginners and intermediate learners."
            ),
        },
        {
            "id": 5,
            "text": (
                "Матеріал курсу актуальний і глибокий. Викладач відповідає на "
                "всі питання і допомагає розібратися навіть у складних темах. "
                "Найкраще навчання, яке я отримував онлайн."
            ),
        },
        {
            "id": 6,
            "text": (
                "Структура курсу бездоганна — кожен модуль логічно продовжує "
                "попередній. Пояснення зрозумілі навіть без попереднього досвіду. "
                "Складність зростає поступово, що дозволяє впевнено рухатись вперед."
            ),
        },
        {
            "id": 7,
            "text": (
                "This course exceeded all my expectations. The instructor is "
                "incredibly knowledgeable and explains every concept with patience. "
                "Highly recommend to anyone serious about learning programming."
            ),
        },
        {
            "id": 8,
            "text": (
                "Неймовірно корисний курс! Всі теми розкриті детально, "
                "практичних завдань достатньо, і вони дійсно закріплюють знання. "
                "Темп і складність підібрані ідеально."
            ),
        },
        {
            "id": 9,
            "text": (
                "Я вже пройшов багато курсів, але цей — окремий рівень якості. "
                "Матеріал глибокий, викладач захоплює з першої лекції. "
                "Дякую за такий чудовий контент!"
            ),
        },
        {
            "id": 10,
            "text": (
                "Excellent material, very clear teaching, and well-balanced difficulty. "
                "Every lecture is packed with useful information without being overwhelming. "
                "This is exactly what quality online education should look like."
            ),
        },
    ],
}

# ─── Сценарій B: НЕГАТИВНІ відгуки ────────────────────────────────────────────
SCENARIO_B = {
    "id":          "B",
    "name":        "Негативний курс",
    "description": "Лише критичні відгуки — очікуємо оцінки ~1–1.5/5",
    "reviews": [
        {
            "id": 1,
            "text": (
                "Жахливий курс. Матеріал поверхневий, скопійований з інтернету "
                "без жодного пояснення. Витратив гроші і час дарма. "
                "Категорично не рекомендую."
            ),
        },
        {
            "id": 2,
            "text": (
                "Викладач читає по слайдах монотонним голосом і не може відповісти "
                "на жодне запитання студентів. Пояснення незрозумілі та заплутані. "
                "Враження — повна катастрофа."
            ),
        },
        {
            "id": 3,
            "text": (
                "Темп курсу неможливо витримати. За один тиждень стільки завдань, "
                "що нереально встигнути. Матеріал перескакує без жодної логіки, "
                "новачки тут просто загубляться."
            ),
        },
        {
            "id": 4,
            "text": (
                "Terrible course. The content is outdated by at least 5 years, "
                "the explanations are confusing and the instructor clearly doesn't "
                "understand the questions students ask. Total waste of money."
            ),
        },
        {
            "id": 5,
            "text": (
                "Контент курсу застарілий і не відповідає опису. Замість реальних "
                "проєктів — нудні теоретичні лекції без практики. "
                "Дуже розчарований покупкою."
            ),
        },
        {
            "id": 6,
            "text": (
                "This is by far the worst course I have ever taken. Poorly structured, "
                "confusing explanations, and an overwhelming workload with no support. "
                "Avoid at all costs."
            ),
        },
        {
            "id": 7,
            "text": (
                "Повне розчарування. Матеріал незрозумілий, пояснення відсутні, "
                "а складність стрибає від нуля до неможливого без жодних сходинок. "
                "Не варто витрачати на це час."
            ),
        },
        {
            "id": 8,
            "text": (
                "Курс абсолютно не відповідає рівню початківців. Темп шалений, "
                "матеріал подається хаотично, а завдання незрозумілі. "
                "Після другого тижня я просто кинув."
            ),
        },
        {
            "id": 9,
            "text": (
                "The course content is shallow and misleading. The instructor "
                "speaks too fast, skips important explanations, and the difficulty "
                "spikes are brutal without any warning or preparation."
            ),
        },
        {
            "id": 10,
            "text": (
                "Найгірший досвід онлайн-навчання. Матеріал нікуди не годиться, "
                "викладач не зацікавлений у навчанні студентів, "
                "а навантаження просто нелюдське. Повернув би гроші якби міг."
            ),
        },
    ],
}

# ─── Сценарій C: ЗМІШАНІ відгуки ──────────────────────────────────────────────
SCENARIO_C = {
    "id":          "C",
    "name":        "Змішаний курс",
    "description": "Реалістичні змішані відгуки — очікуємо оцінки ~2.5–3.5/5",
    "reviews": [
        {
            "id": 1,
            "text": (
                "Матеріал курсу хороший і актуальний, але викладач пояснює "
                "занадто швидко. Деякі теми потребують більше прикладів."
            ),
        },
        {
            "id": 2,
            "text": (
                "Пояснення чіткі і зрозумілі, але контент базовий — "
                "для досвідчених розробників тут мало нового. "
                "Для початківців підійде непогано."
            ),
        },
        {
            "id": 3,
            "text": (
                "Good content overall, but the difficulty level is inconsistent. "
                "Some weeks are very easy and others are overwhelming. "
                "Needs better pacing."
            ),
        },
        {
            "id": 4,
            "text": (
                "Непоганий курс. Є і плюси, і мінуси. Контент цікавий, "
                "але пояснення місцями незрозумілі. В цілому корисно, "
                "але є над чим попрацювати."
            ),
        },
        {
            "id": 5,
            "text": (
                "Курс корисний для початку, але матеріал поверхневий. "
                "Викладач старається, пояснення зрозумілі, "
                "проте глибини знань після курсу не вистачає."
            ),
        },
        {
            "id": 6,
            "text": (
                "The explanations are clear but the course content feels rushed. "
                "Topics are covered too briefly without enough depth or practice. "
                "Average experience overall."
            ),
        },
        {
            "id": 7,
            "text": (
                "Складність курсу підібрана нормально, але є питання до якості "
                "матеріалів. Деякі теми розкриті добре, інші — ледь торкнуті. "
                "Нерівномірна якість по модулях."
            ),
        },
        {
            "id": 8,
            "text": (
                "Interesting topics but mediocre execution. The instructor "
                "sometimes explains clearly and sometimes rushes through complex topics. "
                "The workload is manageable but could be better organized."
            ),
        },
        {
            "id": 9,
            "text": (
                "Матеріал цікавий, але занадто багато теорії і мало практики. "
                "Хотілося б більше реальних завдань. Темп нормальний, "
                "але структура модулів потребує покращення."
            ),
        },
        {
            "id": 10,
            "text": (
                "Середній курс. Викладач намагається, деякі пояснення вдалі, "
                "але загалом враження неоднозначне. Рекомендую лише якщо "
                "немає альтернатив у цій темі."
            ),
        },
    ],
}

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C]


# ─── Аналізатор ───────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    def __init__(self):
        print(f"⏳ Завантаження моделі з {MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✅ Модель готова. Пристрій: {DEVICE}\n")

    def analyze_review(self, text: str) -> dict:
        results = {}
        for asp in ASPECTS:
            enc = self.tokenizer(
                text,
                ASPECT_DESCRIPTIONS[asp],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits

            probs    = torch.softmax(logits, dim=1)[0].cpu().tolist()
            label_id = int(logits.argmax(dim=1).item())

            results[asp] = {
                "label":      "positive" if label_id == 1 else "negative",
                "confidence": round(probs[label_id], 3),
                "pos_prob":   round(probs[1], 3),
                "neg_prob":   round(probs[0], 3),
                "score":      round(1.0 + probs[1] * 4.0, 2),
            }
            # Поріг впевненості — нижче 0.70 → neutral
            if results[asp]["confidence"] < CONFIDENCE_THRESHOLD:
                results[asp]["label"] = "neutral"
                results[asp]["score"] = 3.0

        return results


# ─── Агрегація ────────────────────────────────────────────────────────────────
def compute_summary(analyzed: list) -> dict:
    n = len(analyzed)
    summary = {}
    for asp in ASPECTS:
        scores   = [r["analysis"][asp]["score"]    for r in analyzed]
        pos_probs = [r["analysis"][asp]["pos_prob"] for r in analyzed]
        pos_cnt  = sum(1 for r in analyzed if r["analysis"][asp]["label"] == "positive")
        neu_cnt  = sum(1 for r in analyzed if r["analysis"][asp]["label"] == "neutral")
        neg_cnt  = sum(1 for r in analyzed if r["analysis"][asp]["label"] == "negative")
        summary[asp] = {
            "avg_score":     round(sum(scores) / n, 2),
            "avg_pos_prob":  round(sum(pos_probs) / n, 3),
            "positive_count": pos_cnt,
            "neutral_count":  neu_cnt,
            "negative_count": neg_cnt,
            "positive_pct":  round(pos_cnt / n, 3),
            "neutral_pct":   round(neu_cnt / n, 3),
            "negative_pct":  round(neg_cnt / n, 3),
            "total":         n,
        }
    summary["overall_score"] = round(
        sum(summary[a]["avg_score"] for a in ASPECTS) / len(ASPECTS), 2
    )
    return summary


def _icon(label: str) -> str:
    return {"positive": "✅", "neutral": "➖", "negative": "❌"}.get(label, "❓")

def _stars(score: float) -> str:
    s = int(round(score))
    return "★" * s + "☆" * (5 - s)

def _verdict(overall: float) -> str:
    if overall >= 4.2:  return "🟢 ВІДМІННО"
    if overall >= 3.5:  return "🟡 ДОБРЕ"
    if overall >= 2.5:  return "🟠 ЗАДОВІЛЬНО"
    return "🔴 ПОГАНО"


# ─── Консольний звіт ──────────────────────────────────────────────────────────
def print_scenario_report(scenario: dict, analyzed: list, summary: dict):
    sid   = scenario["id"]
    name  = scenario["name"]
    desc  = scenario["description"]
    n     = len(analyzed)
    now   = datetime.now().strftime("%H:%M:%S")

    print(f"\n{'═'*72}")
    print(f"  Сценарій {sid}: {name}")
    print(f"  {desc}")
    print(f"  Відгуків: {n}  |  Час: {now}")
    print(f"{'═'*72}")

    # Таблиця по відгуках
    hdr = f"  {'№':>3}  {'Відгук (скорочено)':<44}  {'Контент':^12}  {'Ясність':^12}  {'Склад-ть':^12}"
    print(f"\n{hdr}")
    print(f"  {'-'*84}")
    for item in analyzed:
        short = item["text"][:42] + "…" if len(item["text"]) > 42 else item["text"]
        row = f"  {item['id']:>3}  {short:<44}"
        for asp in ASPECTS:
            a    = item["analysis"][asp]
            icon = _icon(a["label"])
            row += f"  {icon} {a['score']:>4.2f}/5  "
        print(row)

    # Зведена таблиця
    print(f"\n  {'─'*72}")
    print(f"  {'Аспект':<28}  {'Оцінка':^7}  {'Зірки':^7}  "
          f"{'✅ Поз':^7}  {'➖ Нейт':^7}  {'❌ Нег':^7}")
    print(f"  {'─'*72}")
    for asp in ASPECTS:
        s = summary[asp]
        print(
            f"  {ASPECT_NAMES_UA[asp]:<28}  "
            f"{s['avg_score']:^7.2f}  "
            f"{_stars(s['avg_score']):^7}  "
            f"{s['positive_count']:^7}  "
            f"{s['neutral_count']:^7}  "
            f"{s['negative_count']:^7}"
        )
    print(f"  {'─'*72}")
    verdict = _verdict(summary["overall_score"])
    print(f"  {'ЗАГАЛЬНА ОЦІНКА':<28}  "
          f"{summary['overall_score']:^7.2f}  "
          f"{_stars(summary['overall_score']):^7}  "
          f"  {verdict}")
    print()


# ─── Збереження JSON ──────────────────────────────────────────────────────────
def save_json(data: dict, path: Path):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  💾 {path}")


# ─── HTML-звіт по всіх сценаріях ─────────────────────────────────────────────
def build_html_report(all_results: list[dict]) -> str:

    SCENARIO_COLORS = {"A": "#27ae60", "B": "#e74c3c", "C": "#f39c12"}
    SCENARIO_ICONS  = {"A": "🟢", "B": "🔴", "C": "🟡"}

    def _badge(label):
        cfg = {
            "positive": ("#27ae60", "Позитивно"),
            "neutral":  ("#7f8c8d", "Нейтрально"),
            "negative": ("#e74c3c", "Негативно"),
        }
        color, text = cfg.get(label, ("#999", label))
        return (f'<span style="background:{color};color:white;padding:2px 7px;'
                f'border-radius:10px;font-size:.82em;font-weight:bold">{text}</span>')

    def _score_color(s):
        if s >= 4.0: return "#27ae60"
        if s >= 3.0: return "#f39c12"
        return "#e74c3c"

    # Порівняльна таблиця сценаріїв
    compare_rows = ""
    for res in all_results:
        sid  = res["scenario_id"]
        name = res["scenario_name"]
        summ = res["summary"]
        col  = SCENARIO_COLORS[sid]
        ico  = SCENARIO_ICONS[sid]
        compare_rows += (
            f"<tr>"
            f"<td><b style='color:{col}'>{ico} Сценарій {sid}</b><br>"
            f"<small>{name}</small></td>"
        )
        for asp in ASPECTS:
            s = summ[asp]
            c = _score_color(s["avg_score"])
            compare_rows += (
                f"<td style='color:{c};font-weight:bold'>{s['avg_score']}/5</td>"
                f"<td style='font-size:.85em'>"
                f"✅{s['positive_count']} ➖{s['neutral_count']} ❌{s['negative_count']}"
                f"</td>"
            )
        ov = summ["overall_score"]
        compare_rows += (
            f"<td style='font-size:1.4em;font-weight:bold;color:{_score_color(ov)}'>"
            f"{ov}/5</td>"
            f"<td>{_verdict(ov)}</td>"
            f"</tr>\n"
        )

    # Деталі по кожному сценарію
    scenario_sections = ""
    for res in all_results:
        sid      = res["scenario_id"]
        name     = res["scenario_name"]
        desc     = res["scenario_description"]
        analyzed = res["reviews"]
        summ     = res["summary"]
        col      = SCENARIO_COLORS[sid]
        ico      = SCENARIO_ICONS[sid]

        review_rows = ""
        for item in analyzed:
            short = item["text"][:90] + ("…" if len(item["text"]) > 90 else "")
            review_rows += f"<tr><td>{item['id']}</td><td class='rv'>{short}</td>"
            for asp in ASPECTS:
                a = item["analysis"][asp]
                c = _score_color(a["score"])
                review_rows += (
                    f"<td>{_badge(a['label'])}<br>"
                    f"<small style='color:{c};font-weight:bold'>{a['score']}/5</small>"
                    f"<br><small style='color:#aaa'>conf={a['confidence']}</small></td>"
                )
            review_rows += "</tr>\n"

        summ_rows = ""
        for asp in ASPECTS:
            s = summ[asp]
            c = _score_color(s["avg_score"])
            summ_rows += (
                f"<tr><td><b>{ASPECT_NAMES_UA[asp]}</b></td>"
                f"<td style='color:{c};font-weight:bold'>{s['avg_score']}/5</td>"
                f"<td>✅ {s['positive_count']} &nbsp; ➖ {s['neutral_count']} "
                f"&nbsp; ❌ {s['negative_count']}</td>"
                f"<td>{s['positive_pct']:.0%} позит.</td></tr>\n"
            )

        scenario_sections += f"""
        <div class="scenario-block" style="border-left:6px solid {col}">
          <h2 style="color:{col}">{ico} Сценарій {sid}: {name}</h2>
          <p style="color:#666">{desc}</p>
          <div style="display:inline-block;background:{col};color:white;
               padding:10px 22px;border-radius:10px;font-size:1.5em;
               font-weight:bold;margin-bottom:15px">
            Загальна оцінка: {summ['overall_score']}/5 &nbsp; {_verdict(summ['overall_score'])}
          </div>
          <h3>Зведення по аспектах</h3>
          <table>
            <tr><th>Аспект</th><th>Середня оцінка</th><th>Розподіл міток</th><th>% позитивних</th></tr>
            {summ_rows}
          </table>
          <h3>Відгуки детально</h3>
          <table>
            <tr><th>#</th><th>Відгук</th>
                <th>Якість контенту</th><th>Зрозумілість</th><th>Складність</th></tr>
            {review_rows}
          </table>
        </div>
        """

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>BERT-ABSA — Перевірка 3 сценаріїв</title>
  <style>
    body  {{font-family:'Segoe UI',sans-serif;max-width:1300px;margin:40px auto;
            background:#f4f6f9;color:#2c3e50;padding:20px}}
    h1    {{color:#2c3e50;border-bottom:4px solid #3498db;padding-bottom:10px}}
    h2    {{margin-top:10px}}
    h3    {{color:#555;margin-top:18px}}
    table {{border-collapse:collapse;width:100%;margin:12px 0;background:white;
            border-radius:10px;overflow:hidden;
            box-shadow:0 2px 8px rgba(0,0,0,.08)}}
    th    {{background:#2c3e50;color:white;padding:11px 9px;text-align:center}}
    td    {{border:1px solid #e8e8e8;padding:9px;text-align:center;vertical-align:middle}}
    .rv   {{text-align:left;font-size:.88em;max-width:300px}}
    tr:hover{{background:#f5f8ff}}
    .scenario-block{{background:white;border-radius:12px;padding:24px 28px;
                     margin:28px 0;box-shadow:0 3px 12px rgba(0,0,0,.09)}}
    .legend{{display:flex;gap:20px;margin:12px 0;font-size:.95em}}
    .leg-item{{padding:6px 14px;border-radius:8px;color:white;font-weight:bold}}
  </style>
</head>
<body>
  <h1>🧪 BERT-ABSA — Перевірка моделі: 3 сценарії</h1>
  <p><b>Дата:</b> {now} &nbsp;|&nbsp;
     <b>Поріг нейтральності:</b> confidence &lt; {CONFIDENCE_THRESHOLD}</p>

  <div class="legend">
    <span class="leg-item" style="background:#27ae60">🟢 Сценарій A — Позитивний</span>
    <span class="leg-item" style="background:#e74c3c">🔴 Сценарій B — Негативний</span>
    <span class="leg-item" style="background:#f39c12">🟡 Сценарій C — Змішаний</span>
  </div>

  <h2>📊 Порівняльна таблиця сценаріїв</h2>
  <table>
    <tr>
      <th>Сценарій</th>
      <th>Контент<br>оцінка</th><th>Контент<br>мітки</th>
      <th>Ясність<br>оцінка</th><th>Ясність<br>мітки</th>
      <th>Складність<br>оцінка</th><th>Складність<br>мітки</th>
      <th>Загальна<br>оцінка</th><th>Вердикт</th>
    </tr>
    {compare_rows}
  </table>

  {scenario_sections}

  <p style="margin-top:40px;color:#aaa;font-size:.82em">
    Модель: saved_models_4_3_bert/bert_absa (XLM-RoBERTa Sentence-Pair ABSA)
  </p>
</body>
</html>"""

    path = RESULTS_DIR / "report_scenarios.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


# ─── Головна функція ──────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  BERT-ABSA — Перевірка моделі: 3 сценарії")
    print("  A=позитивний | B=негативний | C=змішаний")
    print("=" * 72)

    analyzer    = ReviewAnalyzer()
    all_results = []

    for scenario in ALL_SCENARIOS:
        sid  = scenario["id"]
        name = scenario["name"]
        reviews = scenario["reviews"]

        print(f"\n{'─'*72}")
        print(f"▶  Сценарій {sid}: {name}  ({len(reviews)} відгуків)")
        print(f"{'─'*72}")

        analyzed = []
        for rev in reviews:
            result = analyzer.analyze_review(rev["text"])
            analyzed.append({**rev, "analysis": result,
                              "analyzed_at": datetime.now().isoformat()})

        summary = compute_summary(analyzed)
        print_scenario_report(scenario, analyzed, summary)

        # Зберегти JSON
        names_map = {"A": "positive", "B": "negative", "C": "mixed"}
        json_data = {
            "scenario_id":          sid,
            "scenario_name":        name,
            "scenario_description": scenario["description"],
            "generated_at":         datetime.now().isoformat(),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "reviews":              [
                {"id": r["id"], "text": r["text"], "analysis": r["analysis"]}
                for r in analyzed
            ],
            "summary": summary,
        }
        json_path = RESULTS_DIR / f"scenario_{sid}_{names_map[sid]}.json"
        save_json(json_data, json_path)

        all_results.append(json_data)

    # Зведений JSON по всіх сценаріях
    summary_all = {
        "generated_at":         datetime.now().isoformat(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "scenarios": [
            {
                "id":            r["scenario_id"],
                "name":          r["scenario_name"],
                "overall_score": r["summary"]["overall_score"],
                "verdict":       _verdict(r["summary"]["overall_score"]),
                "aspects": {
                    asp: {
                        "avg_score":    r["summary"][asp]["avg_score"],
                        "positive_pct": r["summary"][asp]["positive_pct"],
                    }
                    for asp in ASPECTS
                },
            }
            for r in all_results
        ],
    }
    summary_path = RESULTS_DIR / "summary_all_scenarios.json"
    save_json(summary_all, summary_path)

    # HTML-звіт
    html_path = build_html_report(all_results)

    # Фінальний консольний підсумок
    print(f"\n{'═'*72}")
    print("  📊  ПІДСУМОК ТРЬОХ СЦЕНАРІЇВ")
    print(f"{'═'*72}")
    print(f"  {'Сценарій':<35}  {'Оцінка':^8}  {'Зірки':^7}  Вердикт")
    print(f"  {'─'*60}")
    for r in all_results:
        ov   = r["summary"]["overall_score"]
        ico  = {"A": "🟢", "B": "🔴", "C": "🟡"}[r["scenario_id"]]
        line = f"  {ico} {r['scenario_id']}: {r['scenario_name']:<30}  {ov:^8.2f}  {_stars(ov):^7}  {_verdict(ov)}"
        print(line)
    print(f"\n  Файли збережено у папці: {RESULTS_DIR}/")
    print(f"  💾 scenario_A_positive.json")
    print(f"  💾 scenario_B_negative.json")
    print(f"  💾 scenario_C_mixed.json")
    print(f"  💾 summary_all_scenarios.json")
    print(f"  📄 report_scenarios.html")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
