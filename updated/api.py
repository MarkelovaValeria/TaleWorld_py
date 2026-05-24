"""
api.py  —  FastAPI для BERT-ABSA моделі (XLM-RoBERTa sentence-pair)
============================================================
Запуск:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health          — перевірка стану сервера та моделі
    POST /analyze         — аналіз списку відгуків
    POST /analyze/single  — аналіз одного відгуку
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── Конфігурація ─────────────────────────────────────────────────────────────

MODEL_PATH = Path(os.getenv("MODEL_PATH", "saved_models_4_3_bert/bert_absa"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 128

ASPECTS: list[str] = ["content_quality", "clarity", "difficulty"]

ASPECT_DESCRIPTIONS: dict[str, str] = {
    "content_quality": "quality and depth of course content and materials",
    "clarity":         "clarity of explanations and teaching style",
    "difficulty":      "difficulty level, pace and workload of the course",
}

ASPECT_NAMES_UA: dict[str, str] = {
    "content_quality": "Якість контенту",
    "clarity":         "Зрозумілість пояснень",
    "difficulty":      "Складність та темп",
}

# ─── Стан сервера ─────────────────────────────────────────────────────────────

_model: Any     = None
_tokenizer: Any = None


def _load_model() -> bool:
    """Завантажує модель один раз при старті."""
    global _model, _tokenizer
    if not MODEL_PATH.exists():
        print(f"[WARN] Модель не знайдена: {MODEL_PATH}")
        return False
    print(f"Завантаження моделі з {MODEL_PATH} на {DEVICE} …")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    ).to(DEVICE)
    _model.eval()
    print("Модель завантажена ✓")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ABSA Course Review API",
    description=(
        "Аналіз відгуків на курс за трьома аспектами: "
        "якість контенту, зрозумілість, складність."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # для розробки; у production вказуй конкретний origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Схеми (Pydantic) ─────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    reviews: list[str] = Field(
        ...,
        min_length=1,
        description="Список відгуків для аналізу",
        examples=[["Great course!", "Very confusing explanations."]],
    )


class SingleReviewRequest(BaseModel):
    text: str = Field(..., description="Текст одного відгуку")


class AspectResult(BaseModel):
    label:      str   # "positive" | "negative"
    confidence: float # впевненість у прогнозі (0-1)
    pos_prob:   float # ймовірність positive (0-1)
    neg_prob:   float # ймовірність negative (0-1)
    name_ua:    str   # назва аспекту українською


class ReviewResult(BaseModel):
    id:            int
    text:          str
    aspects:       dict[str, AspectResult]
    overall_label: str   # "positive" якщо більшість аспектів positive
    overall_score: float # середня pos_prob по всіх аспектах (0-1)


class AspectSummary(BaseModel):
    name_ua:       str
    avg_pos_prob:  float  # середня ймовірність positive
    positive_rate: float  # частка відгуків з positive label
    avg_confidence: float # середня впевненість моделі


class MostInfluential(BaseModel):
    aspect:        str
    name_ua:       str
    avg_pos_prob:  float
    direction:     str   # "positive" або "negative" — в який бік впливає
    reason:        str   # пояснення


class SummaryResult(BaseModel):
    total_reviews:          int
    positive_count:         int
    negative_count:         int
    overall_average_score:  float                    # середня pos_prob по всіх
    per_aspect:             dict[str, AspectSummary]
    most_positive_aspect:   MostInfluential
    most_negative_aspect:   MostInfluential
    most_influential_aspect: MostInfluential          # аспект з найбільшим відхиленням


class AnalyzeResponse(BaseModel):
    reviews: list[ReviewResult]
    summary: SummaryResult


# ─── Інференс ────────────────────────────────────────────────────────────────

def _predict_single_aspect(text: str, aspect: str) -> dict[str, Any]:
    """Один прохід моделі для конкретного аспекту."""
    enc = _tokenizer(
        text,
        ASPECT_DESCRIPTIONS[aspect],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    probs    = torch.softmax(logits, dim=1)[0].cpu().tolist()
    label_id = int(logits.argmax(dim=1).item())
    label    = "positive" if label_id == 1 else "negative"
    conf     = probs[label_id]
    return {
        "label":      label,
        "confidence": round(conf, 4),
        "pos_prob":   round(probs[1], 4),
        "neg_prob":   round(probs[0], 4),
        "name_ua":    ASPECT_NAMES_UA[aspect],
    }


def _analyze_text(text: str) -> dict[str, Any]:
    """Аналіз одного відгуку по всіх аспектах."""
    aspects: dict[str, Any] = {}
    for asp in ASPECTS:
        aspects[asp] = _predict_single_aspect(text, asp)

    # Загальний label — мажоритарне голосування
    pos_count = sum(1 for v in aspects.values() if v["label"] == "positive")
    overall_label = "positive" if pos_count > len(ASPECTS) / 2 else "negative"

    # Загальний score — середня pos_prob
    overall_score = round(
        sum(v["pos_prob"] for v in aspects.values()) / len(ASPECTS), 4
    )
    return {
        "aspects":       aspects,
        "overall_label": overall_label,
        "overall_score": overall_score,
    }


def _build_summary(results: list[ReviewResult]) -> SummaryResult:
    """Будує підсумкову статистику по всіх відгуках."""
    n = len(results)

    # Збираємо pos_prob і confidence по кожному аспекту
    asp_pos_probs:   dict[str, list[float]] = {a: [] for a in ASPECTS}
    asp_confs:       dict[str, list[float]] = {a: [] for a in ASPECTS}
    asp_pos_counts:  dict[str, int]         = {a: 0  for a in ASPECTS}

    overall_scores: list[float] = []
    positive_count  = 0
    negative_count  = 0

    for r in results:
        overall_scores.append(r.overall_score)
        if r.overall_label == "positive":
            positive_count += 1
        else:
            negative_count += 1
        for asp in ASPECTS:
            ar = r.aspects[asp]
            asp_pos_probs[asp].append(ar.pos_prob)
            asp_confs[asp].append(ar.confidence)
            if ar.label == "positive":
                asp_pos_counts[asp] += 1

    # Будуємо AspectSummary для кожного аспекту
    per_aspect: dict[str, AspectSummary] = {}
    for asp in ASPECTS:
        probs  = asp_pos_probs[asp]
        confs  = asp_confs[asp]
        per_aspect[asp] = AspectSummary(
            name_ua       = ASPECT_NAMES_UA[asp],
            avg_pos_prob  = round(sum(probs) / n, 4),
            positive_rate = round(asp_pos_counts[asp] / n, 4),
            avg_confidence= round(sum(confs) / n, 4),
        )

    overall_avg = round(sum(overall_scores) / n, 4)

    # Нейтральна точка — 0.5
    # Найбільш позитивний аспект — найбільша avg_pos_prob
    # Найбільш негативний аспект — найменша avg_pos_prob
    # Найвпливовіший — найбільше відхилення від нейтралі (|avg - 0.5|)

    sorted_by_pos  = sorted(ASPECTS, key=lambda a: per_aspect[a].avg_pos_prob, reverse=True)
    sorted_by_neg  = sorted(ASPECTS, key=lambda a: per_aspect[a].avg_pos_prob)
    sorted_by_dev  = sorted(ASPECTS,
                            key=lambda a: abs(per_aspect[a].avg_pos_prob - 0.5),
                            reverse=True)

    def _make_influential(asp: str, direction: str, context: str) -> MostInfluential:
        avg = per_aspect[asp].avg_pos_prob
        rate = per_aspect[asp].positive_rate
        pos_pct = int(rate * 100)
        if direction == "positive":
            reason = (
                f"{ASPECT_NAMES_UA[asp]}: {pos_pct}% відгуків позитивні, "
                f"середня впевненість у позитивному {avg:.0%}. {context}"
            )
        else:
            reason = (
                f"{ASPECT_NAMES_UA[asp]}: лише {pos_pct}% відгуків позитивні, "
                f"середня впевненість у позитивному лише {avg:.0%}. {context}"
            )
        return MostInfluential(
            aspect       = asp,
            name_ua      = ASPECT_NAMES_UA[asp],
            avg_pos_prob = avg,
            direction    = direction,
            reason       = reason,
        )

    most_pos_asp = sorted_by_pos[0]
    most_neg_asp = sorted_by_neg[0]
    most_inf_asp = sorted_by_dev[0]
    inf_dir      = "positive" if per_aspect[most_inf_asp].avg_pos_prob >= 0.5 else "negative"

    return SummaryResult(
        total_reviews         = n,
        positive_count        = positive_count,
        negative_count        = negative_count,
        overall_average_score = overall_avg,
        per_aspect            = per_aspect,
        most_positive_aspect  = _make_influential(
            most_pos_asp, "positive",
            "Цей аспект отримав найвищі оцінки від студентів."
        ),
        most_negative_aspect  = _make_influential(
            most_neg_asp, "negative",
            "Цей аспект викликав найбільше незадоволення."
        ),
        most_influential_aspect = _make_influential(
            most_inf_asp, inf_dir,
            "Цей аспект найсильніше відхиляється від нейтральної позиції."
        ),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Перевірка стану сервера")
def health():
    """Повертає статус сервера та чи завантажена модель."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path":   str(MODEL_PATH),
        "device":       DEVICE,
        "aspects":      ASPECTS,
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Аналіз списку відгуків",
)
def analyze(body: ReviewRequest):
    """
    Аналізує кожен відгук зі списку за трьома аспектами (content_quality,
    clarity, difficulty) та повертає:

    - **reviews** — результат по кожному відгуку
    - **summary** — загальна статистика, який аспект найбільше вплинув
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Модель не завантажена. Перевірте шлях: {MODEL_PATH}",
        )

    results: list[ReviewResult] = []
    for idx, text in enumerate(body.reviews, start=1):
        text = text.strip()
        if not text:
            raise HTTPException(
                status_code=422, detail=f"Відгук #{idx} порожній."
            )
        analysis = _analyze_text(text)
        results.append(
            ReviewResult(
                id            = idx,
                text          = text,
                aspects       = {
                    asp: AspectResult(**vals)
                    for asp, vals in analysis["aspects"].items()
                },
                overall_label = analysis["overall_label"],
                overall_score = analysis["overall_score"],
            )
        )

    summary = _build_summary(results)
    return AnalyzeResponse(reviews=results, summary=summary)


@app.post(
    "/analyze/single",
    summary="Аналіз одного відгуку",
)
def analyze_single(body: SingleReviewRequest):
    """
    Зручний endpoint для аналізу одного відгуку.
    Повертає аспекти + overall_score без summary.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Модель не завантажена. Перевірте шлях: {MODEL_PATH}",
        )
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Текст відгуку порожній.")

    analysis = _analyze_text(text)
    return {
        "text":          text,
        "aspects":       analysis["aspects"],
        "overall_label": analysis["overall_label"],
        "overall_score": analysis["overall_score"],
    }
