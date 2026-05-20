import pickle
import random
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
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "processed_reviews_labeled.csv"
RESULTS_DIR = BASE_DIR / "results_4_3"
MODELS_DIR  = BASE_DIR / "saved_models_4_3"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class Config:
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    FOLDS        = 5
    BATCH_SIZE   = 64
    EPOCHS       = 15
    EMB_DIM      = 100
    HIDDEN_DIM   = 128
    MAX_LEN      = 50
    LR           = 1e-3
    DROPOUT      = 0.3
    TOP_K_TOKENS = 5
    MAX_ASPECT_TERMS = 30
    DOMAIN_LOSS_WEIGHT = 1.5

cfg = Config()
print(f"Device: {cfg.DEVICE}")

DOMAIN_ONTOLOGY: dict[str, dict] = {
    "content_quality": {
        "description": "Якість навчального контенту, матеріалів та структури курсу",
        "sub_aspects": {
            "relevance": {
                "boost": 1.6,
                "terms": {
                    "en": {"content","material","topic","subject","syllabus",
                           "curriculum","relevant","outdated","updated","comprehensive",
                           "thorough","shallow","depth","theory","practice",
                           "concept","knowledge","information","coverage"},
                    "es": {"contenido","material","tema","programa","relevante",
                           "actualizado","completo","profundidad","teoría","práctica"},
                    "ua": {"контент","матеріал","тема","програма","актуальний",
                           "оновлений","всебічний","глибина","теорія","практика"},
                    "ru": {"контент","материал","тема","программа","актуальный",
                           "обновлённый","всесторонний","глубина","теория","практика"},
                }
            },
            "resources": {
                "boost": 1.5,
                "terms": {
                    "en": {"resource","handout","slide","notebook","transcript",
                           "video","pdf","reading","book","article","paper",
                           "reference","example","code","dataset","repository",
                           "exercise","quiz","assignment","project","lab"},
                    "es": {"recurso","diapositiva","cuaderno","video","libro",
                           "artículo","ejemplo","código","ejercicio","cuestionario"},
                    "ua": {"ресурс","слайди","зошит","відео","книга","стаття",
                           "приклад","код","вправа","тест","завдання","проєкт"},
                    "ru": {"ресурс","слайды","тетрадь","видео","книга","статья",
                           "пример","код","упражнение","тест","задание","проект"},
                }
            },
            "structure": {
                "boost": 1.4,
                "terms": {
                    "en": {"structure","organize","module","chapter","section",
                           "unit","week","lesson","lecture","sequence","flow",
                           "logical","coherent","scattered","disorganized"},
                    "es": {"estructura","organizar","módulo","capítulo","sección",
                           "lección","secuencia","lógico","coherente"},
                    "ua": {"структура","організація","модуль","розділ","секція",
                           "урок","послідовність","логічний","зв'язний"},
                    "ru": {"структура","организация","модуль","раздел","секция",
                           "урок","последовательность","логичный","связный"},
                }
            },
        }
    },

    "clarity": {
        "description": "Зрозумілість подачі, якість пояснень і комунікації",
        "sub_aspects": {
            "explanation": {
                "boost": 1.6,
                "terms": {
                    "en": {"clear","clarity","explain","explanation","understandable",
                           "confusing","unclear","vague","ambiguous","straightforward",
                           "concise","verbose","simple","complex","intuitive",
                           "well-explained","poorly-explained","illustrate"},
                    "es": {"claro","claridad","explicar","explicación","comprensible",
                           "confuso","vago","simple","intuitivo","bien explicado"},
                    "ua": {"зрозумілий","ясність","пояснення","незрозумілий","чіткий",
                           "розпливчатий","простий","складний","інтуїтивний"},
                    "ru": {"понятный","ясность","объяснение","непонятный","чёткий",
                           "расплывчатый","простой","сложный","интуитивный"},
                }
            },
            "communication": {
                "boost": 1.5,
                "terms": {
                    "en": {"instructor","teacher","professor","mentor","tutor",
                           "lecturer","speaker","presenter","engaging","boring",
                           "monotone","enthusiastic","passionate","charismatic",
                           "accent","pronunciation","articulate","mumble","pace",
                           "slow","fast","language","vocabulary","terminology"},
                    "es": {"instructor","profesor","aburrido","entusiasta","acento",
                           "ritmo","lenguaje","vocabulario"},
                    "ua": {"викладач","вчитель","нудний","захоплений","акцент",
                           "темп","мова","словниковий запас"},
                    "ru": {"преподаватель","учитель","скучный","увлечённый","акцент",
                           "темп","язык","словарный запас"},
                }
            },
            "examples": {
                "boost": 1.4,
                "terms": {
                    "en": {"example","illustration","demo","demonstration","case",
                           "analogy","metaphor","visual","diagram","chart","graph",
                           "real-world","practical","application","hands-on"},
                    "es": {"ejemplo","ilustración","demostración","caso","analogía",
                           "visual","diagrama","práctico","aplicación"},
                    "ua": {"приклад","ілюстрація","демонстрація","кейс","аналогія",
                           "візуальний","діаграма","практичний","застосування"},
                    "ru": {"пример","иллюстрация","демонстрация","кейс","аналогия",
                           "визуальный","диаграмма","практический","применение"},
                }
            },
        }
    },

    "difficulty": {
        "description": "Складність матеріалу, темп курсу і навчальне навантаження",
        "sub_aspects": {
            "level": {
                "boost": 1.6,
                "terms": {
                    "en": {"difficulty","difficult","easy","hard","challenging",
                           "beginner","intermediate","advanced","expert","level",
                           "appropriate","overwhelming","too basic","too advanced",
                           "steep","learning curve","prerequisite","background"},
                    "es": {"dificultad","difícil","fácil","desafiante","principiante",
                           "intermedio","avanzado","nivel","abrumador","curva de aprendizaje"},
                    "ua": {"складність","складний","легкий","важкий","виклик",
                           "початківець","середній","просунутий","рівень",
                           "перевантажений","крива навчання","передумова"},
                    "ru": {"сложность","сложный","лёгкий","трудный","вызов",
                           "начинающий","средний","продвинутый","уровень",
                           "перегруженный","кривая обучения","предпосылка"},
                }
            },
            "pace": {
                "boost": 1.5,
                "terms": {
                    "en": {"pace","speed","fast","slow","rush","quick","gradual",
                           "overwhelming","manageable","reasonable","too much",
                           "too fast","too slow","pressure","time","duration"},
                    "es": {"ritmo","velocidad","rápido","lento","apresurado",
                           "abrumador","manejable","tiempo","duración"},
                    "ua": {"темп","швидкість","швидко","повільно","поспіх",
                           "перевантажений","керований","час","тривалість"},
                    "ru": {"темп","скорость","быстро","медленно","спешка",
                           "перегруженный","управляемый","время","продолжительность"},
                }
            },
            "workload": {
                "boost": 1.4,
                "terms": {
                    "en": {"workload","homework","assignment","task","project",
                           "quiz","exam","test","deadline","effort","time-consuming",
                           "practical","hands-on","repetitive","tedious","engaging"},
                    "es": {"carga de trabajo","tarea","proyecto","prueba","examen",
                           "plazo","esfuerzo","práctico","repetitivo","tedioso"},
                    "ua": {"навантаження","домашнє завдання","завдання","проєкт",
                           "тест","іспит","дедлайн","зусилля","практичний","нудний"},
                    "ru": {"нагрузка","домашнее задание","задание","проект",
                           "тест","экзамен","дедлайн","усилие","практический","скучный"},
                }
            },
        }
    },
}

TERM_INDEX: dict[str, tuple[str, str, float]] = {}
ALL_DOMAIN_TERMS: set[str] = set()

for _asp, _asp_data in DOMAIN_ONTOLOGY.items():
    for _sub, _sub_data in _asp_data["sub_aspects"].items():
        for _lang_terms in _sub_data["terms"].values():
            for _term in _lang_terms:
                if _term not in TERM_INDEX:
                    TERM_INDEX[_term] = (_asp, _sub, _sub_data["boost"])
                ALL_DOMAIN_TERMS.add(_term)

class DomainKnowledgeScorer:

    def _get_domain_positions(self, words: list[str]) -> list[int]:
        return [i for i, w in enumerate(words) if w in ALL_DOMAIN_TERMS]

    def _positional_weight(self, idx: int, domain_positions: list[int]) -> float:
        if not domain_positions:
            return 1.0
        min_dist = min(abs(idx - dp) for dp in domain_positions)
        return 1.0 / (1.0 + min_dist)

    def _boost(self, word: str, aspect: str) -> float:
        entry = TERM_INDEX.get(word)
        if entry is None:
            return 1.0
        mapped_aspect, _, boost_val = entry
        return boost_val if mapped_aspect == aspect else 0.8

    def score(self, words: list[str], aspect: str, fasttext_model) -> float:
        domain_pos = self._get_domain_positions(words)
        total, weight_sum = 0.0, 0.0
        for i, w in enumerate(words):
            try:
                sim = fasttext_model.wv.similarity(w, aspect)
                pos = self._positional_weight(i, domain_pos)
                bst = self._boost(w, aspect)
                wt  = pos * bst
                total      += sim * wt
                weight_sum += wt
            except KeyError:
                pass
        return total / weight_sum if weight_sum > 0 else 0.0

    def detected_aspects(self, text: str) -> dict[str, list[str]]:
        words = text.lower().split()
        found: dict[str, list[str]] = {a: [] for a in DOMAIN_ONTOLOGY}
        for w in words:
            if w in TERM_INDEX:
                asp, _, _ = TERM_INDEX[w]
                found[asp].append(w)
        return found

class Vocabulary:
    def __init__(self, freq_threshold: int = 3, max_size: int = 15000):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.lower().strip() for t in str(text).split() if t.strip()]

    def build(self, texts: list[str]):
        freq: dict[str, int] = {}
        for t in texts:
            for w in self.tokenize(t):
                freq[w] = freq.get(w, 0) + 1
        freq = {k: v for k, v in freq.items() if v >= self.freq_threshold}
        freq = dict(sorted(freq.items(), key=lambda x: -x[1])[: self.max_size - 4])
        for idx, word in enumerate(freq, start=4):
            self.stoi[word] = idx
            self.itos[idx]  = word

    def numericalize(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.stoi["<UNK>"]) for w in self.tokenize(text)]

class ReviewDataset(Dataset):

    def __init__(self, df: pd.DataFrame, text_col: str, aspects: list[str]):
        self.df      = df.reset_index(drop=True)
        self.texts   = self.df[text_col]
        self.aspects = aspects
        self.vocab   = Vocabulary()
        self.vocab.build(self.texts.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text   = self.texts.iloc[idx]
        labels = {asp: float(self.df[f"{asp}_y"].iloc[idx])        for asp in self.aspects}
        dw     = {asp: float(self.df[f"dw_{asp}"].iloc[idx])       for asp in self.aspects}
        num    = [self.vocab.stoi["<SOS>"]]
        num   += self.vocab.numericalize(text)
        num.append(self.vocab.stoi["<EOS>"])
        return torch.tensor(num, dtype=torch.long), labels, dw

class PadCollate:
    def __init__(self, pad_idx: int, maxlen: int):
        self.pad_idx = pad_idx
        self.maxlen  = maxlen

    def __call__(self, batch):
        padded = torch.zeros(self.maxlen, len(batch), dtype=torch.long)
        for j, (seq, _, _) in enumerate(batch):
            L = min(len(seq), self.maxlen)
            padded[:L, j] = seq[:L]
        aspects = list(batch[0][1].keys())
        labels_dict = {
            asp: torch.tensor([item[1][asp] for item in batch], dtype=torch.float32)
            for asp in aspects
        }
        dw_dict = {
            asp: torch.tensor([item[2][asp] for item in batch], dtype=torch.float32)
            for asp in aspects
        }
        return padded, labels_dict, dw_dict

class _AspectAttn(nn.Module):
    TEMPERATURE = 2.0

    def __init__(self, hidden_dim: int, aspect_emb_dim: int):
        super().__init__()
        self.W_h = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.W_a = nn.Linear(aspect_emb_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        lstm_out:   torch.Tensor,   
        aspect_vec: torch.Tensor,   
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_proj = self.W_h(lstm_out)                         
        a_proj = self.W_a(aspect_vec).unsqueeze(0)          
        score  = self.v(torch.tanh(h_proj + a_proj))        
        score  = score.squeeze(-1)                           
        alpha  = F.softmax(score / self.TEMPERATURE, dim=0).transpose(0, 1)  
        context = torch.bmm(
            alpha.unsqueeze(1),
            lstm_out.permute(1, 0, 2),
        ).squeeze(1)                                         
        return context, alpha

class PerAspectABSAModel(nn.Module):
    GATE_DIM = 64

    def __init__(
        self,
        vocab_size:     int,
        emb_dim:        int,
        hidden_dim:     int,
        emb_layer:      nn.Embedding,
        aspect_emb_dim: int,
        n_aspects:      int,
        dropout:        float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_aspects  = n_aspects
        self.embedding  = emb_layer
        self.lstm       = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=False)
        self.dropout    = nn.Dropout(dropout)

        self.attentions = nn.ModuleList([
            _AspectAttn(hidden_dim, aspect_emb_dim) for _ in range(n_aspects)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(aspect_emb_dim, self.GATE_DIM) for _ in range(n_aspects)
        ])
        self.fc1s = nn.ModuleList([
            nn.Linear(2 * hidden_dim + self.GATE_DIM, 64) for _ in range(n_aspects)
        ])
        self.fc2s = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(n_aspects)
        ])

    def encode(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Спільний Bi-LSTM прохід. Викликається один раз на батч."""
        _, N = text.shape
        h0 = torch.zeros(2, N, self.hidden_dim, device=text.device)
        c0 = torch.zeros(2, N, self.hidden_dim, device=text.device)
        emb = self.embedding(text)
        lstm_out, (h_n, _) = self.lstm(emb, (h0, c0))
        last_h = torch.cat([h_n[0], h_n[1]], dim=-1)   
        return lstm_out, last_h

    def forward_aspect(
        self,
        lstm_out:   torch.Tensor,  
        last_h:     torch.Tensor,  
        aspect_vec: torch.Tensor,  
        asp_idx:    int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Класифікація для одного конкретного аспекту."""
        context, alpha = self.attentions[asp_idx](lstm_out, aspect_vec)
        repr_    = context + last_h                          
        gate     = F.relu(self.gates[asp_idx](aspect_vec))  
        combined = self.dropout(torch.cat([repr_, gate], dim=-1))
        y        = F.relu(self.fc1s[asp_idx](combined))
        y        = torch.sigmoid(self.fc2s[asp_idx](y)).squeeze(-1)  
        return y, alpha

    def forward(
        self,
        text:        torch.Tensor,       
        aspect_vecs: list[torch.Tensor], 
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        lstm_out, last_h = self.encode(text)
        outputs, alphas = [], []
        for i, av in enumerate(aspect_vecs):
            out, alpha = self.forward_aspect(lstm_out, last_h, av, i)
            outputs.append(out)
            alphas.append(alpha)
        return outputs, alphas

def _build_fasttext(df: pd.DataFrame, ft_path: Path) -> FastText:
    if ft_path.exists():
        print(f"Завантаження FastText: {ft_path}")
        return FastText.load(str(ft_path))
    print("Навчання FastText...")
    if "token" not in df.columns:
        df["token"] = df["clean_text"].apply(str.split)
    ft = FastText(
        sentences=df["token"].tolist(),
        vector_size=cfg.EMB_DIM,
        window=5,
        min_count=3,
        workers=4,
        sg=1,
        epochs=10,
    )
    ft.save(str(ft_path))
    print(f"FastText збережено: {ft_path}")
    return ft

def _build_emb_layer(vocab: Vocabulary, ft: FastText, trainable: bool = False) -> nn.Embedding:
    W = np.zeros((len(vocab), cfg.EMB_DIM), dtype=np.float32)
    for word, idx in vocab.stoi.items():
        try:
            W[idx] = ft.wv[word]
        except KeyError:
            pass
    layer = nn.Embedding.from_pretrained(torch.tensor(W))
    layer.weight.requires_grad = trainable
    return layer

class ABSAModelWithAttention:

    ASPECTS = list(DOMAIN_ONTOLOGY.keys())

    def __init__(
        self,
        df:             pd.DataFrame,
        fasttext_model: FastText,
        aspects:        list[str] | None = None,
        text_col:       str = "clean_text",
        primary_aspect: str = "content_quality",
    ):
        self.df      = df.copy()
        self.ft      = fasttext_model
        self.aspects = aspects or self.ASPECTS
        self.text_col = text_col
        self.primary  = primary_aspect
        self.domain   = DomainKnowledgeScorer()

        self.dataset:       ReviewDataset | None = None
        self.model:         PerAspectABSAModel | None = None
        self.aspect_vecs:   dict[str, torch.Tensor] = {}
        self.best_f1:       float | None = None
        self._train_history: list[dict] = []

    def _prepare_labels(self):
        """Бінаризує загальний Label і per-aspect мітки (neutral → fallback до y)."""
        if "y" not in self.df.columns:
            self.df["y"] = (
                self.df["Label"].astype(str).str.lower() == "positive"
            ).astype(int)
        for asp in self.aspects:
            y_col   = f"{asp}_y"
            lbl_col = f"label_{asp}"
            if y_col not in self.df.columns:
                if lbl_col in self.df.columns:
                    lbl = self.df[lbl_col].astype(str).str.lower()
                    self.df[y_col] = np.where(
                        lbl == "neutral",
                        self.df["y"],
                        (lbl == "positive").astype(int),
                    )
                    print(f"  [{asp}] label з '{lbl_col}' "
                          f"(neutral→fallback: {(lbl=='neutral').sum():,})")
                else:
                    self.df[y_col] = self.df["y"]
                    print(f"  [{asp}] fallback → загальний Label")

    def _build_aspect_vec(self, aspect: str) -> torch.Tensor:
        """
        Аспектний вектор = середнє FastText-векторів domain-термінів.

        Стара версія: ft.wv["content_quality"] — вектор одного (OOV) слова.
        Нова версія:  середнє векторів усіх EN-термінів онтології аспекту.

        Різниця між аспектними векторами в старій версії: ~0.99 cosine similarity.
        Після fix: ~0.70-0.85 — аспекти справді різні у векторному просторі.
        """
        vecs = []
        for sub_data in DOMAIN_ONTOLOGY[aspect]["sub_aspects"].values():
            for term in sub_data["terms"].get("en", set()):
                try:
                    vecs.append(self.ft.wv[term])
                except KeyError:
                    pass

        if len(vecs) > cfg.MAX_ASPECT_TERMS:
            random.shuffle(vecs)
            vecs = vecs[:cfg.MAX_ASPECT_TERMS]

        if vecs:
            v = np.mean(vecs, axis=0).astype(np.float32)
        else:
            try:
                v = self.ft.wv[aspect].astype(np.float32)
            except KeyError:
                v = np.zeros(cfg.EMB_DIM, dtype=np.float32)

        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)                

    def _build_domain_weights(self):
        """
        Для кожного аспекту обчислює вагу loss по відгукам:
        - є domain-терміни аспекту → DOMAIN_LOSS_WEIGHT (1.5)
        - немає → 1.0

        Ідея: відгук "the pace is overwhelming" явно про difficulty → loss
        цього відгуку важливіший для навчання difficulty-класифікатора.
        """
        print("\nОбчислення domain weights...")
        for asp in self.aspects:
            col = f"dw_{asp}"
            if col not in self.df.columns:
                weights = []
                for text in self.df[self.text_col].astype(str):
                    det = self.domain.detected_aspects(text)
                    weights.append(
                        cfg.DOMAIN_LOSS_WEIGHT if det[asp] else 1.0
                    )
                self.df[col] = weights
                n_w = sum(1 for w in weights if w > 1.0)
                print(f"  [{asp}]: {n_w:,} відгуків з domain-термінами ({n_w/len(weights):.1%})")

    def fit(self) -> float:
        """
        Навчання з KFold та одночасним multi-task loss по всіх аспектах.

        На кожному батчі:
          1. Один спільний прохід Bi-LSTM
          2. Три паралельних per-aspect forward_aspect (окремі параметри)
          3. Три domain-weighted BCE loss → сума → backward
          4. Gradient через спільний LSTM + три per-aspect шляхи
        """
        self._prepare_labels()
        self._build_domain_weights()

        self.dataset  = ReviewDataset(self.df, self.text_col, self.aspects)
        vocab         = self.dataset.vocab
        emb_layer     = _build_emb_layer(vocab, self.ft)

        print("\nБудуємо аспектні вектори (mean domain terms)...")
        self.aspect_vecs = {a: self._build_aspect_vec(a) for a in self.aspects}
        n_asp = len(self.aspects)

        avs = [self.aspect_vecs[a].squeeze(0).numpy() for a in self.aspects]
        for i in range(n_asp):
            for j in range(i + 1, n_asp):
                cos = float(np.dot(avs[i], avs[j]) /
                            (np.linalg.norm(avs[i]) * np.linalg.norm(avs[j]) + 1e-9))
                print(f"  cos({self.aspects[i]}, {self.aspects[j]}) = {cos:.3f}")

        kfold    = KFold(n_splits=cfg.FOLDS, shuffle=True, random_state=42)
        fold_f1s = []
        prim_idx = self.aspects.index(self.primary)

        for fold, (tr_idx, val_idx) in enumerate(
            kfold.split(np.arange(len(self.dataset)))
        ):
            print(f"\n{'─'*50}")
            print(f"  Fold {fold+1}/{cfg.FOLDS}")
            print(f"{'─'*50}")

            tr_loader = DataLoader(
                self.dataset, batch_size=cfg.BATCH_SIZE,
                sampler=SubsetRandomSampler(tr_idx),
                collate_fn=PadCollate(0, cfg.MAX_LEN),
            )
            val_loader = DataLoader(
                self.dataset, batch_size=cfg.BATCH_SIZE,
                sampler=SubsetRandomSampler(val_idx),
                collate_fn=PadCollate(0, cfg.MAX_LEN),
            )

            self.model = PerAspectABSAModel(
                vocab_size=len(vocab),
                emb_dim=cfg.EMB_DIM,
                hidden_dim=cfg.HIDDEN_DIM,
                emb_layer=emb_layer,
                aspect_emb_dim=cfg.EMB_DIM,
                n_aspects=n_asp,
                dropout=cfg.DROPOUT,
            ).to(cfg.DEVICE)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )

            tr_losses, val_losses, tr_accs, val_accs = [], [], [], []

            for epoch in range(cfg.EPOCHS):

                self.model.train()
                tr_loss, tr_correct, tr_total = 0.0, 0, 0

                for review, labels_dict, dw_dict in tqdm(
                    tr_loader, leave=False, desc=f"Epoch {epoch+1:02d} train"
                ):
                    review = review.to(cfg.DEVICE)
                    B      = review.size(1)

                    lstm_out, last_h = self.model.encode(review)

                    losses     = []
                    prim_out   = None
                    prim_label = None

                    for i, asp in enumerate(self.aspects):
                        av    = self.aspect_vecs[asp].expand(B, -1).to(cfg.DEVICE)
                        label = labels_dict[asp].to(cfg.DEVICE)
                        dw    = dw_dict[asp].to(cfg.DEVICE)

                        out, alpha = self.model.forward_aspect(lstm_out, last_h, av, i)

                        bce  = F.binary_cross_entropy(out, label, reduction="none")
                        loss = (bce * dw).mean()

                        entropy = -(alpha * torch.log(alpha + 1e-9)).sum(dim=1).mean()
                        loss    = loss - 0.05 * entropy

                        losses.append(loss)

                        if asp == self.primary:
                            prim_out   = out.detach()
                            prim_label = label

                    total_loss = sum(losses)

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    tr_loss    += total_loss.item() * B
                    tr_correct += ((prim_out > 0.5).float() == prim_label).sum().item()
                    tr_total   += B

                self.model.eval()
                v_loss, v_correct = 0.0, 0
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for review, labels_dict, dw_dict in val_loader:
                        review = review.to(cfg.DEVICE)
                        B      = review.size(1)

                        lstm_out, last_h = self.model.encode(review)

                        losses     = []
                        prim_out   = None
                        prim_label = None

                        for i, asp in enumerate(self.aspects):
                            av    = self.aspect_vecs[asp].expand(B, -1).to(cfg.DEVICE)
                            label = labels_dict[asp].to(cfg.DEVICE)
                            dw    = dw_dict[asp].to(cfg.DEVICE)

                            out, _ = self.model.forward_aspect(lstm_out, last_h, av, i)
                            bce    = F.binary_cross_entropy(out, label, reduction="none")
                            losses.append((bce * dw).mean())

                            if asp == self.primary:
                                prim_out   = out
                                prim_label = label

                        v_loss    += sum(losses).item() * B
                        preds      = (prim_out > 0.5).float()
                        v_correct += (preds == prim_label).sum().item()
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(prim_label.cpu().tolist())

                n_tr = len(tr_loader.sampler)
                n_v  = len(val_loader.sampler)
                tr_loss /= n_tr * n_asp
                v_loss  /= n_v  * n_asp
                tr_acc   = tr_correct / tr_total * 100
                v_acc    = v_correct  / n_v * 100

                tr_losses.append(tr_loss);  val_losses.append(v_loss)
                tr_accs.append(tr_acc);     val_accs.append(v_acc)
                scheduler.step(v_loss)

                print(
                    f"  Epoch {epoch+1:02d} | "
                    f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1f}% | "
                    f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%"
                )

            fold_f1 = f1_score(all_labels, all_preds, average="weighted")
            fold_f1s.append(fold_f1)
            print(f"  → Fold F1 (weighted, primary={self.primary}): {fold_f1:.4f}")

            self._train_history.append({
                "fold":       fold + 1,
                "tr_losses":  tr_losses,
                "val_losses": val_losses,
                "tr_accs":    tr_accs,
                "val_accs":   val_accs,
                "fold_f1":    fold_f1,
                "all_preds":  all_preds,
                "all_labels": all_labels,
            })

        self.best_f1 = float(np.mean(fold_f1s))
        print(f"\n{'═'*50}")
        print(f"  Per-Aspect ABSA  Mean F1: {self.best_f1:.4f}")
        print(f"{'═'*50}")
        return self.best_f1

    def eval_per_aspect(self, sample_size: int = 2000) -> dict[str, dict]:
        """
        Оцінює кожен аспект на per-aspect мітках з auto_label.py.
        Нейтральні відгуки виключаються з оцінки аспекту (немає сигналу).
        """
        if self.model is None:
            raise RuntimeError("Спочатку запустіть fit()")

        sample = self.df.sample(
            min(sample_size, len(self.df)), random_state=42
        ).reset_index(drop=True)

        per_aspect: dict[str, dict] = {}
        self.model.eval()

        for i, aspect in enumerate(self.aspects):
            lbl_col = f"label_{aspect}"
            y_col   = f"{aspect}_y"

            if lbl_col in sample.columns:
                eval_df = sample[sample[lbl_col].str.lower() != "neutral"].copy()
                eval_df["_y"] = (eval_df[lbl_col].str.lower() == "positive").astype(int)
            else:
                eval_df = sample.copy()
                eval_df["_y"] = sample[y_col].astype(int)

            if len(eval_df) == 0:
                continue

            preds, labels = [], []
            with torch.no_grad():
                for text, true_label in zip(
                    eval_df[self.text_col].astype(str), eval_df["_y"]
                ):
                    tensor = self._text_to_tensor(text)
                    av     = self.aspect_vecs[aspect].to(cfg.DEVICE)

                    lstm_out, last_h = self.model.encode(tensor)
                    out, _           = self.model.forward_aspect(lstm_out, last_h, av, i)

                    preds.append(int(float(out.item()) > 0.5))
                    labels.append(int(true_label))

            per_aspect[aspect] = {
                "f1":           round(f1_score(labels, preds, average="weighted"), 4),
                "accuracy":     round(accuracy_score(labels, preds), 4),
                "positive_rate": round(sum(preds) / max(len(preds), 1), 4),
                "n_eval":       len(labels),
            }
            print(
                f"  {aspect:<22}: F1={per_aspect[aspect]['f1']:.4f}  "
                f"Acc={per_aspect[aspect]['accuracy']:.4f}  "
                f"N={per_aspect[aspect]['n_eval']}  "
                f"Pos%={per_aspect[aspect]['positive_rate']:.2%}"
            )

        return per_aspect

    def save(self, model_dir: Path = MODELS_DIR):
        torch.save(self.model.state_dict(), model_dir / "absa_attention.pt")
        with open(model_dir / "absa_attention_vocab.pkl", "wb") as f:
            pickle.dump(
                {"stoi": self.dataset.vocab.stoi, "itos": self.dataset.vocab.itos}, f
            )
        print(f"Модель збережена у {model_dir}")

    def load(self, model_dir: Path = MODELS_DIR):
        with open(model_dir / "absa_attention_vocab.pkl", "rb") as f:
            vdata = pickle.load(f)
        vocab = Vocabulary()
        vocab.stoi = vdata["stoi"]
        vocab.itos = vdata["itos"]

        emb_layer = _build_emb_layer(vocab, self.ft)
        self.model = PerAspectABSAModel(
            vocab_size=len(vocab),
            emb_dim=cfg.EMB_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            emb_layer=emb_layer,
            aspect_emb_dim=cfg.EMB_DIM,
            n_aspects=len(self.aspects),
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)
        self.model.load_state_dict(
            torch.load(model_dir / "absa_attention.pt", map_location=cfg.DEVICE)
        )
        self.model.eval()

        class _FakeDataset:
            pass
        ds = _FakeDataset()
        ds.vocab = vocab
        self.dataset = ds

        self.aspect_vecs = {a: self._build_aspect_vec(a) for a in self.aspects}
        print("Модель завантажена.")

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        vocab = self.dataset.vocab
        num   = [vocab.stoi["<SOS>"]]
        num  += vocab.numericalize(text.lower())
        num.append(vocab.stoi["<EOS>"])
        if len(num) < cfg.MAX_LEN:
            num += [vocab.stoi["<PAD>"]] * (cfg.MAX_LEN - len(num))
        else:
            num = num[:cfg.MAX_LEN]
        return torch.tensor(num, dtype=torch.long).unsqueeze(1).to(cfg.DEVICE)

    def analyze_review(self, text: str, top_k: int | None = None) -> dict[str, dict]:
        """
        Один спільний Bi-LSTM прохід → три незалежних per-aspect класифікації.
        Кожен аспект використовує свої параметри уваги і FC-голову.
        """
        top_k  = top_k or cfg.TOP_K_TOKENS
        tokens = text.lower().split()
        tensor = self._text_to_tensor(text)
        results = {}

        self.model.eval()
        with torch.no_grad():
                                                           
            lstm_out, last_h = self.model.encode(tensor)

            for i, aspect in enumerate(self.aspects):
                av = self.aspect_vecs[aspect].to(cfg.DEVICE)
                                                                    
                out, alpha = self.model.forward_aspect(lstm_out, last_h, av, i)

                attn = alpha.squeeze(0).cpu().tolist()
                token_attn: list[tuple[str, float]] = []
                for j, tok in enumerate(tokens[: cfg.MAX_LEN - 2]):
                    pos = j + 1
                    if pos < len(attn):
                        token_attn.append((tok, attn[pos]))
                token_attn.sort(key=lambda x: x[1], reverse=True)

                sim = self.domain.score(tokens, aspect, self.ft)
                det = self.domain.detected_aspects(text)
                score = float(out.item())

                results[aspect] = {
                    "sentiment":    score,
                    "label":        "positive" if score > 0.5 else "negative",
                    "similarity":   float(sim),
                    "combined":     score * float(sim),
                    "attention":    attn,
                    "top_tokens":   token_attn[:top_k],
                    "domain_terms": det[aspect],
                }

        return results

    def analyze_dataset_aspects(self) -> pd.DataFrame:
        rows = []
        self.model.eval()
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc="Aspect scoring"):
            text  = str(row[self.text_col])
            label = str(row.get("Label", ""))
            res   = self.analyze_review(text)
            entry = {"text": text[:80], "true_label": label}
            for asp, vals in res.items():
                entry[f"{asp}_sentiment"]  = round(vals["sentiment"], 4)
                entry[f"{asp}_similarity"] = round(vals["similarity"], 4)
                entry[f"{asp}_combined"]   = round(vals["combined"], 4)
                top3 = ", ".join(f"{t}({w:.2f})" for t, w in vals["top_tokens"][:3])
                entry[f"{asp}_top_tokens"] = top3
            rows.append(entry)
        return pd.DataFrame(rows)

class ReportGenerator:

    def __init__(self, absa: ABSAModelWithAttention, results_dir: Path = RESULTS_DIR):
        self.absa        = absa
        self.out         = results_dir
        self.image_paths: list[str] = []

    def plot_training_curves(self):
        history = self.absa._train_history
        if not history:
            return
        n_folds = len(history)
        fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 8))
        if n_folds == 1:
            axes = axes.reshape(2, 1)
        for i, h in enumerate(history):
            axes[0, i].plot(h["tr_losses"], label="Train", color="#4C9BE8")
            axes[0, i].plot(h["val_losses"], label="Val",   color="#E87B4C")
            axes[0, i].set_title(f"Fold {h['fold']} — Loss")
            axes[0, i].set_xlabel("Epoch"); axes[0, i].legend()
            axes[1, i].plot(h["tr_accs"], label="Train", color="#4C9BE8")
            axes[1, i].plot(h["val_accs"], label="Val",   color="#E87B4C")
            axes[1, i].set_title(f"Fold {h['fold']} — Accuracy (%)")
            axes[1, i].set_xlabel("Epoch"); axes[1, i].legend()
        plt.suptitle("Training Curves — Bi-LSTM + Per-Aspect Attention (multi-task)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        p = self.out / "training_curves.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_confusion_matrix(self):
        if not self.absa._train_history:
            return
        last = self.absa._train_history[-1]
        cm   = confusion_matrix(last["all_labels"], last["all_preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"], ax=ax)
        ax.set_title("Confusion Matrix (Last Fold)", fontsize=13, fontweight="bold")
        ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
        plt.tight_layout()
        p = self.out / "confusion_matrix.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_fold_f1(self):
        if not self.absa._train_history:
            return
        folds = [h["fold"]    for h in self.absa._train_history]
        f1s   = [h["fold_f1"] for h in self.absa._train_history]
        mean  = np.mean(f1s)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar([f"Fold {f}" for f in folds], f1s,
                      color="#4C9BE8", edgecolor="white")
        ax.axhline(mean, color="#E87B4C", linestyle="--", label=f"Mean F1={mean:.4f}")
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Weighted F1")
        ax.set_title("F1 по Fold — Bi-LSTM + Per-Aspect Attention", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        p = self.out / "fold_f1.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_per_aspect_f1(self, per_aspect: dict[str, dict]):
        aspects = list(per_aspect.keys())
        f1s     = [per_aspect[a]["f1"]            for a in aspects]
        accs    = [per_aspect[a]["accuracy"]       for a in aspects]
        pos_rt  = [per_aspect[a]["positive_rate"]  for a in aspects]
        x = np.arange(len(aspects)); width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - width, f1s,    width, label="F1 (weighted)", color="#4C9BE8")
        b2 = ax.bar(x,         accs,   width, label="Accuracy",      color="#5CB85C")
        b3 = ax.bar(x + width, pos_rt, width, label="Positive Rate", color="#E87B4C")
        for bars in (b1, b2, b3):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(aspects, fontsize=11)
        ax.set_ylim(0, 1.10); ax.set_ylabel("Метрика", fontsize=12)
        ax.set_title("Per-Aspect Метрики — F1 / Accuracy / Positive Rate",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right"); ax.axhline(0.80, color="gray", linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = self.out / "per_aspect_f1.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_attention_heatmaps(self, n_samples: int = 4):
        df = self.absa.df
        pos_samples = df[df["Label"].str.lower() == "positive"].sample(
            min(n_samples // 2, len(df)), random_state=1)
        neg_samples = df[df["Label"].str.lower() == "negative"].sample(
            min(n_samples // 2, len(df)), random_state=2)
        samples = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)
        aspects_to_show = self.absa.aspects[:3]
        fig, axes = plt.subplots(
            len(samples), len(aspects_to_show),
            figsize=(6 * len(aspects_to_show), 2.5 * len(samples))
        )
        if len(samples) == 1:
            axes = axes.reshape(1, -1)
        for row_i, (_, sample_row) in enumerate(samples.iterrows()):
            text   = str(sample_row[self.absa.text_col])
            tokens = text.lower().split()[:15]
            res    = self.absa.analyze_review(text)
            for col_j, aspect in enumerate(aspects_to_show):
                ax     = axes[row_i, col_j]
                attn   = res[aspect]["attention"]
                weights = np.array(attn[1: len(tokens) + 1]).reshape(1, -1)
                im = ax.imshow(weights, aspect="auto", cmap="Blues",
                               vmin=0, vmax=max(weights.max(), 0.01))
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
                ax.set_yticks([])
                lbl = res[aspect]["label"]
                ax.set_title(
                    f"{aspect} | {lbl} | sent={res[aspect]['sentiment']:.2f}",
                    fontsize=9, fontweight="bold",
                    color="green" if lbl == "positive" else "red",
                )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("Attention Heatmaps per Aspect (перші 15 токенів)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        p = self.out / "attention_heatmaps.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_domain_stats(self):
        df     = self.absa.df
        counts = {a: 0 for a in self.absa.aspects}
        scorer = self.absa.domain
        for text in df[self.absa.text_col].astype(str):
            det = scorer.detected_aspects(text)
            for asp in self.absa.aspects:
                if det[asp]:
                    counts[asp] += 1
        total = len(df)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(counts.keys(),
                      [v / total * 100 for v in counts.values()],
                      color="#5CB85C", edgecolor="white")
        for bar, (asp, cnt) in zip(bars, counts.items()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{cnt} ({cnt/total:.1%})", ha="center", fontsize=9)
        ax.set_ylabel("% відгуків з domain-термінами")
        ax.set_title("Domain Knowledge Coverage per Aspect", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = self.out / "domain_stats.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def plot_aspect_distribution(self, aspect_df: pd.DataFrame):
        records = []
        for asp in self.absa.aspects:
            col = f"{asp}_sentiment"
            if col in aspect_df.columns:
                for v in aspect_df[col]:
                    records.append({"Аспект": asp, "Sentiment Score": v})
        if not records:
            return
        df_m = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=df_m, x="Аспект", y="Sentiment Score",
                       palette="Set2", inner="quartile", ax=ax)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Поріг 0.5")
        ax.set_title("Розподіл Sentiment Score по Аспектах", fontsize=13, fontweight="bold")
        ax.legend(); ax.set_ylim(0, 1)
        plt.tight_layout()
        p = self.out / "aspect_distribution.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        self.image_paths.append(str(p))
        print(f"Збережено: {p}")

    def _review_cards_html(self, demo_reviews: list[str]) -> str:
        cards_html = ""
        for text in demo_reviews:
            res = self.absa.analyze_review(text)
            rows_html = ""
            for asp, vals in res.items():
                lbl_color = "#27ae60" if vals["label"] == "positive" else "#e74c3c"
                top_tok_str = " | ".join(
                    f"<b>{t}</b>({w:.3f})" for t, w in vals["top_tokens"]
                ) or "—"
                dom_str = (
                    ", ".join(f'<span class="dom-term">{d}</span>' for d in vals["domain_terms"])
                    or "—"
                )
                sim_bar = int(vals["similarity"] * 100)
                rows_html += f"""
                <tr>
                  <td><b>{asp}</b></td>
                  <td style="color:{lbl_color};font-weight:bold">
                    {vals['label']} ({vals['sentiment']:.3f})
                  </td>
                  <td>
                    <div class="sim-bar-bg">
                      <div class="sim-bar-fill" style="width:{sim_bar}%"></div>
                    </div>
                    {vals['similarity']:.3f}
                  </td>
                  <td class="top-tok">{top_tok_str}</td>
                  <td>{dom_str}</td>
                </tr>"""
            cards_html += f"""
            <div class="review-card">
              <p class="review-text">"{text}"</p>
              <table class="aspect-table">
                <tr>
                  <th>Аспект</th><th>Сентимент (score)</th><th>Схожість</th>
                  <th>Топ-токени (увага)</th><th>Domain терміни</th>
                </tr>
                {rows_html}
              </table>
            </div>"""
        return cards_html

    def build_html_report(
        self,
        demo_reviews:  list[str]       | None = None,
        aspect_df:     pd.DataFrame    | None = None,
        baseline_f1:   float           | None = None,
        per_aspect:    dict[str, dict] | None = None,
        domain_counts: dict[str, int]  | None = None,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        f1  = self.absa.best_f1 or 0.0

        fold_rows = ""
        if self.absa._train_history:
            for h in self.absa._train_history:
                cr = classification_report(
                    h["all_labels"], h["all_preds"],
                    target_names=["negative", "positive"], output_dict=True,
                )
                fold_rows += (
                    f"<tr><td>Fold {h['fold']}</td>"
                    f"<td>{h['fold_f1']:.4f}</td>"
                    f"<td>{cr['negative']['f1-score']:.4f}</td>"
                    f"<td>{cr['positive']['f1-score']:.4f}</td>"
                    f"<td>{h['val_accs'][-1]:.2f}%</td></tr>"
                )

        compare_html = ""
        if baseline_f1 is not None:
            diff  = f1 - baseline_f1
            color = "green" if diff > 0 else "red"
            compare_html = f"""
            <h2>Порівняння з baseline</h2>
            <table>
              <tr><th>Модель</th><th>Weighted F1</th><th>Δ</th></tr>
              <tr><td>Baseline</td><td>{baseline_f1:.4f}</td><td>—</td></tr>
              <tr>
                <td><b>Bi-LSTM + Per-Aspect Attention + Domain Knowledge</b></td>
                <td><b>{f1:.4f}</b></td>
                <td style="color:{color};font-weight:bold">
                  {'+' if diff>=0 else ''}{diff:.4f}
                </td>
              </tr>
            </table>"""

        per_aspect_html = ""
        if per_aspect:
            rows = "".join(
                f"<tr><td><b>{asp}</b></td>"
                f"<td>{v['f1']:.4f}</td>"
                f"<td>{v['accuracy']:.4f}</td>"
                f"<td>{v['positive_rate']:.2%}</td>"
                f"<td>{v.get('n_eval','—')}</td></tr>"
                for asp, v in per_aspect.items()
            )
            per_aspect_html = f"""
            <h2>Per-Aspect Метрики</h2>
            <p>Нейтральні відгуки виключені з оцінки (label_aspect = neutral).
            Кожен аспект оцінюється незалежно на своїх мітках.</p>
            <table>
              <tr><th>Аспект</th><th>F1 (weighted)</th><th>Accuracy</th>
                  <th>Positive Rate</th><th>N (non-neutral)</th></tr>
              {rows}
            </table>"""

        domain_html = ""
        if domain_counts:
            total = len(self.absa.df)
            rows = "".join(
                f"<tr><td>{asp}</td><td>{cnt}</td><td>{cnt/total:.1%}</td></tr>"
                for asp, cnt in domain_counts.items()
            )
            domain_html = f"""
            <h2>Domain Knowledge — Coverage</h2>
            <table>
              <tr><th>Аспект</th><th>Відгуків з термінами</th><th>%</th></tr>
              {rows}
            </table>"""

        review_cards_html = ""
        if demo_reviews:
            cards = self._review_cards_html(demo_reviews)
            review_cards_html = f"""
            <h2>Аналіз відгуків по аспектах</h2>
            {cards}"""

        onto_rows = ""
        for asp, data in DOMAIN_ONTOLOGY.items():
            subs = ", ".join(data["sub_aspects"].keys())
            total_terms = sum(
                len(terms)
                for s in data["sub_aspects"].values()
                for terms in s["terms"].values()
            )
            onto_rows += (
                f"<tr><td><b>{asp}</b></td><td>{data['description']}</td>"
                f"<td>{subs}</td><td>{total_terms}</td></tr>"
            )

        imgs_html = "".join(
            f'<div class="img-block"><img src="{Path(p).name}" alt="{Path(p).stem}">'
            f'<p>{Path(p).stem.replace("_"," ").title()}</p></div>'
            for p in self.image_paths
        )

        html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>Звіт 4.3 — Per-Aspect ABSA (v3)</title>
  <style>
    body {{ font-family:"Segoe UI",Arial,sans-serif; max-width:1200px;
            margin:0 auto; padding:24px; color:#222; background:#f7f9fc; }}
    h1   {{ color:#1a3a6b; border-bottom:3px solid #1a3a6b; padding-bottom:8px; }}
    h2   {{ color:#2c5f9e; margin-top:32px; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }}
    th, td {{ border:1px solid #ccc; padding:8px 12px; text-align:left; }}
    th   {{ background:#2c5f9e; color:#fff; }}
    tr:nth-child(even) {{ background:#eef2f8; }}
    .review-card {{
      background:#fff; border:1px solid #d0d8e8; border-radius:8px;
      padding:16px; margin:16px 0; box-shadow:0 2px 6px rgba(0,0,0,.06);
    }}
    .review-text {{
      font-style:italic; font-size:15px; color:#333;
      border-left:4px solid #2c5f9e; padding-left:12px; margin-bottom:12px;
    }}
    .aspect-table th {{ font-size:13px; padding:6px 10px; background:#7f8c8d; }}
    .aspect-table td {{ font-size:13px; padding:6px 10px; }}
    .top-tok {{ font-family:monospace; font-size:12px; color:#444; }}
    .dom-term {{ background:#dceeff; border-radius:4px; padding:1px 5px; font-weight:bold; }}
    .sim-bar-bg   {{ background:#e0e0e0; border-radius:4px;
                     height:6px; width:80px; display:inline-block; vertical-align:middle; }}
    .sim-bar-fill {{ background:#4C9BE8; border-radius:4px; height:6px; }}
    .img-gallery {{ display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; }}
    .img-block   {{ text-align:center; }}
    .img-block img {{ max-width:560px; border:1px solid #ccc; border-radius:6px; }}
    .img-block p {{ font-size:12px; color:#666; margin:4px 0; }}
    footer {{ margin-top:48px; font-size:12px; color:#888;
              border-top:1px solid #ccc; padding-top:10px; }}
  </style>
</head>
<body>
<h1>Звіт 4.3 — Per-Aspect ABSA: Bi-LSTM + Per-Aspect Attention (v3)</h1>
<p>Дата генерації: {now}</p>

<h2>Загальні результати</h2>
<p><b>Weighted F1 (5-fold CV, primary={self.absa.primary}):</b> {f1:.4f}</p>
<table>
  <tr><th>Fold</th><th>F1 (weighted)</th><th>F1 Negative</th>
      <th>F1 Positive</th><th>Val Accuracy</th></tr>
  {fold_rows}
</table>

{compare_html}
{per_aspect_html}
{domain_html}
{review_cards_html}

<h2>Архітектура моделі (v3)</h2>
<ul>
  <li><b>Embedding</b>: предтреновані FastText (100d).</li>
  <li><b>Bi-LSTM</b> (shared): один прохід на батч, спільний для всіх аспектів.</li>
  <li><b>Per-Aspect Attention</b>: окремі W_h, W_a, v для кожного аспекту →
      content_quality "дивиться" на текст інакше ніж difficulty.</li>
  <li><b>Aspect Vector</b>: середнє FastText-векторів {cfg.MAX_ASPECT_TERMS} domain-термінів
      (не одне слово) → аспектні вектори справді різні.</li>
  <li><b>Per-Aspect Gate + FC Head</b>: окремі параметри класифікатора → різний поріг рішення.</li>
  <li><b>Multi-Task Loss</b>: на кожному батчі loss по ВСІХ трьох аспектах одночасно.</li>
  <li><b>Domain-Weighted Loss</b>: відгуки з domain-термінами аспекту мають вагу
      {cfg.DOMAIN_LOSS_WEIGHT} у loss.</li>
</ul>

<h2>Онтологія (DOMAIN_ONTOLOGY)</h2>
<table>
  <tr><th>Аспект</th><th>Опис</th><th>Підаспекти</th><th>Термінів</th></tr>
  {onto_rows}
</table>

<h2>Графіки</h2>
<div class="img-gallery">
  {imgs_html}
</div>

<footer>Згенеровано автоматично | task_4_3_absa_attention.py v3 | {now}</footer>
</body>
</html>"""

        p = self.out / "report_4_3.html"
        p.write_text(html, encoding="utf-8")
        print(f"HTML-звіт збережено: {p}")
        return str(p)

    def generate_all(
        self,
        demo_reviews:        list[str]       | None = None,
        aspect_df:           pd.DataFrame    | None = None,
        baseline_f1:         float           | None = None,
        per_aspect:          dict[str, dict] | None = None,
        n_attention_samples: int = 4,
    ) -> str:
        print("\n Генерую звіт...")
        self.plot_training_curves()
        self.plot_confusion_matrix()
        self.plot_fold_f1()
        self.plot_attention_heatmaps(n_samples=n_attention_samples)
        self.plot_domain_stats()
        if per_aspect:
            self.plot_per_aspect_f1(per_aspect)
        domain_counts = {a: 0 for a in self.absa.aspects}
        for text in self.absa.df[self.absa.text_col].astype(str):
            det = self.absa.domain.detected_aspects(text)
            for asp in self.absa.aspects:
                if det[asp]:
                    domain_counts[asp] += 1
        if aspect_df is not None:
            self.plot_aspect_distribution(aspect_df)
        return self.build_html_report(
            demo_reviews=demo_reviews,
            aspect_df=aspect_df,
            baseline_f1=baseline_f1,
            per_aspect=per_aspect,
            domain_counts=domain_counts,
        )

def main():
    print(f"Завантаження: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Розмір: {df.shape}  Колонки: {df.columns.tolist()}")

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["Label"]      = df["Label"].astype(str).str.lower().str.strip()
    df["y"]          = (df["Label"] == "positive").astype(int)

    print(f"Позитивних: {df['y'].sum():,}  |  Негативних: {(1-df['y']).sum():,}")

    if "token" not in df.columns:
        df["token"] = df["clean_text"].apply(str.split)

    ft_path = MODELS_DIR / "absa_attention_ft.model"
    ft      = _build_fasttext(df, ft_path)

    ASPECTS = list(DOMAIN_ONTOLOGY.keys())

    absa = ABSAModelWithAttention(
        df=df,
        fasttext_model=ft,
        aspects=ASPECTS,
        text_col="clean_text",
        primary_aspect="content_quality",
    )

    print("\n========== НАВЧАННЯ: Bi-LSTM + Per-Aspect Attention (multi-task) ==========")
    attention_f1 = absa.fit()
    absa.save(MODELS_DIR)

    print("\n========== PER-ASPECT МЕТРИКИ ==========")
    per_aspect = absa.eval_per_aspect(sample_size=2000)

    demo_reviews = [
        "The course material is very comprehensive and well-structured with great examples.",
        "The instructor explains concepts in a very confusing and monotone way.",
        "The pace is overwhelming, assignments pile up and there is no time to understand.",
        "Excellent depth of material but way too advanced for beginners, very steep curve.",
        "Very clear explanations with real-world examples, easy to follow along.",
        "Workload is reasonable and the difficulty level is just right for intermediate learners.",
        "Чудовий матеріал, але викладач пояснює нечітко і занадто швидко.",
        "Завдання надто складні і незрозумілі, темп курсу надмірний для початківців.",
    ]

    print("\n========== ІНФЕРЕНС: демо-відгуки ==========")
    for text in demo_reviews:
        print(f'\n"{text}"')
        res = absa.analyze_review(text)
        for asp, vals in res.items():
            top = " | ".join(f"{t}({w:.3f})" for t, w in vals["top_tokens"][:3])
            det = f" [domain: {vals['domain_terms']}]" if vals["domain_terms"] else ""
            print(
                f"  {asp:<22} | {vals['label']:<8} | "
                f"sent={vals['sentiment']:.3f} | "
                f"sim={vals['similarity']:.3f} | "
                f"top=[{top}]{det}"
            )

    print("\n========== АСПЕКТНИЙ АНАЛІЗ (вибірка 500) ==========")
    sample_df = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
    absa_sample = ABSAModelWithAttention(
        df=sample_df, fasttext_model=ft, aspects=ASPECTS, text_col="clean_text",
    )
    absa_sample.model       = absa.model
    absa_sample.dataset     = absa.dataset
    absa_sample.aspect_vecs = absa.aspect_vecs
    aspect_df = absa_sample.analyze_dataset_aspects()
    aspect_df.to_csv(RESULTS_DIR / "aspect_scores.csv", index=False)
    print(f"Аспектні оцінки збережено: {RESULTS_DIR / 'aspect_scores.csv'}")

    reporter  = ReportGenerator(absa)
    html_path = reporter.generate_all(
        demo_reviews=demo_reviews,
        aspect_df=aspect_df,
        baseline_f1=None,
        per_aspect=per_aspect,
        n_attention_samples=4,
    )

    print(f"\n{'═'*55}")
    print(f"  Per-Aspect ABSA F1 : {attention_f1:.4f}")
    print(f"  HTML-звіт          : {html_path}")
    print(f"  Результати         : {RESULTS_DIR}")
    print(f"  Модель             : {MODELS_DIR}")
    print(f"{'═'*55}")

if __name__ == "__main__":
    main()
