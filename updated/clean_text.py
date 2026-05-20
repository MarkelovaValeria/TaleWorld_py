import nltk
import stanza

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stanza.download("en")
stanza.download("uk")
stanza.download("ru")
stanza.download("zh")

import re
import unicodedata as uni
import string
import random
import emoji
import demoji
import jieba                        
import pandas as pd
import numpy as np

from tqdm import tqdm
from langdetect import detect
from collections import Counter

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def build_stopwords():
    stop_en = set(stopwords.words("english"))
    stop_ru = set(stopwords.words("russian"))

    try:
        stop_uk = set(stopwords.words("ukrainian"))
    except OSError:
        stop_uk = set()

    stop_uk.update({
        "i", "й", "та", "але", "проте", "однак", "або", "чи",
        "що", "як", "коли", "де", "хто", "це", "той",
        "із", "зі", "від", "до", "за", "по", "на", "у", "в",
        "з", "про", "для", "через", "під", "над", "між",
        "не", "ні", "так", "вже", "ще", "теж", "також",
        "він", "вона", "воно", "вони", "ми", "ви", "я",
        "його", "її", "їх", "нас", "вас", "мене", "тебе",
        "цей", "ця", "ці", "той", "ті",
        "бути", "є", "був", "була", "були", "буде", "будуть",
        "мати", "має", "мав", "мала", "мали",
        "який", "яка", "яке", "які", "яким", "якою",
        "дуже", "більш", "менш", "більше", "менше",
    })

    stop_zh = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不",
        "人", "都", "一", "一个", "上", "也", "很", "到", "说",
        "要", "去", "你", "会", "着", "没有", "看", "好", "自己",
        "这", "那", "他", "她", "它", "们", "我们", "你们",
        "他们", "她们", "什么", "怎么", "为什么", "哪", "谁",
        "因为", "所以", "但是", "然后", "如果", "虽然", "还是",
        "已经", "可以", "能", "应该", "这个", "那个", "这些",
        "那些", "一些", "一种", "非常", "比较", "真的", "真是",
        "对", "吧", "啊", "哦", "嗯", "呢", "吗", "呀", "哈",
        "就是", "还有", "而且", "不过", "其实", "只是", "可能",
        "一直", "一样", "之后", "以后", "之前", "以前",
        "从", "到", "在", "被", "把", "让", "给", "跟", "和",
        "与", "或", "虽", "但", "而", "于", "以", "为", "又",
        "再", "更", "最", "很", "太", "真", "还", "都", "只",
    }

    return {
        "en": stop_en,
        "uk": stop_uk,
        "ru": stop_ru,
        "zh": stop_zh,
    }

class AdvancedTextCleaner:
    def __init__(self):

        print("Loading NLP pipelines...")

        self.pipelines = {
            "en": stanza.Pipeline(
                lang="en",
                processors="tokenize,pos,lemma",
                verbose=False
            ),
            "uk": stanza.Pipeline(
                lang="uk",
                processors="tokenize,pos,lemma",
                verbose=False
            ),
            "ru": stanza.Pipeline(
                lang="ru",
                processors="tokenize,pos,lemma",
                verbose=False
            ),
            "zh": stanza.Pipeline(
                lang="zh",
                processors="tokenize,pos,lemma",
                verbose=False
            ),
        }

        self.stop_words = build_stopwords()

        self.punct = set(string.punctuation) | {
            "。", "，", "！", "？", "；", "：", "、", "…"
        }

        self.language_cache = {}

        print("Pipelines loaded successfully.")

    def _get_pipeline(self, lang: str):
        """Returns the correct Stanza pipeline, falls back to English."""
        return self.pipelines.get(lang, self.pipelines["en"])

    def _get_stopwords(self, lang: str) -> set:
        """Returns stopwords for the given language, falls back to English."""
        return self.stop_words.get(lang, self.stop_words["en"])

    def remove_nulls(self, df):
        df = df.dropna(subset=["Review"])
        df = df[df["Review"].apply(lambda x: isinstance(x, str))]
        df = df.reset_index(drop=True)
        return df

    def remove_urls(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        return text

    def remove_html(self, text):
        return re.sub(r"<.*?>", "", text)

    def normalize_text(self, text):
        text = uni.normalize("NFKC", text)
        text = emoji.demojize(text)
        emojis = demoji.findall(text)
        for e, desc in emojis.items():
            text = text.replace(e, f" {desc} ")
        return text

    def clean_special_characters(self, text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def detect_language(self, text):
        """
        Detects language and maps to one of: en, uk, ru, zh.
        Everything else falls back to 'en'.
        """
        try:
            if text in self.language_cache:
                return self.language_cache[text]

            lang = detect(text)

            if lang.startswith("zh"):
                lang = "zh"

            if lang not in self.pipelines:
                lang = "en"

            self.language_cache[text] = lang
            return lang

        except Exception:
            return "en"

    def lemmatize_text(self, text, lang):
        stop_words = self._get_stopwords(lang)
        pipeline   = self._get_pipeline(lang)

        if lang == "zh":
            text = " ".join(jieba.cut(text))

        doc = pipeline(text.lower())

        lemmas = []

        for sentence in doc.sentences:
            for word in sentence.words:

                lemma = (word.lemma or word.text).lower()

                if (
                    lemma not in stop_words
                    and lemma not in self.punct
                    and not lemma.isdigit()
                    and len(lemma) > 1
                ):
                    lemmas.append(lemma)

        return lemmas

    def extract_aspects(self, text, lang):
        pipeline = self._get_pipeline(lang)

        if lang == "zh":
            text = " ".join(jieba.cut(text))

        doc = pipeline(text)

        aspects = []

        for sentence in doc.sentences:
            words = sentence.words

            for i in range(len(words) - 1):
                cur = words[i]
                nxt = words[i + 1]

                if cur.upos == "ADJ" and nxt.upos == "NOUN":
                    aspects.append(f"{cur.lemma} {nxt.lemma}")

                if cur.upos == "NOUN" and nxt.upos == "ADJ":
                    aspects.append(f"{cur.lemma} {nxt.lemma}")

        return list(set(aspects))

    def synonym_replacement(self, text, n=1):
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set(words))
        random.shuffle(random_word_list)
        replaced = 0

        for random_word in random_word_list:
            synonyms = []

            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym != random_word:
                        synonyms.append(synonym)

            synonyms = list(set(synonyms))

            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [
                    synonym if word == random_word else word
                    for word in new_words
                ]
                replaced += 1

            if replaced >= n:
                break

        return " ".join(new_words)

    BACK_TRANSLATION_PIVOT = {
        "en": "fr",                                  
        "uk": "en",                                    
        "ru": "en",                                  
        "zh": "en",                                  
    }

    def back_translation(self, text, source_lang="en"):
        try:
            pivot = self.BACK_TRANSLATION_PIVOT.get(source_lang, "fr")

            translated = GoogleTranslator(
                source=source_lang,
                target=pivot
            ).translate(text)

            back = GoogleTranslator(
                source=pivot,
                target=source_lang
            ).translate(translated)

            return back

        except Exception:
            return text     

    def augment_rare_classes(self, df, min_samples=100):
        print("\nAUGMENTING RARE CLASSES...")

        augmented_rows = []
        class_counts   = df["Label"].value_counts()

        print("\nClass distribution BEFORE:")
        print(class_counts)

        for label, count in class_counts.items():

            if count < min_samples:
                needed = min_samples - count
                subset = df[df["Label"] == label]

                for _ in range(needed):
                    row  = subset.sample(1).iloc[0]
                    text = row["Review"]
                    lang = row["language"]

                    if lang == "en" and random.random() < 0.5:
                        augmented_text = self.synonym_replacement(text)
                    else:
                        augmented_text = self.back_translation(
                            text,
                            source_lang=lang
                        )

                    augmented_rows.append({
                        "CourseId": row["CourseId"],
                        "Review":   augmented_text,
                        "Label":    label,
                        "language": lang,
                    })

        augmented_df = pd.DataFrame(augmented_rows)
        final_df     = pd.concat([df, augmented_df], ignore_index=True)

        print("\nClass distribution AFTER:")
        print(final_df["Label"].value_counts())

        return final_df

    def analyze_data_drift(self, old_texts, new_texts):
        print("\n==============================")
        print("DATA DRIFT ANALYSIS")
        print("==============================")

        vectorizer = TfidfVectorizer()
        combined   = old_texts + new_texts
        X          = vectorizer.fit_transform(combined)

        old_vectors = X[:len(old_texts)]
        new_vectors = X[len(old_texts):]

        similarity = cosine_similarity(
            np.asarray(old_vectors.mean(axis=0)),
            np.asarray(new_vectors.mean(axis=0))
        )[0][0]

        drift_score = 1 - similarity

        print(f"Similarity:  {similarity:.4f}")
        print(f"Drift Score: {drift_score:.4f}")

        if drift_score > 0.3:
            print("WARNING: SIGNIFICANT DATA DRIFT DETECTED")
        else:
            print("No significant drift detected.")

        return drift_score

    def balance_sentiment(
        self,
        df,
        positive_label="positive",
        negative_label="negative",
        label_col="Label",
        random_state=42,
    ):

        print("\nBALANCING SENTIMENT (negative : positive = 1 : 1)...")

        pos_df = df[df[label_col] == positive_label]
        neg_df = df[df[label_col] == negative_label]

        n_pos = len(pos_df)
        n_neg = len(neg_df)

        print(f"  Before — positive: {n_pos}, negative: {n_neg}")

        target_size = min(n_pos, n_neg)

        pos_df = pos_df.sample(n=target_size, random_state=random_state)
        neg_df = neg_df.sample(n=target_size, random_state=random_state)

        balanced_df = pd.concat(
            [pos_df, neg_df],
            ignore_index=True
        ).sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"  After  — positive: {target_size}, negative: {target_size}")
        print(f"  Total rows: {len(balanced_df)}")

        return balanced_df

    def map_labels(self, df, label_col="Label"):
        print("\nMAPPING LABELS: 4-5 → positive, 1-3 → negative")

        df[label_col] = df[label_col].apply(
            lambda x: "positive" if int(x) >= 4 else "negative"
        )

        print(df[label_col].value_counts())

        return df

    def clean_and_detect(self, df):
        print("\n==============================")
        print("STAGE 1: CLEANING + LANGUAGE DETECTION")
        print("Supported: English | Ukrainian | Russian | Chinese")
        print("==============================")

        df = self.remove_nulls(df)
        df["Review"] = df["Review"].astype(str)
        df["Review"] = df["Review"].apply(self.remove_urls)
        df["Review"] = df["Review"].apply(self.remove_html)
        df["Review"] = df["Review"].apply(self.normalize_text)
        df["Review"] = df["Review"].apply(self.clean_special_characters)

        tqdm.pandas(desc="Detecting languages")
        df["language"] = df["Review"].progress_apply(self.detect_language)

        print("\nLanguage distribution in dataset:")
        print(df["language"].value_counts())

        print("\nSTAGE 1 FINISHED")

        return df

    def process_dataframe(self, df):
        print("\n==============================")
        print("STAGE 2: LEMMATIZATION + ASPECT EXTRACTION")
        print("==============================")

        tqdm.pandas(desc="Lemmatizing")
        df["tokens"] = df.progress_apply(
            lambda row: self.lemmatize_text(row["Review"], row["language"]),
            axis=1,
            result_type="reduce"
        )

        tqdm.pandas(desc="Extracting aspects")
        df["aspects"] = df.progress_apply(
            lambda row: self.extract_aspects(row["Review"], row["language"]),
            axis=1
        )

        df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

        print("\nSTAGE 2 FINISHED SUCCESSFULLY")

        return df

if __name__ == "__main__":
    df = pd.read_csv("reviews_by_course.csv")

    cleaner = AdvancedTextCleaner()
    df = cleaner.clean_and_detect(df)
    df = cleaner.map_labels(df)
    df = cleaner.balance_sentiment(
        df,
        positive_label="positive",
        negative_label="negative",
    )
    df = cleaner.process_dataframe(df)
    df = cleaner.augment_rare_classes(df, min_samples=300)
    df[["Review", "clean_text", "Label"]].to_csv("processed_reviews.csv", index=False)
    print("\nSaved: processed_reviews.csv")

    old_reviews = df["clean_text"][:1000].tolist()
    new_reviews = df["clean_text"][1000:1200].tolist()
    cleaner.analyze_data_drift(old_reviews, new_reviews)

    df[["Review", "clean_text", "Label", "aspects"]].to_csv("processed_reviews.csv", index=False)
    print("\nSaved: processed_reviews.csv")