#!pip install langdetect
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import re
import unicodedata as uni
import demoji
from spellchecker import SpellChecker
import spacy
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import string
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import FastText
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import Dataset
import pickle
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import ssl

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)
spell = SpellChecker()
cache = {}
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.punct = set(string.punctuation)
        self.spell = SpellChecker()
        self.cache = {}
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
    
    # видалення пyстих значень
    def remove_null(self, df):
        df = df.dropna(subset=["Review"])
        df = df[df["Review"].apply(lambda x: isinstance(x, str))]
        df.info()
        return df
    
    # видалення URL адрес
    def remove_urls(self, text):
        return re.sub(r'http\S+|www\.\S+', '', text)
    
    # визначення мови та виправлення змішуваного коду та транслітерацій
    def lg_detextion(self, text):
        return detect(text)
    
    # нормалізація унікального тексту (зміна шрифту на стандартний та перетворення емодзі у текст)
    def normalize_and_convert(self, text):
        new = uni.normalize("NFKC", text)
        emojis = demoji.findall(new)
        for e, desc in emojis.items():
            new = new.replace(e, " " + desc)
        return new
    
    # виправлення орфографічних помилок
    def correct_text(self, text):
        words = text.split()
        corrected = []
        for w in words:
            if w in self.cache:
                corrected.append(self.cache[w])
                continue

            if w in self.spell:
                self.cache[w] = w
            else:
                fixed = self.spell.correction(w)
                self.cache[w] = fixed if fixed else w

            corrected.append(self.cache[w])
        return " ".join(corrected)
    
    #
    def preprocess_and_lemmatize_text(self, text):

        def get_wordnet_pos(tag):
            return (
                wordnet.ADJ if tag.startswith("J") else
                wordnet.VERB if tag.startswith("V") else
                wordnet.NOUN if tag.startswith("N") else
                wordnet.ADV if tag.startswith("R") else 
                wordnet.NOUN
            )

        sentences = sent_tokenize(text.lower())
        tokens = []
        aspects = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [
                w for w in words 
                if w not in self.stop_words and w not in self.punct and not w.isdigit()
            ]
            pos = pos_tag(words)
            lemmas = [self.lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos]

            tokens.extend(lemmas)  # ← для LDA

            for i in range(len(lemmas) - 1):
                if pos[i][1].startswith("JJ") and pos[i+1][1].startswith("NN"):
                    aspects.append(f"{lemmas[i]} {lemmas[i+1]}")
                if pos[i][1].startswith("NN") and pos[i+1][1].startswith("JJ"):
                    aspects.append(f"{lemmas[i]} {lemmas[i+1]}")

        return tokens, aspects
    
    # графік пyстих значень
    def plt_null(self, df):
        plt.figure(figsize=(25, 20))
        msno.matrix(df, color=[0.2, 0.4, 1])
        plt.show()
    
    # запyснк
    def clean_dataframe(self, df):

        print("------------------------- 1. перевірка даних на пyсті значення -------------------------")
        if df.isnull().values.any():
            self.plt_null(df)
            df = self.remove_null(df)
            self.plt_null(df)

        df["Review"] = df["Review"].astype(str)

        print("------------------------- 2. видалення URL адрес -------------------------")
        df["Review"] = df["Review"].apply(self.remove_urls)
        print(df["Review"])

        print("------------------------- 3. нормалізація унікального тексту (зміна шрифту на стандартний та перетворення емодзі у текст) -------------------------")
        df["Review_cleaned"] = df["Review"].apply(self.normalize_and_convert)
        
        print(f"старий текст - {df['Review']}")
        print(f"новий текст - {df['Review_cleaned']}")

        print("------------------------- 4. виправлення орфографічних помилок -------------------------")
        df["Review_corrected"] = df["Review_cleaned"].apply(self.correct_text)

        print(f"старий текст - {df['Review_cleaned']}")
        print(f"новий текст - {df['Review_corrected']}")

        print("------------------------- 5. визначення мови та виправлення змішуваного коду та транслітерацій -------------------------")
        df["language"] = df["Review_corrected"].apply(self.lg_detextion)

        print(f"{df['Review_corrected']} - {df['language']}")

        print("------------------------- 6. токенізація слів, видалення стоп слів, лематизація -------------------------")
        df["tokens"], df["aspects"] = zip(
            *df["Review_corrected"].apply(self.preprocess_and_lemmatize_text)
        )

        print(f"старий текст - {df['Review_corrected']}")
        print(f"новий текст - {df['tokens']}")
        print(f"новий текст - {df['aspects']}")

        return df
