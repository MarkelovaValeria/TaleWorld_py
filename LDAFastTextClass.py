import gensim
from gensim import corpora
from gensim.models import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import FastText
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import FastText
import ast
from sklearn.cluster import KMeans


class LDAFastTextModel:
    def __init__(
        self,
        df,
        num_topics=3,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        aspects=None
    ):
        self.df = df
        self.num_topics = num_topics
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.aspects = aspects or ["food", "service", "place"]

        self.lda_model = None
        self.fasttext_model = None
        self.corpus = None
        self.id2word = None
        self.df_sim = None
    
    def build_lda_model(self, filter_extremes=True, no_below=5, no_above=0.5):
        self.id2word = corpora.Dictionary(self.df["token"])

        if filter_extremes:
            self.id2word.filter_extremes(no_below=no_below, no_above=no_above)

        self.corpus = [self.id2word.doc2bow(text) for text in self.df["token"]]

        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
            iterations=400
        )

        return self.lda_model
    
    def show_lda(self):
        for idx, topic in self.lda_model.print_topics(-1):
            print(f"Тема {idx+1}: {topic}\n")

        vis = gensimvis.prepare(self.lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis, "lda_visualization.html")
        print("LDA-візуалізацію збережено у файл: lda_visualization.html")
    

    def build_fasttext_model(self):
        data_words = self.df["token"].values.tolist()

        self.fasttext_model = FastText(
            sentences=data_words,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )

        return self.fasttext_model
    
    def get_similarity(self, words, aspect):
        sim = 0
        count = 0

        for w in words:
            try:
                sim += self.fasttext_model.wv.similarity(w, aspect)
                count += 1
            except KeyError:
                pass

        return sim / count if count > 0 else 0
    
    def get_lda_topics_words(self, topn=10):
        topics = []
        for i in range(self.num_topics):
            words = [w for w, _ in self.lda_model.show_topic(i, topn=topn)]
            topics.append(words)
        return topics
    
    def topic_aspect_similarity(self):
        lda_topics = self.get_lda_topics_words()
        similarities = []

        for topic_words in lda_topics:
            row = []

            for asp in self.aspects:
                valid_words = [w for w in topic_words if w in self.fasttext_model.wv]

                if len(valid_words) == 0 or asp not in self.fasttext_model.wv:
                    row.append(0)
                    continue

                sim = self.fasttext_model.wv.n_similarity(valid_words, [asp])
                row.append(sim)

            similarities.append(row)

        self.df_sim = pd.DataFrame(similarities, columns=self.aspects)
        self.df_sim.index = [f"Topic {i}" for i in range(len(lda_topics))]
        return self.df_sim
    
    def apply_fasttext_aspects(self):
        tqdm.pandas()

        for aspect in self.aspects:
            self.df[aspect] = self.df["token"].progress_map(
                lambda text: self.get_similarity(text, aspect)
            )

        return self.df
    
    def plot_heatmap(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df_sim, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title("LDA Topic ↔ FastText Aspect Similarity")
        plt.show()

    def run(self):
        print("------------------------- 1. Навчання LDA -------------------------")
        self.build_lda_model()
        self.show_lda()

        print("------------------------- 2. Навчання FastText -------------------------")
        self.build_fasttext_model()

        print("------------------------- 3. Розрахунок схожості тем та аспектів -------------------------")
        self.topic_aspect_similarity()
        self.plot_heatmap()

        print("------------------------- 4. Додавання FastText ознак у DataFrame -------------------------")
        self.apply_fasttext_aspects()    

        return self.lda_model, self.fasttext_model, self.df_sim, self.df
