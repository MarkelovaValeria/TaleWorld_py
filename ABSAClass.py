import warnings
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gensim
import ast
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore, FastText
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
from LDAFastTextClass import LDAFastTextModel
from config import config
from CustomDataset import CustomDataset
from MyCollate import MyCollate
from Model import Model


def get_emb_layer_with_weights(target_vocab, emb_model, trainable = False):

    weights_matrix = np.zeros((len(target_vocab), config.EMB_DIM))
    words_found = 0

    for i, word in enumerate(target_vocab):
        weights_matrix[i] = np.concatenate([emb_model.wv[word]])
        words_found += 1

    print(f"Words found are : {words_found}")

    weights_matrix = torch.tensor(weights_matrix, dtype = torch.float32).reshape(len(target_vocab), config.EMB_DIM)
    emb_layer = nn.Embedding.from_pretrained(weights_matrix)
    print(emb_layer)
    if trainable:
        emb_layer.weight.requires_grad = True
    else:
        emb_layer.weight.requires_grad = False

    return emb_layer
  
def train_epochs(dataloader,model, loss_fn, optimizer):
    train_correct = 0
    train_loss = 0

    model.train()

    for review, label in tqdm(dataloader):

        review, label = review.to(config.DEVICE), label.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(review)
        output = output.reshape(-1)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*review.size(1)
        prediction = (output > 0.5).float()
        train_correct += (prediction == label).float().sum()

    return train_loss, train_correct

def val_epochs(dataloader, model, loss_fn):
    val_correct = 0
    val_loss = 0

    model.eval()
    for review, label in dataloader:

        review, label = review.to(config.DEVICE), label.to(config.DEVICE)

        output = model(review)
        output = output.reshape(-1)

        loss = loss_fn(output, label)

        val_loss += loss.item()*review.size(1)
        prediction = (output > 0.5).float()
        val_correct += (prediction == label).float().sum()
    return val_loss, val_correct

def plot_accuracy(train_accs, val_accs, title='Model Accuracy'):
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()

def plot_losses(train_losses, val_losses, title='Model Loss'):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

class ABSAModel:
    def __init__(self, df, fasttext_model, aspects, text_column="Review_corrected"):
        self.df = df.copy()
        self.fasttext_model = fasttext_model
        self.aspects = aspects
        self.text_column = text_column

        self.dataset = None
        self.model = None


    # ПІДГОТОВКА ДАНИХ
    def prepare_labels(self):
        def label(y):
            return 1 if y in ['4', '5'] else 0

        tqdm.pandas()
        self.df['y'] = self.df.Rating.progress_map(label)


    def prepare_dataset(self):
        self.dataset = CustomDataset(self.df, self.text_column)
        print(self.dataset[0])

    # ТРЕНУВАННЯ МОДЕЛІ
    def fit(self):
        self.prepare_labels()
        self.prepare_dataset()

        kfold = KFold(n_splits=config.FOLDS)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(self.dataset)))):
            print(f"\n-------- Fold {fold} --------")

            train_loader = DataLoader(
                self.dataset,
                batch_size=config.BATCH_SIZE,
                sampler=SubsetRandomSampler(train_idx),
                collate_fn=MyCollate(0, config.MAX_LEN)
            )

            val_loader = DataLoader(
                self.dataset,
                batch_size=config.BATCH_SIZE,
                sampler=SubsetRandomSampler(val_idx),
                collate_fn=MyCollate(0, config.MAX_LEN)
            )

            VOCAB = list(self.dataset.source_vocab.stoi)
            VOCAB_SIZE = len(VOCAB)

            embedding_layer = get_emb_layer_with_weights(
                target_vocab=VOCAB,
                emb_model=self.fasttext_model,
                trainable=False
            )

            self.model = Model(
                VOCAB_SIZE,
                config.EMB_DIM,
                hidden_dim=128,
                output_dim=1,
                embedding_layer=embedding_layer
            ).to(config.DEVICE)

            loss_fn = nn.BCELoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            for epoch in range(config.EPOCHS):
                train_loss, train_correct = train_epochs(train_loader, self.model, loss_fn, optimizer)
                val_loss, val_correct = val_epochs(val_loader, self.model, loss_fn)

                train_loss /= len(train_loader.sampler)
                val_loss /= len(val_loader.sampler)
                train_acc = (train_correct / len(train_loader.sampler)) * 100
                val_acc = (val_correct / len(val_loader.sampler)) * 100

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc.cpu().numpy())
                val_accs.append(val_acc.cpu().numpy())

                print(f"| Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% |")

            plot_accuracy(train_accs, val_accs)
            plot_losses(train_losses, val_losses)


    def text_to_tensor(self, text):
        numericalized = [self.dataset.source_vocab.stoi["<SOS>"]]
        numericalized += self.dataset.source_vocab.numericalize(text.lower())
        numericalized.append(self.dataset.source_vocab.stoi["<EOS>"])
        if len(numericalized) < config.MAX_LEN:
            numericalized += [self.dataset.source_vocab.stoi["<PAD>"]] * (config.MAX_LEN - len(numericalized))
        else:
            numericalized = numericalized[:config.MAX_LEN]
        return torch.tensor([numericalized], dtype=torch.long, device=config.DEVICE)
    
    def predict_sentiment(self, text):
        tensor = self.text_to_tensor(text)
        self.model.eval()
        with torch.no_grad():
            return float(torch.sigmoid(self.model(tensor)).cpu().numpy().flatten()[0])
        
    def analyze_review_aspects_score(self, text):
        sentiment = self.predict_sentiment(text)
        tokens = text.lower().split()
        results = {}

        for aspect in self.aspects:
            try:
                sim = self.fasttext_model.wv.n_similarity(tokens, [aspect])
            except KeyError:
                sim = 0.0
            combined = sentiment * sim
            results[aspect] = {
                "sentiment": sentiment,
                "similarity": sim,
                "combined": combined
            }
        return results
    
    def analyze_all_restaurants(self):
        restaurant_scores = {}
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            restaurant = row["Restaurant"]
            text = row[self.text_column]
            aspect_scores = self.analyze_review_aspects_score(text)
            if restaurant not in restaurant_scores:
                restaurant_scores[restaurant] = {a: [] for a in self.aspects}
            for aspect in self.aspects:
                restaurant_scores[restaurant][aspect].append(aspect_scores[aspect]["combined"])
        return restaurant_scores

    def average_restaurant_scores(self, restaurant_scores):
        final_results = {}
        for restaurant, aspect_dict in restaurant_scores.items():
            final_results[restaurant] = {aspect: round(np.mean(scores), 2)
                                         for aspect, scores in aspect_dict.items()}
        return final_results
    
    def print_results(self, final_results):
        for restaurant, aspects in final_results.items():
            print(f"\nРесторан: {restaurant}")
            for aspect, score in aspects.items():
                print(f"{aspect:<12} → {score:.2f}")

    def evaluate_restaurants(self):
        scores = self.analyze_all_restaurants()
        final_results = self.average_restaurant_scores(scores)
        self.print_results(final_results)
        return final_results
    
    def evaluate_custom_review(self, text):
        print("Аналізую коментар...")
        scores = self.analyze_review_aspects_score(text)

        print("\nРезультат:")
        for aspect, vals in scores.items():
            print(f"{aspect}: sentiment={vals['sentiment']:.3f}, similarity={vals['similarity']:.3f}, combined={vals['combined']:.3f}")

        return scores