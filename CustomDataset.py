
import torch
from torch.utils.data import Dataset
from Vocabulary import Vocabulary

class CustomDataset(Dataset):
    '''
    Ініціальні змінні
    df: навчальний фрейм даних
    source_column: назва стовпця вихідного тексту у фреймі даних
    transform: якщо ми хочемо додати будь-яке доповнення freq_threshold: мінімальна кількість разів, коли слово має зустрічатися в корпусі, щоб його було оброблено у словнику
    source_vocab_max_size: максимальний розмір вихідного словника
    '''

    def __init__(self, df, source_column, freq_threshold=3,
                 source_vocab_max_size=10000, transform=None):

        # Скидаємо індекси, щоб точно працювало з DataLoader
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # отримати вихідні тексти
        self.source_texts = self.df[source_column]

        # Ініціалізувати об'єкт source vocab та створити словник
        self.source_vocab = Vocabulary(freq_threshold, source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Використовуємо iloc для доступу по позиції
        source_text = self.source_texts.iloc[index]
        label = self.df.y.iloc[index]

        if self.transform is not None:
            source_text = self.transform(source_text)

        # нумерація текстів ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])

        # перетворити список на тензор та повернути
        return torch.tensor(numerialized_source), torch.tensor(label, dtype=torch.float)