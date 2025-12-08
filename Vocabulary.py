class Vocabulary:

    '''
    Метод __init__ - для ініціалізації словників
    '''
    def __init__(self, freq_threshold, max_size):
        '''
        freq_threshold: мінімальна кількість разів, коли слово має зустрічатися в корпусі, щоб його можна було обробляти у словнику
        max_size: максимальний розмір вихідного словника. Наприклад, якщо встановити значення 10 000, ми вибираємо 10 000 найчастіших слів та відкидаємо інші.
        '''
        #initiate the index to token dict
        ## <PAD> -> доповнення, що використовується для доповнення коротших речень у пакеті, щоб вони відповідали довжині найдовшого речення в пакеті
        ## <SOS> -> токен початку, що додається перед кожним реченням для позначення початку речення
        ## <EOS> -> Лексема кінця речення, що додається в кінець кожного речення для позначення його кінця.
        ## <UNK> -> Слова, яких немає у словнику, замінюються цією лексемою
        self.itos = {0: '<PAD>', 1:'<SOS>', 2:'<EOS>', 3: '<UNK>'}
        #ініціювати токен для індексації словника
        self.stoi = {k:j for j,k in self.itos.items()}

        self.freq_threshold = freq_threshold
        self.max_size = max_size

    '''
    __len__ використовується завантажувачем даних пізніше для створення пакетів
    '''
    def __len__(self):
        return len(self.itos)

    '''
    простий токенізатор для розбиття речення на простір та перетворення його на список слів
    '''
    @staticmethod
    def tokenizer(text):
        return [tok.lower().strip() for tok in text.split(' ')]

    '''
    Збірка словника: створення словникового відображення індексу в рядок (itos) та рядка в індекс (stoi)
    наприклад, для stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocabulary(self, sentence_list):
        #спочатку обчисліть частоти кожного слова, щоб видалити слова з freq < freq_threshold
        frequencies = {}  #ініціалізація частотного словника
        idx = 4 #індекс, з якого ми хочемо, щоб починався наш словник. Ми вже використовували 4 індекси для pad, start, end, unk

        #обчислення частоти слів
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1


        #обмежити словниковий запас, видаливши слова з низькою частотою вживання
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold}

        #обмежити словниковий запас до вказаного max_size
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk

        #створення словника
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1


    '''
    перетворити список слів на список відповідних індексів
    '''
    def numericalize(self, text):
        #токенізувати текст
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else: #слова, що не входять до словникового запасу, представлені індексом токенів UNK
                numericalized_text.append(self.stoi['<UNK>'])

        return numericalized_text