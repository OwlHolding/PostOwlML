import nltk
import string
import razdel
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import SVC
from multiprocessing import RLock

from core.files import save_model, load_model

lock = RLock()

warnings.filterwarnings("ignore")


class KeyWords:
    def __init__(self, language='russian'):
        self.punctuation = string.punctuation + '»' + '«'
        self.lang = language
        try:
            self.stop_words = stopwords.words(language)
        except:
            nltk.download('stopwords')
            self.stop_words = stopwords.words(language)
        if language == 'russian':
            self.stem = Mystem()
        else:
            self.stem = SnowballStemmer(language='english')
            self.stop_words += ['i']

    @staticmethod
    def remove_urls(documents: list) -> list:
        return [re.sub('https?:\/\/.*?[\s+]', '', text) for text in documents]

    @staticmethod
    def replace_newline(documents: list) -> list:
        documents = [text.replace('\n', ' ') + ' ' for text in documents]
        return documents

    @staticmethod
    def remove_strange_symbols(documents: list) -> list:
        return [re.sub(f'[^A-Za-zА-Яа-яё0-9{string.punctuation}\ ]+', ' ', text) for text in documents]

    def tokenize(self, documents: list) -> list:
        if self.lang == 'english':
            return [nltk.word_tokenize(text) for text in documents]
        else:
            return [[token.text for token in razdel.tokenize(text)] for text in documents]

    @staticmethod
    def to_lower(documents: list) -> list:
        return [text.lower() for text in documents]

    @staticmethod
    def remove_punctuation(tokenized_documents) -> list:
        ttt = set(string.punctuation)
        return [[token for token in tokenized_text if not set(token) < ttt] for tokenized_text in tokenized_documents]

    @staticmethod
    def remove_numbers(documents: list) -> list:
        return [re.sub('(?!:\s)\d\.?\d*', ' ', text) for text in documents]

    def remove_stop_words(self, tokenized_documents) -> list:
        return [[token for token in tokenized_text if token not in self.stop_words] for tokenized_text in
                tokenized_documents]

    def lemmatize(self, documents: list) -> list:
        if self.lang == 'russian':
            return [''.join(self.stem.lemmatize(text)) for text in documents]
        else:
            return [' '.join(self.stem.stem(token) for token in text.split()) for text in documents]

    def preprocessing(self, documents: list) -> list:
        documents = self.replace_newline(documents)
        documents = self.remove_urls(documents)
        documents = self.remove_strange_symbols(documents)
        documents = self.to_lower(documents)
        documents = self.lemmatize(documents)
        documents = self.remove_numbers(documents)
        tokenized_documents = self.tokenize(documents)
        tokenized_documents = self.remove_stop_words(tokenized_documents)
        tokenized_documents = self.remove_punctuation(tokenized_documents)
        documents = [' '.join(tokenized_text) for tokenized_text in tokenized_documents]
        return documents

    def get_tfifd(self, documents: list) -> TfidfVectorizer:
        clean_documents = self.preprocessing(documents)

        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer.fit(clean_documents)

        return tf_idf_vectorizer


def fit(texts: list[str], labels: list[int], path: str, language='russian') -> None:

    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(texts)
    model = SVC(probability=True)
    model.fit(tfidf.transform(feature_extractor.preprocessing(texts)).toarray(), labels)
    save_model(path, model, tfidf)


def predict(texts: list[str], path: str, language='russian') -> list:
    feature_extractor = KeyWords(language)

    model, tfidf = load_model(path)

    return model.predict_proba(tfidf.transform(feature_extractor.preprocessing(texts)).toarray()).tolist()
