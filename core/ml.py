import nltk
import string

import pandas as pd
import razdel
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from core.files import save_model, load_model
import logging

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


def get_confidence(config: dict, texts: list[str], user_id: [int, str], channel: str, language='russian') -> list[float]:
    feature_extractor = KeyWords(language)

    model, tfidf = load_model(user_id, channel, config)

    prediction = model.predict_proba(tfidf.transform(feature_extractor.preprocessing(texts)).toarray()).tolist()

    return [i[1] for i in prediction]


async def fit(config: dict, texts: list[str], labels: list[int], user_id: [int, str], channel: str,
              texts_tf_idf: list[str], language='russian') -> None:
    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(texts_tf_idf)

    model = SVC(probability=True)
    model.fit(tfidf.transform(feature_extractor.preprocessing(texts)).toarray(), labels)

    save_model(user_id, channel, model, tfidf, config)


def predict(config: dict, texts: list[str], user_id: [int, str], channel: str, language='russian') -> list:
    """Вызывает модель для выбора лучших постов из texts"""

    feature_extractor = KeyWords(language)

    model, tfidf = load_model(user_id, channel, config)

    return model.predict(tfidf.transform(feature_extractor.preprocessing(texts)).toarray()).tolist()


def finetune(config: dict, texts: list[str], labels: list[int], texts_tf_idf: list[str], user_id: [int, str],
             channel: str, language='russian') -> None:
    """Дообучает модель на новых данных"""

    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(texts_tf_idf)
    X = tfidf.transform(feature_extractor.preprocessing(texts)).toarray()

    if config['model'] != 'CatBoost':

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)
        model_1 = SVC(probability=True)
        model_1.fit(X_train, y_train)

        model_2 = CatBoostClassifier()
        model_2.fit(X_train, y_train)
        svm_f1 = f1_score(y_test, model_1.predict(X_test))
        cb_f1 = f1_score(y_test, model_1.predict(X_test))

        logging.info(
            f'Dataset size: {len(texts)}\n\tCatboost f1: {cb_f1}\n\tSVM f1: {svm_f1} for user {user_id}:{channel}')

        if cb_f1 > svm_f1:
            logging.info(f"Set model CatBoost for user {user_id}:{channel}")
            config['model'] = 'CatBoost'
            model = CatBoostClassifier()
            model.fit(X, labels)
            save_model(user_id, channel, model, tfidf, config)
        else:
            model = SVC(probability=True)
            model.fit(X, labels)
            save_model(user_id, channel, model, tfidf, config)

    else:
        model = CatBoostClassifier()
        model.fit(X, labels)
        save_model(user_id, channel, model, tfidf, config)
