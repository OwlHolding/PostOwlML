from sklearnex import patch_sklearn

patch_sklearn()

import logging
import re
import string
import warnings

import nltk
import razdel
from catboost import CatBoostClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


warnings.filterwarnings("ignore")


class KeyWords:
    def __init__(self, language='russian'):
        self.punctuation = string.punctuation + '»' + '«'
        self.lang = language
        try:
            self.stop_words = stopwords.words(language)
        except Exception as e:
            logging.info(f'{e}\nStopwords install')
            nltk.download('stopwords')
            self.stop_words = stopwords.words(language)
        if language == 'russian':
            self.stem = Mystem()
        else:
            self.stem = SnowballStemmer(language='english')
            self.stop_words += ['i']

    @staticmethod
    def remove_urls(documents: list) -> list:
        return [re.sub(r'https?:\/\/.*?[\s+]', '', text) for text in documents]

    @staticmethod
    def remove_tags(documents: list) -> list:
        return [re.sub('<[^<]+?>', '', i) for i in documents]

    @staticmethod
    def replace_newline(documents: list) -> list:
        documents = [text.replace('\n', ' ') + ' ' for text in documents]
        return documents

    @staticmethod
    def remove_strange_symbols(documents: list) -> list:
        return [re.sub(fr'[^A-Za-zА-Яа-яё\d{string.punctuation}\ ]+', ' ', text) for text in documents]

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
        return [re.sub(r'(?!:\s)\d\.?\d*', ' ', text) for text in documents]

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
        documents = self.remove_tags(documents)
        tokenized_documents = self.tokenize(documents)
        tokenized_documents = self.remove_stop_words(tokenized_documents)
        tokenized_documents = self.remove_punctuation(tokenized_documents)
        documents = [' '.join(tokenized_text) for tokenized_text in tokenized_documents]
        return documents

    def get_tfifd(self, documents: list) -> TfidfVectorizer:
        clean_documents = self.preprocessing(documents)

        tf_idf_vectorizer = TfidfVectorizer(max_features=5000)
        tf_idf_vectorizer.fit(clean_documents)

        return tf_idf_vectorizer


def get_confidence(texts: list[str], model, tfidf, language='russian') -> list[float]:
    feature_extractor = KeyWords(language)

    prediction = model.predict_proba(tfidf.transform(feature_extractor.preprocessing(texts)).toarray()).tolist()

    return [(abs(i[0] - 0.5) + abs(i[1] - 0.5)) / 2 for i in prediction]


def fit(config: dict, texts: list[str], labels: list[int], texts_tf_idf: list[str], language='russian'):
    """Инициализирует и учит модель"""

    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(texts_tf_idf)

    model = SVC(probability=True)
    model.fit(tfidf.transform(feature_extractor.preprocessing(texts)).toarray(), labels)

    return model, tfidf


def predict(texts: list[str], model, tfidf, language='russian') -> list:
    """Вызывает модель для выбора лучших постов из texts"""

    feature_extractor = KeyWords(language)

    return model.predict(tfidf.transform(feature_extractor.preprocessing(texts)).toarray()).tolist()


def finetune(config: dict, texts: list[str], labels: list[int], texts_tf_idf: list[str], user_id: [int, str],
             channel: str, language='russian'):
    """Дообучает модель на новых данных"""

    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(texts_tf_idf)
    x = tfidf.transform(feature_extractor.preprocessing(texts)).toarray()

    if config['model'] != 'CatBoost':

        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.3, random_state=42)
        model_1 = SVC(probability=True)
        model_1.fit(x_train, y_train)

        model_2 = CatBoostClassifier(verbose=0)
        model_2.fit(x_train, y_train)
        svm_f1 = f1_score(y_test, model_1.predict(x_test))
        cb_f1 = f1_score(y_test, model_1.predict(x_test))

        logging.info(
            f'Dataset size: {len(texts)}\n\tCatboost f1: {cb_f1}\n\tSVM f1: {svm_f1} for user {user_id}:{channel}')

        if cb_f1 > svm_f1:
            logging.info(f"Set model CatBoost for user {user_id}:{channel}")
            config['model'] = 'CatBoost'
            model = CatBoostClassifier(verbose=0)
            model.fit(x, labels)

            return model, tfidf

        else:
            model = SVC(probability=True)
            model.fit(x, labels)

            return model, tfidf

    else:
        model = CatBoostClassifier(verbose=0)
        model.fit(x, labels)

        return model, tfidf