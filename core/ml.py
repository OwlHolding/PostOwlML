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
import pickle
from multiprocessing import RLock

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

    def remove_urls(self, documents) -> list:
        return [re.sub('https?:\/\/.*?[\s+]', '', text) for text in documents]

    def replace_newline(self, documents):
        documents = [text.replace('\n', ' ') + ' ' for text in documents]
        return documents

    def remove_strange_symbols(self, documents):
        return [re.sub(f'[^A-Za-zА-Яа-яё0-9{string.punctuation}\ ]+', ' ', text) for text in documents]

    def tokenize(self, documents) -> list:
        if self.lang == 'english':
            return [nltk.word_tokenize(text) for text in documents]
        else:
            return [[token.text for token in razdel.tokenize(text)] for text in documents]

    def to_lower(self, documents) -> list:
        return [text.lower() for text in documents]

    def remove_punctuation(self, tokenized_documents):
        ttt = set(string.punctuation)
        return [[token for token in tokenized_text if not set(token) < ttt] for tokenized_text in tokenized_documents]

    def remove_numbers(self, documents):
        return [re.sub('(?!:\s)\d\.?\d*', ' ', text) for text in documents]

    def remove_stop_words(self, tokenized_documents) -> list:
        return [[token for token in tokenized_text if token not in self.stop_words] for tokenized_text in
                tokenized_documents]

    def lemmatize(self, documents) -> list:
        if self.lang == 'russian':
            return [''.join(self.stem.lemmatize(text)) for text in documents]
        else:
            return [' '.join(self.stem.stem(token) for token in text.split()) for text in documents]

    def preprocessing(self, documents):
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

    def get_tfifd(self, documents):
        clean_documents = self.preprocessing(documents)

        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer.fit(clean_documents)

        return tf_idf_vectorizer


def fit(dataset, all_texts, path_model, path_tfidf, language='russian'):
    feature_extractor = KeyWords(language)
    tfidf = feature_extractor.get_tfifd(all_texts)
    model = SVC(probability=True)
    model.fit(tfidf.transform(feature_extractor.preprocessing(dataset['text'])).toarray(), dataset['label'])
    with lock:
        with open(path_model, 'wb') as f:
            pickle.dump(model, f)
        with open(path_tfidf, 'wb') as f:
            pickle.dump(tfidf, f)


def predict(X, path_model, path_tfidf, language='russian'):
    feature_extractor = KeyWords(language)

    with lock:
        with open(path_model, 'rb') as f:
            model = pickle.load(f)
        with open(path_tfidf, 'rb') as f:
            tfidf = pickle.load(f)

    return model.predict_proba(tfidf.transform(feature_extractor.preprocessing(X)).toarray())
