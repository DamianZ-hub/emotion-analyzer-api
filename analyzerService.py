import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
import re

from joblib import load

class AnalyzerService:

    stop_words = None
    records_count = 8000
    le = None

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        _stop_words_ = set(stopwords.words("english"))
        additional_stop_words = set({"ive", "id", "im"})
        self.stop_words = _stop_words_.union(additional_stop_words)
        self.le = load('le_scaler.bin')

    def analyze_emotion(self, text):
        df = pd.DataFrame([text], columns=['text'])
        return self.process(df, 'text')

    def process(self, df, column):
        X_txt = df[column]
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK')
        tokenizer.fit_on_texts(X_txt)

        sequences_train = tokenizer.texts_to_sequences(X_txt)
        maxlen = 35
        X = tf.keras.utils.pad_sequences(sequences_train, maxlen=maxlen, truncating='pre')

        model = tf.keras.models.load_model("model.h5",
                                           custom_objects={"recall_m": self.recall_m, "precision_m": self.precision_m,
                                                           "f1_m": self.f1_m})

        results = model.predict(X)
        results_df = pd.DataFrame(results, columns=self.le.classes_)
        results_df['prediction'] = results_df.apply(lambda row : np.argmax(row, axis=-1), axis = 1)
        results_df['prediction_txt'] = results_df['prediction'].apply(lambda row : self.le.inverse_transform([row])[0])
        return results_df.to_json(orient='records')

    def recall_m(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives +
                                   tf.keras.backend.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)

    def remove_stop_words(self, text):
        Text = [i for i in str(text).split() if i not in self.stop_words]
        return " ".join(Text)

    def Removing_numbers(self, text):
        text = ''.join([i for i in text if not i.isdigit()])
        return text

    def Removing_linebreaks(self, text):
        text = text.replace('\n', ' ').replace('\r', '')
        return text

    def lower_case(self, text):
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)

    def Removing_punctuations(self, text):
        text = re.sub('[%s]' % re.escape("""!"$%&'()*+,،-./:;<=>؟?[\]^_’`{|}~"""), '', text)
        text = text.replace('؛', "", )

        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()

    def remove_usermentions(self, text):
        text = re.sub("@[A-Za-z0-9_]+", "", text)
        return text

    def remove_hashtags(self, text):
        text = re.sub("#[A-Za-z0-9_]+", "", text)
        return text

    def remove_non_alphanumerical(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        return text

    def remove_unnecesary_whitespaces(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_text(self, df, columnName):
        df[columnName] = df[columnName].apply(lambda row: self.lower_case(row))
        df[columnName] = df[columnName].apply(lambda row: self.remove_stop_words(row))
        df[columnName] = df[columnName].apply(lambda row: self.Removing_numbers(row))
        df[columnName] = df[columnName].apply(lambda row: self.Removing_linebreaks(row))
        df[columnName] = df[columnName].apply(lambda row: self.Removing_punctuations(row))
        df[columnName] = df[columnName].apply(lambda row: self.remove_usermentions(row))
        df[columnName] = df[columnName].apply(lambda row: self.remove_hashtags(row))
        df[columnName] = df[columnName].apply(lambda row: self.remove_unnecesary_whitespaces(row))
        df[columnName] = df[columnName].apply(lambda row: self.remove_non_alphanumerical(row))
        df[columnName] = df[columnName].apply(lambda row: self.lemmatization(row))
        return self
