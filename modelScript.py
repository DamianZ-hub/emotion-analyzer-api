import io
import itertools

import keras.layers
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import tensorflow as tf
import os
from sklearn.utils import shuffle


root_logdir = os.path.join(os.curdir, "diaries")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
additional_stop_words = set({"ive", "id", "im"})
stop_words = stop_words.union(additional_stop_words)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def to_lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )

    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalization(df):
    df.Text = df.Text.apply(lambda Text: to_lower_case(Text))
    df.Text = df.Text.apply(lambda Text: remove_stop_words(Text))
    df.Text = df.Text.apply(lambda Text: remove_numbers(Text))
    df.Text = df.Text.apply(lambda Text: remove_punctuations(Text))
    df.Text = df.Text.apply(lambda Text: remove_urls(Text))
    df.Text = df.Text.apply(lambda Text: lemmatization(Text))
    return df

def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(X_val)

    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = sklearn.metrics.confusion_matrix(np.argmax(y_val, axis=1), test_pred)

    _class_names = [0,1,2,3,4,5]
    figure = plot_confusion_matrix(cm, class_names=_class_names)
    cm_image = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

def plot_to_image(figure):

    buf = io.BytesIO()

    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)

    return image

def plot_confusion_matrix(cm, class_names):

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives +
    tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))


df_train = pd.read_csv('./inputs/train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('./inputs/val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('./inputs/test.txt', names=['Text', 'Emotion'], sep=';')


index = df_train[df_train['Text'].duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

index = df_val[df_val['Text'].duplicated() == True].index
df_val.drop(index, axis = 0, inplace = True)
df_val.reset_index(inplace=True, drop = True)

index = df_test[df_train['Text'].duplicated() == True].index
df_test.drop(index, axis = 0, inplace = True)
df_test.reset_index(inplace=True, drop = True)


df_train= normalization(df_train)

df_test= normalization(df_test)

df_val= normalization(df_val)

X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

vocabSize = len(tokenizer.index_word) + 1
maxlen = max([len(t) for t in sequences_train])

X_train = tf.keras.utils.pad_sequences(sequences_train, maxlen=maxlen, truncating='pre')
X_test = tf.keras.utils.pad_sequences(sequences_test, maxlen=maxlen, truncating='pre')
X_val = tf.keras.utils.pad_sequences(sequences_val, maxlen=maxlen, truncating='pre')

X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_val = tf.keras.utils.to_categorical(y_val)


dims = 200
glove_file = './glove.6B.200d.txt'
num_tokens = vocabSize
embedding_dim = dims
embeddings_index = {}

with open(glove_file, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("%s word vectors have been found." % len(embeddings_index))

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

adam = tf.keras.optimizers.Adam(learning_rate=0.005)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocabSize, dims, input_length=X_train.shape[1], weights=[embedding_matrix], trainable=False))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, dropout=0.3, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', f1_m,precision_m, recall_m])
model.summary()

cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
)
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(),
                                                histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)

file_writer_cm = tf.summary.create_file_writer(get_run_logdir() + '/cm')

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    batch_size=32,
                    epochs=30,
                    callbacks=[callback, tensorboard_cb, cm_callback]
                   )
model.save("model.h5")
print(model.evaluate(X_test, y_test, verbose=1))
print("done")
