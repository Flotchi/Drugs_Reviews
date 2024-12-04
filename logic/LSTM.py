import joblib
import os
import logic.processing as lp
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import InputLayer,Dropout,BatchNormalization, Bidirectional,LSTM,Dense
from tensorflow.keras.callbacks import EarlyStopping



def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])

    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    embed = []
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed


def create_embeding():
    df1 = lp.load_data('drugsComTrain_raw.csv')
    print('data loaded, filtering')
    df2 = lp.data_filter(df1)
    print('filtering done, preprocessing')
    df3 = lp.preproc(df2)
    print('preprocessing done, balancing')
    X = df3['clean']
    y = df3['sentiment']
    Xb, yb = lp.balance_dataset(X,y)
    print('balancing done, tokenizing')
    Xtok = [text_to_word_sequence(_) for _ in Xb]
    word2vec = Word2Vec(sentences=Xtok, vector_size=60, min_count=10, window=10)
    Xemb = embedding(word2vec, Xtok)
    Xpad = pad_sequences(Xemb, dtype='float32', padding= 'post', maxlen=200)
    print('padding done')

    return (Xpad, y)


def save_padded():
    X = create_embeding()
    path = os.path.dirname(os.path.dirname(__file__))
    joblib.dump(X, os.path.join(path,'models', 'Xpad.pkl'))
    print('dumping done')
    return None


def init_model():
    model = Sequential()
    model.add(InputLayer((200,60)))
    model.add(layers.Masking(mask_value=0.))
    model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_model(X,y):
    model = init_model()
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X, y,
          batch_size=8,
          epochs=50,
          validation_split=0.3,
          callbacks=[es]
         )

    path = os.path.dirname(os.path.dirname(__file__))
    joblib.dump(model, os.path.join(path,'models', 'LSTM.pkl'))
    return model




if __name__ == '__main__':
    # save_padded()
    path = os.path.dirname(os.path.dirname(__file__))
    X, y = joblib.load(os.path.join(path,'models', 'Xpad.pkl'))
    train_model(X,y)
