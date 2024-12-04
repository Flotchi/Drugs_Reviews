import joblib
import os
import logic.processing as lp
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences




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

    return Xpad


def save_padded():
    X = create_embeding()
    path = os.path.dirname(os.path.dirname(__file__))
    joblib.dump(X, os.path.join(path,'models', 'Xpad.pkl'))
    print('dumping done')
    return None


if __name__ == '__main__':
    save_padded()
