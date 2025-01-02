import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import os
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer

def processing_new(review):
    path = os.path.dirname(os.path.dirname(__file__))
    word2vec = joblib.load(os.path.join(path,'models', 'word2vec.pkl'))

    for punc in string.punctuation:
        review = review.replace(punc, '')
    review = review.casefold().replace('\n', ' ')
    review = ''.join(x for x in review if not x.isdigit())
    review = [WordNetLemmatizer().lemmatize(word, pos='v') for word in review.split()]
    review = [WordNetLemmatizer().lemmatize(word, pos='n') for word in review]
    review_clean = ' '.join(review)

    review_clean_tk = text_to_word_sequence(review_clean)

    embedded_sentence = []
    for word in review_clean_tk:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])

    embedded_sentence = np.array(embedded_sentence)

    embedded_sentence = embedded_sentence.reshape(1,embedded_sentence.shape[0],embedded_sentence.shape[1])

    embedded_sentence_pad = pad_sequences(np.array(embedded_sentence), dtype='float32', padding='post', maxlen=200)

    return embedded_sentence_pad

if __name__ == '__main__':
    print(processing_new('test').shape)
