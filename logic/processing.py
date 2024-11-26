import pandas as pd
import numpy as np
import os
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(file):
    path = __file__
    rpath = os.path.join(os.path.dirname(os.path.dirname(path)),'raw_data', file)
    df = pd.read_csv(rpath)
    return df
def preproc(df, bi = False):

    df.dropna(inplace=True)

    def process(st):
        for punc in string.punctuation:
            st = st.replace(punc, '')
        ans = st.casefold()
        ansd = ''.join(x for x in ans if not x.isdigit())
        stop = set(stopwords.words('english'))
        tokens = word_tokenize(ansd)
        ansdd = [y for y in tokens if y not in stop]
        lemmaverb = [WordNetLemmatizer().lemmatize(word, pos='v') for word in ansdd]
        lemmanouns = [WordNetLemmatizer().lemmatize(word, pos='n') for word in lemmaverb]
        nans = ' '.join(lemmanouns)
        return nans
    df['clean'] = df['statement'].apply(process)

    def encode_bi(st):
        return int(st == 'Normal')
    if bi:
        df['label'] = df['status'].apply(encode_bi)
    return df

if __name__ == '__main__':
    print(preproc(load_data(), bi = True))
