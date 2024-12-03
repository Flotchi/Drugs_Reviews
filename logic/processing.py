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
    df = pd.read_csv(rpath, na_values=['NaN'], keep_default_na=False)
    return df


def preproc(df, bi = False):

    df.dropna(inplace=True)

    def process(st):
        for punc in string.punctuation:
            st = st.replace(punc, '')
        ans = st.casefold().replace('\n', ' ')
        ansd = ''.join(x for x in ans if not x.isdigit())
        stop = set(stopwords.words('english'))
        tokens = word_tokenize(ansd)

        ansdd = [y for y in tokens if y not in stop]
        ansdd = [y for y in tokens if y not in stop]

        lemmaverb = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]
        lemmanouns = [WordNetLemmatizer().lemmatize(word, pos='n') for word in lemmaverb]
        nans = ' '.join(lemmanouns)
        return nans

    df['clean'] = df['review'].apply(process)


    if bi:
        df['label'] = df['status'].apply(lambda st: int(st == 'Normal'))
    return df

def balance_dataset(X, y):

    df = pd.DataFrame({'review': X, 'sentiment': y})

    good_reviews = df[df['sentiment'] == 'Good']
    bad_reviews = df[df['sentiment'] == 'Bad']


    min_size = min(len(good_reviews), len(bad_reviews))


    good_balanced = good_reviews.sample(n=min_size, random_state=42)
    bad_balanced = bad_reviews.sample(n=min_size, random_state=42)


    balanced_df = pd.concat([good_balanced, bad_balanced]).sample(frac=1, random_state=42)


    X_balanced = balanced_df['review']
    y_balanced = balanced_df['sentiment']

    return X_balanced, y_balanced

if __name__ == '__main__':
    data = load_data('Combined Data.csv')
    df = preproc(data, bi=True)
    path = __file__
    rpath = os.path.join(os.path.dirname(os.path.dirname(path)), 'raw_data', 'processed_combined_data.csv')
    df.to_csv(rpath, na_rep='NaN', index=False)
