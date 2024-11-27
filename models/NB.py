import logic.processing as lp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

df = lp.load_data('processed_combined_data.csv')


def create_NB():

    vecto = TfidfVectorizer(ngram_range=(1,1), max_df= 0.9)
    model = MultinomialNB(alpha=0.01)

    X = vecto.fit_transform(df['clean'])
    y = df['label']

    model.fit(X,y)

    X_new = vecto.transform(['I love life and want to live forever'])
    test = model.predict(X_new)
    return model, vecto


def save_model():
    model, vecto = create_NB()
    path = os.path.dirname(__file__)

    joblib.dump(vecto, os.path.join(path, 'tfidvecto.pkl'))
    joblib.dump(model, os.path.join(path, 'NB.pkl'))
    return None

def load_NB():
    path = os.path.dirname(__file__)
    vecto = joblib.load(os.path.join(path,'tfidvecto.pkl'))
    model = joblib.load(os.path.join(path,'NB.pkl'))
    return model, vecto




if __name__ == '__main__':
    save_model()
