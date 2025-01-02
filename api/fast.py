# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logic.LSTM

app = FastAPI()
app.state.model, app.state.vecto = logic.NB.load_NB()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }


# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(review):
    model = app.state.model
    word2vec = app.state.word2vec

    review_preproc = prepoc(review)
    review_tk =

    X_new = embed(st)
    pred = model.predict(X_new)
    ans = int(pred[0])
    if ans:
        response = "Oh no, you are normal"
    else: response = 'Congratulations ! You are crazy'
    return {
        'prediction': response
        }
