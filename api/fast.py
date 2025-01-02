# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logic.NB
import logic.processing_new_input
import joblib
import os


app = FastAPI()
app.state.model, app.state.vecto = logic.NB.load_NB()
path = os.path.dirname(os.path.dirname(__file__))
app.state.seb = joblib.load(os.path.join(path,'models', 'lstm2.pkl'))

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
def get_predict(st):
    model = app.state.model
    vecto = app.state.vecto
    X_new = vecto.transform([str(st)])
    pred = model.predict(X_new)
    ans = int(pred[0])
    if ans:
        response = "Oh no, you are normal"
    else: response = 'Congratulations ! You are crazy'
    return {
        'prediction': response
        }



@app.get('/lstm')
def get_lstm(st):
    model = app.state.seb
    X = logic.processing_new_input.processing_new(st)
    pred = model.predict(X)[0][0]
    return {'answer' : float(pred) }
