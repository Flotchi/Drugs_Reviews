# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import models.NB

app = FastAPI()
app.state.model, app.state.vecto = models.NB.load_NB()

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
        response = 'Normal!'
    else: response = 'Crazy!'
    return {
        'prediction': response
        }
