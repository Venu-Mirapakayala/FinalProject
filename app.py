from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all domains, for dev purposes
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods: GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

# Load the model and tokenizer
model = load_model("sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    seq = tokenizer.texts_to_sequences([input.text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    return {"prediction": sentiment, "confidence": float(pred)}
