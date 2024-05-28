from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from googletrans import Translator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

translator = Translator()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


DATABASE_URL = "postgresql://postgres:erkebulan2001@localhost/CustomerSentimentAnalysis"

Base = declarative_base()

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Dataset(Base):
    __tablename__ = "dataset1"
    
    comment_id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    date = Column(Date)
    customer_reviews = Column(String)
    usefulnes = Column(String)
    device = Column(String)
    translated_text = Column(String)
    helpfulness_count = Column(Integer)
    out_of = Column(Integer)
    ratio = Column(Float)
    aspects = Column(String)
    sentiment = Column(String)

Base.metadata.create_all(bind=engine)




# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load models
sentiment_model = load_model('sentiment_model.h5')
aspect_models = {
    "delivery": load_model('aspect_model_delivery.h5'),
    "price": load_model('aspect_model_price.h5'),
    "packaging": load_model('aspect_model_packaging.h5'),
    "quality": load_model('aspect_model_quality.h5'),
}

MAX_SEQUENCE_LENGTH = 100  

class AnalysisRequest(BaseModel):
    text: str
    username: str
    device: str


def translate_to_english(text):
    try:
        translation = translator.translate(text, dest='en')
        return translation.text

    except Exception as e:
        print(f"Error during translation: {e}")
        return text


def preprocess_text(text: str):
    # translated_text = translate_to_english(text)
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences

@tf.function
def predict_sentiment(input_data):
    return sentiment_model(input_data)

    
@app.post("/predict/analysis/")
async def predict_analysis(request: AnalysisRequest):
    translated_text = translate_to_english(request.text)
    processed_text = preprocess_text(translated_text)

    sentiment_prediction = predict_sentiment(processed_text)
    sentiment_score = float(sentiment_prediction[0][0])
    sentiment = "Positive" if sentiment_score > 0.3 else "Negative"

    aspects = {}
    for aspect, model in aspect_models.items():
        prediction = model.predict(processed_text)
        aspects[aspect] = float(prediction[0][0])

    highest_scoring_aspect = max(aspects, key=aspects.get)

    db = SessionLocal()
    try:
        db_item = Dataset(
            username=request.username,
            date=datetime.date.today(),
            customer_reviews=request.text,
            usefulnes='',
            device=request.device,
            translated_text=translated_text,
            helpfulness_count=0,
            out_of=0,
            ratio=0,
            aspects=highest_scoring_aspect,
            sentiment=sentiment
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()

    return {
        "sentiment": sentiment,
        "aspects": highest_scoring_aspect
    }