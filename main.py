from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import re
import spacy
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from random import choice
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Load the models and vectorizer
with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load pre-trained emotion model (for emotion detection)
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Synonym Replacement for Augmentation
def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = choice(synonyms).lemmas()[0].name()
            new_words.append(synonym if synonym != word else word)
        else:
            new_words.append(word)
    return " ".join(new_words)

# FastAPI Request Model (For individual text input)
class FeedbackRequest(BaseModel):
    text: str

# Route for analyzing individual feedback
@app.post("/analyze/")
async def analyze(feedback: FeedbackRequest):
    try:
        text = feedback.text
        # Preprocess and vectorize text
        cleaned_text = preprocess_text(text)
        augmented_text = synonym_replacement(cleaned_text)
        vectorized_text = vectorizer.transform([augmented_text])

        # Predict using the stacked model
        prediction = model.predict(vectorized_text)
        emotion_scores = emotion_model(cleaned_text)
        emotion_scores = {e['label']: e['score'] for e in emotion_scores[0]}
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "feedback": text,
            "cleaned_feedback": cleaned_text,
            "augmented_feedback": augmented_text,
            "emotion_scores": emotion_scores,
            "predicted_emotion": predicted_emotion
        }

    except Exception as e:
        return {"error": str(e)}

# Route for predicting from CSV
@app.post("/predict/")
async def predict(csv_file: UploadFile = File(...)):
    try:
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode()))

        # Preprocess and predict
        df["cleaned_text"] = df["text"].apply(preprocess_text)
        df["augmented_text"] = df["cleaned_text"].apply(synonym_replacement)
        vectorized_data = vectorizer.transform(df["augmented_text"])

        predictions = model.predict(vectorized_data)
        df["prediction"] = predictions
        df["emotion_scores"] = df["cleaned_text"].apply(lambda text: emotion_model(text)[0])
        df["predicted_emotion"] = df["emotion_scores"].apply(lambda x: max({e['label']: e['score'] for e in x}, key=lambda k: x[k]))

        result = df[["text", "prediction", "emotion_scores", "predicted_emotion"]].to_dict(orient="records")
        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}
