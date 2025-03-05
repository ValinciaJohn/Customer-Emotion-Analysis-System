import pandas as pd
import re
import spacy
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from transformers import pipeline
from random import choice
import io


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("test.csv")

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Synonym Replacement for Augmentation
nltk.download('wordnet')
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

# Apply Preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Data Augmentation using Synonym Replacement
df["augmented_text"] = df["cleaned_text"].apply(synonym_replacement)

# Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["augmented_text"])
y = df["label"].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Class Imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train SVM with GridSearchCV
svm = SVC()
svm.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

# Stacked Model (SVM + XGBoost)
stacked_model = StackingClassifier(
    estimators=[("svm", svm), ("xgb", xgb)],
    final_estimator=SVC(kernel="linear", probability=True)
)
stacked_model.fit(X_train, y_train)

# Save Model & Vectorizer
with open("final_model.pkl", "wb") as f:
    pickle.dump(stacked_model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Load pre-trained emotion model (for emotion detection)
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Function to Detect Emotions using Pre-trained Model
def detect_emotions(text):
    emotions = emotion_model(text)
    emotions = emotions[0]
    scores = {e['label']: e['score'] for e in emotions}
    return scores

# FastAPI Request Model (For individual text input)
class FeedbackRequest(BaseModel):
    text: str

# Automate predictions for the CSV and save as JSON on startup
@app.on_event("startup")
def startup_event():
    # Preprocess the data in the test CSV file
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    df["augmented_text"] = df["cleaned_text"].apply(synonym_replacement)

    # Vectorization (TF-IDF)
    vectorized_data = vectorizer.transform(df["augmented_text"])

    # Load the model
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict using the trained model
    predictions = model.predict(vectorized_data)
    df["prediction"] = predictions

    # Detect emotions using the emotion detection model
    df["emotion_scores"] = df["cleaned_text"].apply(detect_emotions)
    df["predicted_emotion"] = df["emotion_scores"].apply(lambda x: max(x, key=x.get))

    # Generate JSON output and save to file
    result = df[["text", "prediction", "emotion_scores", "predicted_emotion"]].to_dict(orient="records")
    with open("predictions_output.json", "w") as json_file:
        json.dump({"predictions": result}, json_file)

# Route for predicting from CSV
@app.post("/predict/")
async def predict(csv_file: UploadFile = File(...)):
    try:
        contents = await csv_file.read()
        # Use io.StringIO instead of pandas.compat.StringIO
        df = pd.read_csv(io.StringIO(contents.decode()))
        
        # Preprocess and predict like above
        df["cleaned_text"] = df["text"].apply(preprocess_text)
        df["augmented_text"] = df["cleaned_text"].apply(synonym_replacement)
        vectorized_data = vectorizer.transform(df["augmented_text"])

        # Load the model
        with open("final_model.pkl", "rb") as f:
            model = pickle.load(f)

        predictions = model.predict(vectorized_data)
        df["prediction"] = predictions
        df["emotion_scores"] = df["cleaned_text"].apply(detect_emotions)
        df["predicted_emotion"] = df["emotion_scores"].apply(lambda x: max(x, key=x.get))

        result = df[["text", "prediction", "emotion_scores", "predicted_emotion"]].to_dict(orient="records")
        return {"predictions": result}
    
    except Exception as e:
        return {"error": str(e)}

# Route for analyzing individual feedback
@app.post("/analyze/")
async def analyze(feedback: FeedbackRequest):
    try:
        text = feedback.text
        # Preprocess and vectorize text
        cleaned_text = preprocess_text(text)
        augmented_text = synonym_replacement(cleaned_text)  # Apply augmentation
        vectorized_text = vectorizer.transform([augmented_text])

        # Predict using the stacked model
        prediction = stacked_model.predict(vectorized_text)
        emotion_scores = detect_emotions(cleaned_text)
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
    
    # Predict using the stacked model
    prediction = stacked_model.predict(vectorized_text)
    emotion_scores = detect_emotions(cleaned_text)
    predicted_emotion = max(emotion_scores, key=emotion_scores.get)
    
    return {
        "feedback": text,
        "cleaned_feedback": cleaned_text,
        "augmented_feedback": augmented_text,
        "emotion_scores": emotion_scores,
        "predicted_emotion": predicted_emotion
    }

# app.py (for FastAPI to run directly without Uvicorn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

#http://127.0.0.1:8000/docs 