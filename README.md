# Customer-Emotion-Analysis-System
1.	Introduction:
The Customer Emotion Analysis System is designed to analyze customer feedback from social media platforms, particularly Twitter, using Natural Language Processing (NLP) techniques. The system identifies emotions, assesses their intensity, maps them to topics, and provides structured insights, including an engagement score (Adorescore). This enables businesses to understand customer sentiments effectively and respond accordingly.
2.	Methodology:
Data Collection:Used the Kaggle Emotion Classification Dataset consisting of six emotions.
Preprocessing:Cleaned text by removing noise such as hashtags, mentions, URLs, and special characters.
Feature Extraction:Utilized TF-IDF and word embeddings to transform text data into meaningful numerical representations.
Emotion Classification:Implemented a stacked model combining multiple classifiers, achieving a final accuracy of 0.77.
Model Selection:Experimented with different machine learning models and transformer-based models (e.g., BERT, RoBERTa, DistilBERT).
                Selected the best-performing model based on F1-score for backend implementation.
                Transformer-based models did not yield better accuracy in this case.
Backend & FrontendBackend: Developed using FastAPI, supporting both CSV and text input.
Frontend: Built using React.js, though full desired functionality was not achieved.
Frontend Integration: Created an interactive visualization interface to explore emotion trends in real time.
3.Findings:
The model effectively classified emotions with an accuracy of 0.68.
The most prevalent emotions detected were sadness, joy, and surprise.
Peak emotional expressions correlated with joy, highlighting key moments of heightened customer sentiment.
4.Conclusion:
The Customer Emotion Analysis System provides valuable insights into customer sentiments, enabling businesses to make data-driven decisions. Future enhancements, such as real-time emotion tracking and multilingual capabilities, will further improve its impact and usability.
