# Customer-Emotion-Analysis-System
Customer Emotion Analysis SystemIntroductionThe Customer Emotion Analysis System is designed to analyze customer feedback from social media platforms, particularly Twitter, using Natural Language Processing (NLP) techniques. The system identifies emotions, assesses their intensity, maps them to topics, and provides structured insights, including an engagement score (Adorescore). This enables businesses to understand customer sentiments effectively and respond accordingly.
MethodologyData CollectionUsed the Kaggle Emotion Classification Dataset consisting of six emotions.
PreprocessingCleaned text by removing noise such as hashtags, mentions, URLs, and special characters.
Feature ExtractionUtilized TF-IDF and word embeddings to transform text data into meaningful numerical representations.
Emotion ClassificationImplemented a stacked model combining multiple classifiers, achieving a final accuracy of 0.77.
Model SelectionExperimented with different machine learning models and transformer-based models (e.g., BERT, RoBERTa, DistilBERT).
Selected the best-performing model based on F1-score for backend implementation.
Transformer-based models did not yield better accuracy in this case.
Backend & FrontendBackend: Developed using FastAPI, supporting both CSV and text input.
Frontend: Built using React.js, though full desired functionality was not achieved.
Frontend Integration: Created an interactive visualization interface to explore emotion trends in real time.
FindingsThe model effectively classified emotions with an accuracy of 0.68.
The most prevalent emotions detected were sadness, joy, and surprise.
Peak emotional expressions correlated with joy, highlighting key moments of heightened customer sentiment.
RecommendationsImprove accuracy by integrating advanced transformer-based models (e.g., BERT, RoBERTa), though previous attempts did not yield better results.
Enable real-time monitoring of customer emotions for dynamic insights.
Expand the dataset to incorporate multiple languages and cultural contexts for broader applicability.
ConclusionThe Customer Emotion Analysis System provides valuable insights into customer sentiments, enabling businesses to make data-driven decisions. Future enhancements, such as real-time emotion tracking and multilingual capabilities, will further improve its impact and usability.
Installation & SetupClone the repository:
git clone https://github.com/your-username/Customer-Emotion-Analysis.git
cd Customer-Emotion-AnalysisInstall dependencies:
pip install -r requirements.txtRun the backend server:
uvicorn main:app --reloadRun the frontend:
cd frontend
npm install
npm startTechnologies UsedPython (FastAPI, scikit-learn, TensorFlow, PyTorch, Transformers)
Machine Learning & NLP (BERT, RoBERTa, TF-IDF, Word2Vec)
Frontend: React.js
Data Processing: Pandas, NumPy
