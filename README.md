# Customer Emotion Analysis System 📌

## Introduction  
The **Customer Emotion Analysis System** is a powerful tool designed to analyze customer feedback from social media platforms, particularly Twitter, using **Natural Language Processing (NLP)** techniques. By identifying emotions, assessing their intensity, mapping them to relevant topics, and providing structured insights — including an engagement score (Adorescore) — businesses can understand customer sentiments effectively and respond accordingly.

---

## 🛠️ Methodology  
### Data Collection  
- The system utilizes the **Kaggle Emotion Classification Dataset**, consisting of six labeled emotions: **sadness**, **joy**, **anger**, **fear**, **love**, and **surprise**.

### Data Preprocessing  
- **Noise Removal:** Eliminated hashtags, mentions, URLs, and special characters to retain meaningful text.  
- Applied **tokenization** and **lemmatization** to standardize and simplify text.

### Feature Extraction  
- Used **TF-IDF** and **word embeddings** to convert text data into numerical representations.

### Emotion Classification  
- Implemented a stacked model combining multiple classifiers, achieving a final accuracy of **0.77**.  
- Experimented with machine learning models and transformer-based models (**BERT**, **RoBERTa**, **DistilBERT**).  
- The best-performing model based on **F1-score** was selected for backend implementation.  
- Transformer-based models did not significantly improve accuracy.

### Backend and Frontend  
- **Backend:** Developed with **FastAPI**, allowing CSV and text input for prediction.  
- **Frontend:** Built using **React.js**. An interactive interface for visualizing real-time emotion trends has been integrated, though full functionality is still in progress.

---

## 📊 Findings  
- The model effectively classified emotions with an accuracy of **0.68**.  
- The most prevalent emotions detected were **sadness**, **joy**, and **surprise**.  
- Peak emotional expressions often correlated with **joy**, indicating key moments of positive customer sentiment.

---

## 🚀 Conclusion & Future Scope  
The **Customer Emotion Analysis System** provides valuable insights for businesses to make data-driven decisions. Future improvements include:  
- **Real-time emotion tracking** for instant feedback monitoring.  
- **Multilingual capabilities** to extend reach and enhance usability.  
- Integration of advanced transformer-based models for improved accuracy.


