import streamlit as st
import plotly.graph_objects as go
import json

# Set page configuration for Streamlit app
st.set_page_config(page_title="Customer Emotion Analysis System", layout="wide")

# Title of the App
st.title("Customer Emotion Analysis System")
st.markdown("""
This dashboard provides an in-depth analysis of the customer feedback, showcasing various emotional insights, activation levels, and their overall engagement (**Adorescore**). 
The visualizations will help you understand the emotional intensity and themes associated with customer sentiment.
""")

# Sidebar for navigation and input
st.sidebar.header("Input Options")

# Option to Upload JSON file or Enter Sentence for prediction
input_option = st.sidebar.radio("Choose input type", ["Upload JSON File", "Enter a Sentence"])

# Initialize variables for emotion scores and adorescore
emotion_scores = {}
adorescore = 0
top_themes = {}

# Function to generate emotion intensity plot
def plot_emotion_intensity(emotion_scores):
    fig = go.Figure()

    for emotion, score in emotion_scores.items():
        fig.add_trace(go.Bar(
            x=[emotion],
            y=[score],
            name=emotion.capitalize(),
            marker_color='rgba(50, 50, 255, 0.6)',
            text=[f"{score:.2f}"],
            textposition='outside',
            showlegend=False
        ))

    fig.update_layout(
        title="Emotion Intensity Plot",
        xaxis_title="Emotions",
        yaxis_title="Intensity",
        barmode="group",
        template="plotly_dark",
        height=400,
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="rgba(0, 0, 0, 0)"
    )

    st.plotly_chart(fig)

# Function to handle JSON file upload
def handle_json_upload(uploaded_file):
    global emotion_scores, adorescore, top_themes

    # Load and process JSON data
    emotion_data = json.load(uploaded_file)

    # Get emotion scores and top themes
    emotion_scores = emotion_data["emotion_scores"]
    adorescore = emotion_data["adorescore"]
    top_themes = emotion_data["top_themes"]

    # Plot emotion intensity
    plot_emotion_intensity(emotion_scores)

    # Display Adorescore
    st.metric(label="Adorescore", value=f"{adorescore}", delta="Joy - 50%")

    # Display Top Themes
    st.subheader("Top Themes in Dataset")
    for theme, score in top_themes.items():
        st.write(f"{theme}: {score}")

# Function to handle sentence input
def handle_sentence_input(sentence):
    # Here, you would add your sentence analysis and JSON response generation logic
    # For now, we use a dummy JSON structure for demonstration
    dummy_json = {
        "emotion_scores": {
            "sadness": 0.85,
            "joy": 0.15,
            "love": 0.05,
            "anger": 0.25,
            "fear": 0.1,
            "surprise": 0.05
        },
        "adorescore": 65,
        "top_themes": {
            "Delivery": 80,
            "Quality": 75,
            "Clothing": 60
        }
    }

    # Simulate processing sentence (replace with actual processing logic)
    emotion_scores = dummy_json["emotion_scores"]
    adorescore = dummy_json["adorescore"]
    top_themes = dummy_json["top_themes"]

    # Plot emotion intensity
    plot_emotion_intensity(emotion_scores)

    # Display Adorescore
    st.metric(label="Adorescore", value=f"{adorescore}", delta="Joy - 50%")

    # Display Top Themes
    st.subheader("Top Themes in Dataset")
    for theme, score in top_themes.items():
        st.write(f"{theme}: {score}")

# Based on the selected input option, either upload JSON file or process the sentence
if input_option == "Upload JSON File":
    uploaded_file = st.file_uploader("Upload Emotion Analysis JSON", type=["json"])
    if uploaded_file is not None:
        handle_json_upload(uploaded_file)

elif input_option == "Enter a Sentence":
    sentence = st.text_area("Enter a sentence for emotion analysis:")
    if sentence:
        handle_sentence_input(sentence)

