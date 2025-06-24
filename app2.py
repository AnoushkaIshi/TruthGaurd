import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

# Initialize Core AI Models
fact_checking_pipeline = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")  # Fact-checking placeholder
sentiment_analysis = pipeline("sentiment-analysis")  # Sentiment analysis placeholder

def credibility_scoring(source):
    # Placeholder for assigning credibility scores to sources
    source_scores = {
        "reliable_source.com": 95,
        "suspicious_source.com": 50,
        "unknown_source.com": 30
    }
    return source_scores.get(source, 50)

def analyze_bias_and_sentiment(content):
    # Analyze bias and sentiment
    sentiment_result = sentiment_analysis(content)
    return sentiment_result[0]["label"], sentiment_result[0]["score"]

def summarize_article(article):
    # Placeholder for AI-Powered Summary
    return "This is a summarized version of the article: " + article[:100] + "..."

def validate_image(image):
    # Placeholder for deepfake and manipulation detection
    return "No manipulation detected."

def plot_propagation_trends():
    # Create a sample graph for propagation trends
    G = nx.erdos_renyi_graph(10, 0.3)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="gray")
    plt.title("Propagation Trends")
    st.pyplot(plt)

# Streamlit App Layout
st.set_page_config(page_title="TruthGuard", layout="wide")

# Adding custom CSS for styling the app
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: 'Arial', sans-serif;
        }
        
        .stButton>button {
            background-color: #ff007f;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }
        
        .stTextArea textarea {
            background-color: #333333;
            color: #f0f0f0;
            border-radius: 8px;
            padding: 10px;
        }

        .stTextInput input {
            background-color: #333333;
            color: #f0f0f0;
            border-radius: 8px;
            padding: 10px;
        }

        .stMarkdown {
            color: #f0f0f0;
        }

        .sidebar .sidebar-content {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }

        .sidebar .sidebar-header {
            color: #ff007f;
        }

        .sidebar .sidebar-radio label {
            color: #f0f0f0;
        }

        /* 24/7 Laser Effect */
        .laser-text {
            font-size: 6em;
            color: #ff007f;
            text-align: center;
            font-family: 'Courier New', monospace;
            animation: laser-effect 1s linear infinite;
            text-shadow: 0 0 5px #ff007f, 0 0 10px #ff007f, 0 0 15px #ff007f;
        }

        @keyframes laser-effect {
            0% { text-shadow: 0 0 5px #ff007f, 0 0 10px #ff007f, 0 0 15px #ff007f; }
            50% { text-shadow: 0 0 20px #ff007f, 0 0 40px #ff007f, 0 0 60px #ff007f; }
            100% { text-shadow: 0 0 5px #ff007f, 0 0 10px #ff007f, 0 0 15px #ff007f; }
        }

    </style>
""", unsafe_allow_html=True)

# 24/7 Laser Text Effect
st.markdown('<div class="laser-text">24/7</div>', unsafe_allow_html=True)

# Streamlit Sidebar for Navigation
st.sidebar.title("Navigation")
sections = ["Home", "Fact-Checking", "Multimedia Validation", "Propagation Analysis", "User Feedback"]
choice = st.sidebar.radio("Go to", sections)

if choice == "Home":
    st.header("Welcome to TruthGuard")
    st.markdown("""
    TruthGuard is an AI-powered platform designed to combat misinformation in real time. It offers:
    - Fact-checking of text and news articles.
    - Validation of multimedia content.
    - Analysis of misinformation propagation trends.
    - Insights into bias and sentiment.
    """)

elif choice == "Fact-Checking":
    st.header("AI-Driven Fact-Checking")
    article = st.text_area("Enter the news article content:")
    source = st.text_input("Enter the source URL:")

    if st.button("Analyze Article"):
        if article:
            credibility_score = credibility_scoring(source)
            sentiment_label, sentiment_score = analyze_bias_and_sentiment(article)
            summary = summarize_article(article)

            st.subheader("Results")
            st.write(f"*Source Credibility Score:* {credibility_score}/100")
            st.write(f"*Sentiment Analysis:* {sentiment_label} (Score: {sentiment_score:.2f})")
            st.write(f"*Summary:* {summary}")
        else:
            st.warning("Please enter article content.")

elif choice == "Multimedia Validation":
    st.header("Deepfake and Image Validation")
    uploaded_file = st.file_uploader("Upload an image or video:", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        validation_result = validate_image(image)
        st.subheader("Validation Result")
        st.write(validation_result)

elif choice == "Propagation Analysis":
    st.header("Propagation and Trend Analysis")
    st.markdown("Visualizing the spread of misinformation across networks.")
    plot_propagation_trends()

elif choice == "User Feedback":
    st.header("User Feedback and Customization")
    st.markdown("Provide feedback to improve our system.")

    feedback = st.text_area("Your Feedback:")
    alert_preference = st.multiselect("Set Alerts for Topics:", ["Politics", "Health", "Environment", "Technology"])

    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please provide some feedback.")

st.sidebar.markdown("---")
st.sidebar.write("TruthGuard \u00a9 2024")