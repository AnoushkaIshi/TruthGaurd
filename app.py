import streamlit as st
from claim_parsing import speech_to_text
from misinformation_check import classify_claims
from social_listening import fetch_tweets
from visualization import visualize_data

# Page title and layout
st.set_page_config(page_title="Misinformation Detection System", layout="wide")

# Sidebar navigation
st.sidebar.title("Modules")
selected_module = st.sidebar.radio("Choose a module", [
    "Real-Time Claim Parsing",
    "Misinformation Detection",
    "Social Listening",
    "Visualization"
])


# Real-Time Claim Parsing
if selected_module == "Real-Time Claim Parsing":
    st.title("Real-Time Claim Parsing")
    uploaded_file = st.file_uploader("Upload an Audio/Video file", type=["wav", "mp3", "mp4"])

    if uploaded_file:
        st.write("Processing the uploaded file...")

        try:
            if uploaded_file.name.endswith('.mp4'):
                st.write("Detected Video file...")
                audio_path = extract_audio_from_video(uploaded_file)
                text = speech_to_text(audio_path)

            else:
                st.write("Detected Audio File...")
                text = speech_to_text(uploaded_file)

            st.write(f"Transcribed Text (English & Hindi): {text}")

        except Exception as err:
            st.error(f"Failed transcription: {str(err)}")
# Misinformation Detection
elif selected_module == "Misinformation Detection":
    st.title("Misinformation Detection")
    claims = st.text_area("Enter claims (one per line):")

    if st.button("Check Claims"):
        if claims:
            claims_list = claims.split("\n")
            results = classify_claims(claims_list)

            st.write("## Detection Results 🧵")
            for res in results:
                st.write(f"✅ **Claim:** {res['claim']}")
                st.write(f"🔍 **Classification:** {res['classification']['label']}")
                st.write(f"📊 **Confidence:** {res['classification']['confidence']}")
                st.write("---")

# Social Listening
# Social Listening
if selected_module == "Social Listening":
    st.title("Social Listening")
    keyword = st.text_input("Enter a keyword to monitor:")

    if st.button("Fetch News"):
        articles = fetch_tweets(keyword, count=10)

        if articles:
            st.write("### Trending News Articles")
            for idx, article in enumerate(articles):
                st.markdown(f"**{idx + 1}. {article}**")
        else:
            st.write("No news articles were found. Try a different keyword.")

# Visualization
elif selected_module == "Visualization":
    st.title("Visualization")
    sample_data = [{"claim": "Earth is flat", "classification": "False"},
                   {"claim": "COVID-19 vaccines save lives", "classification": "True"}]
    visualize_data(sample_data)
