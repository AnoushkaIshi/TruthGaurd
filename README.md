# TruthGuard: Misinformation Detection System 🚀

An AI-powered platform that detects and analyzes misinformation in real-time across multiple media formats. This project was a top 25 finalist in a global hackathon with 5000+ competing teams.

## Features ✨

- **Real-Time Claim Parsing**: Extract claims from audio/video content
- **Multilingual Support**: Works with English and Hindi content
- **Misinformation Classification**: AI-powered fact-checking of textual claims
- **Social Listening**: Monitor trending news and social media content
- **Interactive Visualization**: Dashboard with insightful data visualizations
- **Collaborative Fact-Checking**: Crowdsourced verification system
- **Deepfake Detection**: Multimedia validation capabilities

## Tech Stack 🛠️

### Backend
- Python 3.9+
- Streamlit (Web Framework)
- HuggingFace Transformers (NLP Models)
- PyTorch (Deep Learning)

### NLP Models
- BERT (Claim Classification)
- DeBERTa (Fact-Checking)
- NER Models (Claim Extraction)

### APIs
- Google Speech Recognition (Audio Transcription)
- NewsAPI (Social Listening)

## Installation ⚙️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/misinformation-detection-system.git
cd misinformation-detection-system
``` 
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set up environment variables:
```bash
echo "NEWS_API_KEY=your_api_key_here" > .env
```
## Usage 
1. Running the Application
```bash
streamlit run app.py
```
## Modules Overview
### Real-Time Claim Parsing:
- Upload audio/video files

- Automatic transcription to text

- Claim extraction from transcribed content

### Misinformation Detection:

- Enter claims manually or paste text

- AI classification with confidence scores

- Historical analysis of similar claims

### Social Listening:

- Monitor keywords across news sources

- Trend analysis visualization

- Source credibility scoring

### Visualization Dashboard:

- Interactive charts and graphs

- Temporal analysis of misinformation spread

## API Keys 🔑
The system requires the following API keys (add to .env file):

1. NEWS_API_KEY - From newsapi.org

2. GOOGLE_APPLICATION_CREDENTIALS - For speech recognition (optional)


