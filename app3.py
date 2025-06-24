import streamlit as st
# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Misinformation Detection System")

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from datetime import datetime
import requests
from typing import List, Dict, Any
import json
import feedparser

# Configuration
class Config:
    NEWS_FEEDS = [
        "http://rss.cnn.com/rss/cnn_topstories.rss",
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reddit.com/r/news/.rss",
        "https://news.google.com/news/rss"
    ]
    SUPPORTED_LANGUAGES = ["en", "es", "fr", "de"]

class ClaimExtractor:
    def __init__(self):
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    def extract_claims(self, text: str) -> List[str]:
        entities = self.ner_pipeline(text)
        claims = []
        current_claim = ""
        for entity in entities:
            if entity["entity"] in ["PER", "ORG", "LOC"]:
                current_claim += f"{entity['word']} "
            if len(current_claim) > 50:
                claims.append(current_claim.strip())
                current_claim = ""
        return claims

class MisinformationDetector:
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
    def check_claim(self, claim: str, reference_data: List[str]) -> tuple:
        labels = ["true", "false", "uncertain"]
        scores = []
        for reference in reference_data:
            inputs = self.tokenizer(
                f"Claim: {claim} Reference: {reference}",
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores.append(torch.softmax(outputs.logits, dim=1))
        
        final_score = torch.mean(torch.stack(scores), dim=0)
        prediction = labels[torch.argmax(final_score).item()]
        confidence = final_score.max().item()
        return prediction, confidence

class NewsMonitor:
    def __init__(self, feeds: List[str]):
        self.feeds = feeds
        
    def get_recent_news(self) -> List[Dict[str, Any]]:
        news_items = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    news_items.append({
                        'title': entry.title,
                        'content': entry.description,
                        'timestamp': datetime.now(),
                        'source': feed_url
                    })
            except Exception as e:
                st.error(f"Error fetching feed {feed_url}: {str(e)}")
        return news_items
    
    def search_news(self, keywords: List[str]) -> List[Dict[str, Any]]:
        relevant_news = []
        all_news = self.get_recent_news()
        for news in all_news:
            if any(keyword.lower() in news['content'].lower() for keyword in keywords):
                relevant_news.append(news)
        return relevant_news

class VisualizationDashboard:
    def __init__(self):
        self.claims_data = []
        
    def update_data(self, new_claims: List[Dict[str, Any]]):
        self.claims_data.extend(new_claims)
    
    def render_dashboard(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live Claims Analysis")
            if self.claims_data:
                df = pd.DataFrame(self.claims_data)
                fig_heatmap = px.density_heatmap(
                    df,
                    x="timestamp",
                    y="confidence",
                    color_continuous_scale="Viridis",
                    title="Claims Confidence Distribution Over Time"
                )
                st.plotly_chart(fig_heatmap)
                
                st.subheader("Recent Claims Timeline")
                for claim in reversed(self.claims_data[-5:]):
                    st.write(f"**{claim['classification']}** ({claim['confidence']:.2f}): {claim['claim']}")
        
        with col2:
            st.subheader("Claim Accuracy Distribution")
            if self.claims_data:
                df = pd.DataFrame(self.claims_data)
                fig_pie = px.pie(
                    df,
                    names="classification",
                    values="confidence",
                    title="Distribution of Claim Classifications"
                )
                st.plotly_chart(fig_pie)
                
                st.subheader("Summary Statistics")
                total_claims = len(df)
                if total_claims > 0:
                    true_claims = len(df[df['classification'] == 'true'])
                    false_claims = len(df[df['classification'] == 'false'])
                    uncertain_claims = len(df[df['classification'] == 'uncertain'])
                    
                    st.write(f"Total Claims Analyzed: {total_claims}")
                    st.write(f"True Claims: {true_claims} ({true_claims/total_claims*100:.1f}%)")
                    st.write(f"False Claims: {false_claims} ({false_claims/total_claims*100:.1f}%)")
                    st.write(f"Uncertain Claims: {uncertain_claims} ({uncertain_claims/total_claims*100:.1f}%)")

class CollaborativeFactChecking:
    def __init__(self):
        self.claims = []
        
    def add_claim(self, claim: str, evidence: str):
        self.claims.append({
            'claim': claim,
            'evidence': evidence,
            'timestamp': datetime.now(),
            'votes': {'true': 0, 'false': 0, 'uncertain': 0},
            'comments': []
        })
    
    def add_vote(self, claim_index: int, vote_type: str):
        if 0 <= claim_index < len(self.claims):
            self.claims[claim_index]['votes'][vote_type] += 1
    
    def add_comment(self, claim_index: int, comment: str):
        if 0 <= claim_index < len(self.claims):
            self.claims[claim_index]['comments'].append({
                'text': comment,
                'timestamp': datetime.now()
            })

def render_news_analysis(news_monitor, claim_extractor, misinfo_detector, dashboard):
    st.header("News Analysis")
    
    if st.button("Fetch Latest News"):
        with st.spinner("Fetching and analyzing news..."):
            news_items = news_monitor.get_recent_news()
            for item in news_items:
                claims = claim_extractor.extract_claims(item['content'])
                for claim in claims:
                    prediction, confidence = misinfo_detector.check_claim(
                        claim,
                        [item['content']]
                    )
                    dashboard.update_data([{
                        "claim": claim,
                        "classification": prediction,
                        "confidence": confidence,
                        "timestamp": item['timestamp'],
                        "source": item['source']
                    }])
    
    st.subheader("Search News")
    keywords = st.text_input("Enter keywords (comma-separated)").split(",")
    if st.button("Search") and keywords:
        with st.spinner("Searching news..."):
            relevant_news = news_monitor.search_news(keywords)
            st.write(f"Found {len(relevant_news)} relevant news items")
            for item in relevant_news:
                st.write(f"**{item['title']}**")
                st.write(item['content'])
                st.write(f"Source: {item['source']}")
                st.write("---")

def render_manual_check(claim_extractor, misinfo_detector, dashboard):
    st.header("Manual Claim Check")
    
    claim_text = st.text_area("Enter text to analyze for claims:")
    reference_text = st.text_area("Enter reference text (optional):")
    
    if st.button("Analyze") and claim_text:
        with st.spinner("Analyzing claims..."):
            claims = claim_extractor.extract_claims(claim_text)
            st.subheader("Extracted Claims")
            for claim in claims:
                prediction, confidence = misinfo_detector.check_claim(
                    claim,
                    [reference_text] if reference_text else [claim_text]
                )
                st.write(f"**Claim:** {claim}")
                st.write(f"**Analysis:** {prediction} (confidence: {confidence:.2f})")
                st.write("---")
                dashboard.update_data([{
                    "claim": claim,
                    "classification": prediction,
                    "confidence": confidence,
                    "timestamp": datetime.now()
                }])

def render_collaborative_checking(fact_checker):
    st.header("Collaborative Fact-Checking")
    
    st.subheader("Submit New Claim")
    new_claim = st.text_input("Enter claim:")
    evidence = st.text_area("Enter evidence or source:")
    if st.button("Submit") and new_claim and evidence:
        fact_checker.add_claim(new_claim, evidence)
        st.success("Claim submitted successfully!")
    
    st.subheader("Current Claims")
    for i, claim in enumerate(fact_checker.claims):
        st.write(f"**Claim:** {claim['claim']}")
        st.write(f"**Evidence:** {claim['evidence']}")
        
        cols = st.columns(3)
        if cols[0].button("True", key=f"true_{i}"):
            fact_checker.add_vote(i, 'true')
        if cols[1].button("False", key=f"false_{i}"):
            fact_checker.add_vote(i, 'false')
        if cols[2].button("Uncertain", key=f"uncertain_{i}"):
            fact_checker.add_vote(i, 'uncertain')
        
        st.write(f"Votes: True ({claim['votes']['true']}), "
                f"False ({claim['votes']['false']}), "
                f"Uncertain ({claim['votes']['uncertain']})")
        
        comment = st.text_input("Add a comment:", key=f"comment_{i}")
        if st.button("Submit Comment", key=f"submit_comment_{i}") and comment:
            fact_checker.add_comment(i, comment)
        
        if claim['comments']:
            st.write("**Comments:**")
            for comment in claim['comments']:
                st.write(f"- {comment['text']}")
        
        st.write("---")

def main():
    st.title("Misinformation Detection System")
    
    # Initialize components
    config = Config()
    claim_extractor = ClaimExtractor()
    misinfo_detector = MisinformationDetector()
    news_monitor = NewsMonitor(Config.NEWS_FEEDS)
    dashboard = VisualizationDashboard()
    fact_checker = CollaborativeFactChecking()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a function",
        ["News Analysis", "Manual Claim Check", "Collaborative Fact-Checking"]
    )
    
    # Render selected page
    if page == "News Analysis":
        render_news_analysis(news_monitor, claim_extractor, misinfo_detector, dashboard)
    elif page == "Manual Claim Check":
        render_manual_check(claim_extractor, misinfo_detector, dashboard)
    elif page == "Collaborative Fact-Checking":
        render_collaborative_checking(fact_checker)
    
    # Always show the dashboard at the bottom
    st.write("---")
    st.header("Dashboard")
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()