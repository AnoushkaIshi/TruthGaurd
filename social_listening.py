import requests
import os
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Fetch news articles based on a keyword
def fetch_tweets(keyword, count=10):
    params = {
        "q": keyword,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": count
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        news_data = response.json()

        if news_data.get('status') == 'ok':
            articles = [
                f"{article['title']} - {article['source']['name']}" for article in news_data.get('articles', [])
            ]
            return articles
        else:
            return [f"Failed to fetch news: {news_data.get('message')}"]

    except Exception as e:
        return [f"Unable to connect to NewsAPI: {str(e)}"]
