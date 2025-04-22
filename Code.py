# ***Etapes préliminaires ***

# Installer les bibliothèques 
!pip install yfinance --quiet
!pip install vaderSentiment

# Importer les modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta

# **Etape 1 : Analyse Technique **

# Récupération des données de prix
def get_price_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

#Indicateurs techniques : SMA, RSI, MACD
def add_sma(df, short=20, long=50):
    df[f"SMA{short}"] = df["Close"].rolling(window=short).mean()
    df[f"SMA{long}"] = df["Close"].rolling(window=long).mean()
    return df

def compute_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return df

def compute_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

# Générer les signaux techniques
def generate_signals(df):
    df["Signal_Long"] = (
        (df["Close"] > df["SMA20"]) &
        (df["Close"].shift(1) <= df["SMA20"].shift(1)) &
        (df["RSI"] < 70) &
        (df["MACD"] > df["Signal_Line"])
    )
    df["Signal_Short"] = (
        (df["Close"] < df["SMA20"]) &
        (df["Close"].shift(1) >= df["SMA20"].shift(1)) &
        (df["RSI"] > 30) &
        (df["MACD"] < df["Signal_Line"])
    )
    df["Signal"] = np.select(
        [df["Signal_Long"], df["Signal_Short"]],
        ["buy", "short"],
        default="hold"
    )
    return df

# Évaluation du succès des signaux à 5 jours
def evaluate_signals(df):
    df["Price_Now"] = df["Close"]
    df["Price_5d_After"] = df["Close"].shift(-5)
    df["Return_5d"] = ((df["Price_5d_After"] - df["Price_Now"]) / df["Price_Now"]) * 100

    def is_success(row):
        if row["Signal"] == "buy":
            return row["Return_5d"] > 0
        elif row["Signal"] == "short":
            return row["Return_5d"] < 0
        return np.nan

    df["Success"] = df.apply(is_success, axis=1)
    return df

# **Etape 2 : Analyse des actualités**

API_KEY = "d012njpr01qv3oh2b3a0d012njpr01qv3oh2b3ag"

def get_finnhub_news(ticker, from_date, to_date):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    news = []
    for article in data:
        news.append({
            'title': article.get('headline', ''),
            'description': article.get('summary', ''),
            'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)),
            'url': article.get('url', '')
        })
    return pd.DataFrame(news)

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["description"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

def get_recent_news(ticker, days=7):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    df = get_finnhub_news(ticker, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
    if not df.empty:
        df = analyze_sentiment(df)
    return df

def is_signal_favorable_with_news(signal, ticker):
    news_df = get_recent_news(ticker)
    if news_df.empty:
        print("❌ Aucune actualité → Signal rejeté.")
        return False
    avg_sentiment = news_df["sentiment_score"].mean()
    print(f"📊 Sentiment moyen sur 7 jours : {avg_sentiment:.2f}")
    return avg_sentiment >= 0.05

def generate_final_signal(df, ticker):
    df = generate_signals(df)
    df["Final_Signal"] = df.apply(
        lambda row: row["Signal"] if row["Signal"] != "hold" and is_signal_favorable_with_news(row["Signal"], ticker) else "hold",
        axis=1
    )
    return df

# **Etape 3 : Pipeline complet**

# 🎯 Exemple avec Tesla
ticker = "TSLA"
data = get_price_data(ticker)
data = add_sma(data)
data = compute_rsi(data)
data = compute_macd(data)
data = generate_signals(data)
data = evaluate_signals(data)
final_data = generate_final_signal(data, ticker)

# 🔍 Résultats
print(final_data[["Close", "Signal", "Final_Signal", "Return_5d", "Success"]].tail(10))

# 📊 Taux de réussite des signaux
signals_only = final_data[final_data["Signal"] != "hold"]
success_rate = signals_only["Success"].mean()
print(f"\n✅ Taux de signaux gagnants : {round(success_rate * 100, 2)} %")
print("📈 Rendement moyen par type de signal :")
print(signals_only.groupby("Signal")["Return_5d"].mean())
