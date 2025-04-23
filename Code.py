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

#**Etape 1 : Analyse Technique**

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
    # Signal long si le prix est au-dessus de la SMA20, RSI pas trop élevé, MACD > Signal_Line
    df["Signal_Long"] = (
        (df["Close"] > df["SMA20"]) &
        (df["RSI"] < 70) &
        (df["MACD"] > df["Signal_Line"])
    )
    
    # Signal short si le prix est en dessous de la SMA20, RSI pas trop bas, MACD < Signal_Line
    df["Signal_Short"] = (
        (df["Close"] < df["SMA20"]) &
        (df["RSI"] > 30) &
        (df["MACD"] < df["Signal_Line"])
    )
    
    # Attribution des signaux
    df["Signal"] = np.select(
        [df["Signal_Long"], df["Signal_Short"]],
        ["long", "short"],
        default="hold"
    )
    
    return df


# **Etape 2 : Analyse des actualités, analyse fondamentale, et signaux**

import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY = "d012njpr01qv3oh2b3a0d012njpr01qv3oh2b3ag"

# 📥 Récupération des actualités via l'API Finnhub
def get_finnhub_news(ticker, from_date, to_date):
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Erreur API Finnhub: {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    news = []
    for article in data:
        title = article.get('headline', '')
        description = article.get('summary', '')
        if description:
            news.append({
                'title': title,
                'description': description,
                'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)),
                'url': article.get('url', '')
            })
    return pd.DataFrame(news)

# 🧠 Analyse de sentiment avec VADER
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["description"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

# 🗞️ Actualités des 7 derniers jours avec score de sentiment moyen
def get_recent_news(ticker, days=7):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    df = get_finnhub_news(ticker, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
    if not df.empty:
        df = analyze_sentiment(df)
    return df

# 🧮 Fondamentaux de l'entreprise
def is_fundamentally_solid(ticker, per_thresh=25, roe_thresh=0.10, de_thresh=1.5):
    try:
        info = yf.Ticker(ticker).info
        per = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        if per is None or roe is None or de is None:
            return None, None, None, "hold"
        signal = "buy" if (per < per_thresh and roe > roe_thresh and de < de_thresh * 100) else "sell"
        return per, roe, de, signal
    except Exception:
        return None, None, None, "hold"

#  Signal presse ajusté (inspiré MSPR)
def signal_presse(sentiment, seuil_pos=0.2, seuil_neg=-0.2):
    if sentiment is None:
        return "hold"
    if sentiment >= seuil_pos:
        return "buy"
    elif sentiment <= seuil_neg:
        return "sell"
    else:
        return "hold"



def tableau_synthetique(resultats, sentiments, per_thresh=25, roe_thresh=0.10, de_thresh=1.5, window=5):
    def map_to_emoji(signal, is_tech=False):
        if signal in ["buy", "long"] or (is_tech and signal == "long"):
            return "🟢"
        elif signal in ["sell", "short"] or (is_tech and signal == "short"):
            return "🔴"
        else:
            return "🟡"

    tableau = []
    for ticker, df in resultats.items():
        df_tail = df.tail(window)

        close_mean = df_tail["Close"].mean()
        sma20_mean = df_tail["SMA20"].mean()
        rsi_mean = df_tail["RSI"].mean()
        macd_mean = df_tail["MACD"].mean()

        df_tail = generate_signals(df_tail)
        signal_tech = df_tail["Signal"].mode()[0] if not df_tail["Signal"].mode().empty else "hold"

        avg_sentiment = sentiments.get(ticker)
        signal_presse = "buy" if avg_sentiment is not None and avg_sentiment > 0.05 else (
                         "sell" if avg_sentiment is not None and avg_sentiment < -0.05 else "hold")

        info = yf.Ticker(ticker).info
        per = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")

        signal_fondamental = "hold"
        if per is not None and roe is not None and de is not None:
            if per < per_thresh and roe > roe_thresh and de < de_thresh * 100:
                signal_fondamental = "buy"
            elif per > per_thresh or roe < roe_thresh or de > de_thresh * 100:
                signal_fondamental = "sell"

        emoji_tech = map_to_emoji(signal_tech, is_tech=True)
        emoji_presse = map_to_emoji(signal_presse)
        emoji_fond = map_to_emoji(signal_fondamental)
        signal_final = emoji_tech + emoji_presse + emoji_fond

        tableau.append({
            "Ticker": ticker,
            "Date": df_tail.index[-1],
            "Close (moy5j)": round(close_mean, 2),
            "SMA20 (moy5j)": round(sma20_mean, 2),
            "RSI (moy5j)": round(rsi_mean, 2),
            "MACD (moy5j)": round(macd_mean, 2),
            "Signal_Technique": signal_tech,
            "Sentiment_Moyen": round(avg_sentiment, 2) if avg_sentiment is not None else None,
            "Signal_Presse": signal_presse,
            "PER": per,
            "ROE": roe,
            "D/E": de,
            "Signal_Fondamental": signal_fondamental,
            "Signal_Final": signal_final
        })

    return pd.DataFrame(tableau)


#Voici la fonction pour faire apparaitre notre tableau synthétique
tickers = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "NVDA", "JPM", "DIS"] #<- ici on peut en mettre autant qu'on veut
sentiments = {}
resultats = {}

for ticker in tickers:
    news_df = get_recent_news(ticker)
    if not news_df.empty:
        sentiments[ticker] = news_df["sentiment_score"].mean()
    else:
        sentiments[ticker] = None

for ticker in tickers:
    try:
        data = get_price_data(ticker)  
        data = add_sma(data)  
        data = compute_rsi(data)  
        data = compute_macd(data)  
        data = generate_signals(data)  
        data = evaluate_signals(data) 
        data = generate_final_signal(data, ticker, sentiments)
        resultats[ticker] = data
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} : {e}")

# Résumé des résultats backtest
for ticker, df in resultats.items():
    df = backtest_signals(df)
    print(f"\n🔁 Résultats backtest pour {ticker}")
    resume_backtest(df)

# Tableau synthétique
tableau = tableau_synthetique(resultats, sentiments)
print(tableau)

# **Etape 3: Visualisation** 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualiser_strategie(df, ticker):
    """
    Visualisation de la courbe de prix avec les signaux d'achat et de vente.
    """
    plt.figure(figsize=(12, 6))
    
    # Tracer la courbe de prix
    plt.plot(df['Close'], label='Prix de clôture', color='blue', lw=2)
    
    # Tracer la SMA20
    plt.plot(df['SMA20'], label='SMA20', color='orange', linestyle='--', lw=2)
    
    # Tracer les points d'achat (flèches vertes)
    achat = df[df['Signal_Long'] == 1]
    plt.scatter(achat.index, achat['Close'], marker='^', color='green', label='Point d\'achat', zorder=5)
    
    # Tracer les points de vente (flèches rouges)
    vente = df[df['Signal_Short'] == 1]
    plt.scatter(vente.index, vente['Close'], marker='v', color='red', label='Point de vente', zorder=5)
    
    # Ajouter des titres et labels
    plt.title(f"Stratégie de trading pour {ticker}", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Prix', fontsize=12)
    plt.legend(loc='best')
    
    # Rotation des labels de l'axe x
    plt.xticks(rotation=45)
    
    # Afficher le graphique
    plt.tight_layout()
    plt.show()

def stats_strategie(df):
    """
    Affiche les statistiques de la stratégie, comme les retours cumulatifs, le profit, etc.
    """
    stats = {}
    
    # Retour cumulé de la stratégie
    df['Cumul_Return'] = df['Return_5d'].cumsum()
    stats['Retour Cumulé'] = df['Cumul_Return'].iloc[-1]
    
    # Calcul du nombre de trades gagnants et perdants
    trades_gagnants = len(df[df['Signal_Long'] == 1][df['Return_5d'] > 0])
    trades_perdants = len(df[df['Signal_Short'] == 1][df['Return_5d'] < 0])
    
    stats['Trades gagnants'] = trades_gagnants
    stats['Trades perdants'] = trades_perdants
    stats['Taux de gain'] = trades_gagnants / (trades_gagnants + trades_perdants) if trades_gagnants + trades_perdants > 0 else 0
    
    # Return des derniers trades
    stats['Dernier retour'] = df['Return_5d'].iloc[-1]
    
    return stats

def visualiser_tous_les_tickers(resultats):
    """
    Visualise pour tous les tickers dans 'resultats' la courbe de prix, les points d'achat/vente et les stats.
    """
    for ticker, df in resultats.items():
        print(f"\n📊 Visualisation et stats pour {ticker}")
        try:
            # Visualisation de la stratégie
            visualiser_strategie(df, ticker)
            
            # Affichage des stats de la stratégie
            stats = stats_strategie(df)
            for k, v in stats.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la visualisation de {ticker} : {e}")
            continue

# Appel de la fonction pour visualiser tous les tickers
visualiser_tous_les_tickers(resultats)
