import pandas as pd
from textblob import TextBlob

# Charger les données depuis le fichier CSV
df = pd.read_csv("alibaba_reviews.csv")

# Vérifier les colonnes et afficher les premiers avis pour le débogage
print("DataFrame columns:", df.columns)
print(df.head())

# Convertir la colonne 'date' en format datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print("Date conversion successful.")

# Ajouter une colonne 'satisfied' si 'rating' existe
df['satisfied'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Supprimer les doublons
df = df.drop_duplicates(subset=['title', 'body'], keep='first')
print("Duplicates removed, if any.")

# Réorganiser les catégories
def categorize_review(body):
    body = body.lower()
    if "livre" in body or "book" in body:
        return "Books"
    elif "aliment" in body or "food" in body:
        return "Food"
    elif "vêtement" in body or "clothes" in body:
        return "Clothing"
    else:
        return "Uncategorized"

df['category'] = df['body'].apply(categorize_review)

# Analyser les sentiments en tenant compte de la note
def analyze_sentiment(row):
    body = row['body']
    rating = row['rating']
    polarity = TextBlob(body).sentiment.polarity
    if rating >= 4:
        return "Positive"
    elif rating == 3 and polarity > 0:
        return "Positive"
    elif polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df.apply(analyze_sentiment, axis=1)

# Distribution des avis par catégorie de produit
category_distribution = df['category'].value_counts()
print("Category distribution:")
print(category_distribution)

# Distribution des avis par sentiment
sentiment_distribution = df['sentiment'].value_counts()
print("Sentiment distribution:")
print(sentiment_distribution)

# Sauvegarder le DataFrame nettoyé en CSV
df.to_csv("new_alibaba_reviews.csv", index=False)
print("Data saved to bis_cleaned_trustpilot_reviews.csv")

# Afficher le DataFrame en format tableau
print(df.to_string())
