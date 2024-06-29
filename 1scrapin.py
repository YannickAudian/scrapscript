import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from textblob import TextBlob

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_trustpilot_reviews(base_url, num_pages=100):
    reviews_list = []
    review_id = 1
    for page in range(1, num_pages + 1):
        page_url = f"{base_url}?page={page}"
        response = requests.get(page_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")

        if not script_tag:
            print(f"No JSON data found on page {page}")
            continue
        
        json_data = json.loads(script_tag.string)
        
        reviews = json_data['props']['pageProps']['reviews']

        for review in reviews:
            try:
                title = review['title'] if 'title' in review else "No Title"
                rating = review['rating'] if 'rating' in review else 0
                date = review['dates']['experiencedDate'] if 'dates' in review and 'experiencedDate' in review['dates'] else "No Date"
                body = review['text'] if 'text' in review else "No Content"
                polarity = TextBlob(body).sentiment.polarity

                # Déterminer le texte du sentiment
                if polarity > 0:
                    sentiment = "Positive"
                elif polarity == 0:
                    sentiment = "Neutral"
                else:
                    sentiment = "Negative"

                # Catégoriser les produits (à ajuster selon les besoins)
                category = "Uncategorized"
                if "livre" in body.lower() or "book" in body.lower():
                    category = "Books"
                elif "aliment" in body.lower() or "food" in body.lower():
                    category = "Food"
                elif "vêtement" in body.lower() or "clothes" in body.lower():
                    category = "Clothing"
                # Ajouter d'autres catégories selon les besoins

                reviews_list.append({
                    "id": review_id,
                    "title": title,
                    "rating": rating,
                    "date": date,
                    "body": body,
                    "sentiment": sentiment,
                    "category": category
                })
                review_id += 1
            except KeyError as e:
                print(f"Error parsing review: {e}")
        
        time.sleep(1)
    
    if not reviews_list:
        print("No reviews collected.")
        return pd.DataFrame(columns=["id", "title", "rating", "date", "body", "sentiment", "category"])
    
    return pd.DataFrame(reviews_list)

# Nouvelle URL de la page Trustpilot
base_url = "https://fr.trustpilot.com/review/www.cdiscount.com"

# Récupérer les avis
df = get_trustpilot_reviews(base_url, num_pages=160)

# Vérifier les colonnes et afficher les premiers avis pour le débogage
print("DataFrame columns:", df.columns)
print(df.head())

# Convertir la colonne 'date' en format datetime si elle existe
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("Date conversion successful.")
else:
    print("No 'date' column found in the data.")

# Ajouter une colonne 'satisfied' si 'rating' existe
if 'rating' in df.columns:
    df['satisfied'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
else:
    print("No 'rating' column found in the data.")

# Supprimer les doublons
df = df.drop_duplicates(subset=['title', 'body'], keep='first')
print("Duplicates removed, if any.")

# Sauvegarder le DataFrame en CSV
df.to_csv("1lbc.csv", index=False)
print("Data saved to trustpilot_reviews.csv")

# Afficher le DataFrame en format tableau
print(df.to_string())
