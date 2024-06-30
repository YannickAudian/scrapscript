import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from textblob import TextBlob
import re
from collections import Counter
import matplotlib.pyplot as plt

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
BASE_URL = "https://fr.trustpilot.com/review/www.cdiscount.com"
NUM_PAGES = 160
STOP_WORDS = set([
    'le', 'la', 'les', 'un', 'une', 'de', 'des', 'et', 'à', 'en', 'du', 'pour', 
    'par', 'avec', 'plus', 'moins', 'est', 'sont', 'ce', 'cette', 'ces', 'sur', 
    'dans', 'se', 'au', 'aux', 'que', 'qui', 'quoi', 'où', 'quand', 'comment', 
    'ne', 'pas', 'n\'', 'y', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 
    'je', 'tu', 'me', 'te', 'se', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 
    'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'donc', 'ainsi', 
    'bien', 'très', 'comme', 'mais', 'ou', 'encore', 'très', 'bien', 'fait', 'chez'
])

CATEGORIES = {
    'Textile': ['jeans', 'levis', 'vêtements', 'tshirt', 'pantalon', 'robe', 'chemise', 'pull', 'jacket', 'manteau', 'jupe', 'short'],
    'Electronics': ['tv', 'laptop', 'phone', 'tablet', 'électroménager', 'ordinateur', 'caméra', 'téléphone', 'écouteurs', 'haut-parleur', 'imprimante'],
    'Food': ['alimentaire', 'nourriture', 'boisson', 'chocolat', 'biscuits', 'café', 'thé', 'pâtes', 'riz', 'gâteau', 'bonbon', 'sauce', 'huile', 'épices'],
    'Customer Service': ['service', 'clients', 'commande', 'problème', 'remboursement', 'réclamation', 'satisfaction', 'support', 'contact'],
    'Shipping': ['livraison', 'colis', 'expédition', 'livré', 'retour', 'délai', 'envoi', 'poste', 'transporter', 'acheminement', 'livreur'],
    'Marketplace': ['amazon', 'prime', 'site', 'vente', 'produit', 'article', 'marque', 'magasin', 'boutique', 'commande'],
}

# Functions
def get_trustpilot_reviews(base_url, num_pages):
    reviews_list = []
    for page in range(1, num_pages + 1):
        page_url = f"{base_url}?page={page}"
        response = requests.get(page_url, headers=HEADERS)
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
                title = review.get('title', "No Title")
                rating = review.get('rating', 0)
                date = review.get('dates', {}).get('experiencedDate', "No Date")
                body = review.get('text', "No Content")
                polarity = TextBlob(body).sentiment.polarity
                sentiment = "Positive" if polarity > 0 else "Neutral" if polarity == 0 else "Negative"
                category = categorize_comment(body)

                reviews_list.append({
                    "title": title,
                    "rating": rating,
                    "date": date,
                    "body": body,
                    "sentiment": sentiment,
                    "category": category
                })
            except KeyError as e:
                print(f"Error parsing review: {e}")
        
        time.sleep(1)
    
    return pd.DataFrame(reviews_list)

def categorize_comment(comment):
    comment = comment.lower()
    for category, keywords in CATEGORIES.items():
        if any(keyword in comment for keyword in keywords):
            return category
    return 'Uncategorized'

def get_top_keywords(text, stop_words, n=100):
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(n)

def analyze_sentiment(row):
    body = row['body']
    rating = row['rating']
    polarity = TextBlob(body).sentiment.polarity
    if rating >= 4 or (rating == 3 and polarity > 0):
        return "Positive"
    elif polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Main Process
# Step 1: Scrape Trustpilot reviews
df = get_trustpilot_reviews(BASE_URL, NUM_PAGES)

# Step 2: Data Cleaning and Processing
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone information
df['month_year'] = df['date'].dt.to_period('M')
df['satisfied'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
df = df.drop_duplicates(subset=['title', 'body'], keep='first')

# Step 3: Sentiment Analysis
df['sentiment'] = df.apply(analyze_sentiment, axis=1)

# Step 4: Keyword Extraction
all_text = ' '.join(df['title'].astype(str) + ' ' + df['body'].astype(str))
top_keywords = get_top_keywords(all_text, STOP_WORDS)

# Save the DataFrame to CSV
df.to_csv("cdiscount_cleaned_reviews.csv", index=False)
print("Data saved to cdiscount_cleaned_reviews.csv")

# Step 5: Visualization
monthly_avg_rating = df.groupby('month_year')['rating'].mean().reset_index()
monthly_avg_rating['month_year'] = monthly_avg_rating['month_year'].dt.to_timestamp()
monthly_comments = df.groupby('month_year').size().reset_index(name='comment_count')
monthly_comments['month_year'] = monthly_comments['month_year'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg_rating['month_year'], monthly_avg_rating['rating'], marker='o', linestyle='-', label='Average Rating per Month')
plt.plot(monthly_comments['month_year'], monthly_comments['comment_count'], marker='x', linestyle='-', label='Number of Comments per Month', color='orange')
plt.title('Ratings and Comments Evolution')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('viz.png')
plt.show()
