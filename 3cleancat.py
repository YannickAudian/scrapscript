import pandas as pd
import re
from collections import Counter

# Stopwords personnalisés
stop_words = set([
    'le', 'la', 'les', 'un', 'une', 'de', 'des', 'et', 'à', 'en', 'du', 'pour', 
    'par', 'avec', 'plus', 'moins', 'est', 'sont', 'ce', 'cette', 'ces', 'sur', 
    'dans', 'se', 'au', 'aux', 'que', 'qui', 'quoi', 'où', 'quand', 'comment', 
    'ne', 'pas', 'n\'', 'y', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 
    'je', 'tu', 'me', 'te', 'se', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 
    'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'donc', 'ainsi', 
    'bien', 'très', 'comme', 'mais', 'ou', 'encore', 'très', 'bien', 'fait', 'chez'
])

# Charger les données
file_path = 'new_alibaba_reviews.csv'
df = pd.read_csv(file_path)

# Combiner les titres et les commentaires
df['text'] = df['title'].astype(str) + ' ' + df['body'].astype(str)

# Fonction pour extraire les mots-clés les plus fréquents
def get_top_keywords(text, stop_words, n=100):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace characters with a space
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(n)

# Appliquer la fonction pour extraire les mots-clés
all_text = ' '.join(df['text'])
top_keywords = get_top_keywords(all_text, stop_words)

# Afficher les mots-clés les plus fréquents
top_keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'frequency'])
print(top_keywords_df.head(50))

# Associer les mots-clés aux catégories
categories = {
    'Textile': ['jeans', 'levis', 'vêtements', 'tshirt', 'pantalon', 'robe', 'chemise', 'pull', 'jacket', 'manteau', 'jupe', 'short'],
    'Electronics': ['tv', 'laptop', 'phone', 'tablet', 'électroménager', 'ordinateur', 'caméra', 'téléphone', 'écouteurs', 'haut-parleur', 'imprimante'],
    'Food': ['alimentaire', 'nourriture', 'boisson', 'chocolat', 'biscuits', 'café', 'thé', 'pâtes', 'riz', 'gâteau', 'bonbon', 'sauce', 'huile', 'épices'],
    'Customer Service': ['service', 'clients', 'commande', 'problème', 'remboursement', 'réclamation', 'satisfaction', 'support', 'contact'],
    'Shipping': ['livraison', 'colis', 'expédition', 'livré', 'retour', 'délai', 'envoi', 'poste', 'transporter', 'acheminement', 'livreur'],
    'Marketplace': ['amazon', 'prime', 'site', 'vente', 'produit', 'article', 'marque', 'magasin', 'boutique', 'commande'],
    # Ajouter d'autres catégories et mots-clés selon vos besoins
}

# Fonction pour catégoriser les commentaires
def categorize_comment(comment, categories):
    comment = comment.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in comment:
                return category
    return 'Uncategorized'

# Appliquer la fonction sur les commentaires et les titres en évitant les doublons
def assign_category(row, categories):
    combined_text = str(row['body']) + ' ' + str(row['title'])
    return categorize_comment(combined_text, categories)

df['category'] = df.apply(lambda row: assign_category(row, categories), axis=1)

# Fonction pour détecter le sentiment (simpliste)
def detect_sentiment(text):
    negative_words = ['fuir', 'problème', 'jamais', 'mauvais', 'horrible', 'dégueulasse', 'retard', 'erreur', 'nul', 'déçu', 'pas', 'n']
    positive_words = ['bien', 'excellent', 'parfait', 'satisfait', 'bonne', 'top', 'super', 'génial', 'rapide']
    
    text = text.lower()
    neg_count = sum(1 for word in negative_words if word in text)
    pos_count = sum(1 for word in positive_words if word in text)
    
    if neg_count > pos_count:
        return 'Negative'
    elif pos_count > neg_count:
        return 'Positive'
    else:
        return 'Neutral'

df['sentiment'] = df['text'].apply(detect_sentiment)

# Sauvegarder le DataFrame avec les nouvelles catégories
output_path = '3alibaba.csv'
df.to_csv(output_path, index=False)

# Afficher un aperçu des données pour confirmation
print(df.head())
