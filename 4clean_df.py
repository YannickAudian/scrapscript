import pandas as pd

# Charger le fichier CSV depuis le chemin local
file_path = 'lv_categorized_cleaned_trustpilot_reviews.csv'
df = pd.read_csv(file_path)

# Convertir la colonne de date en format datetime
df['date'] = pd.to_datetime(df['date'])

# Convertir les dates au format DD-MM-YYYY
df['date'] = df['date'].dt.strftime('%d-%m-%Y')

# Supprimer la dernière colonne (doublon de la colonne 2)
df = df.drop(columns=['text'])

# Enregistrer le dataframe nettoyé dans un nouveau fichier CSV
cleaned_file_path = 'test_clean_df.csv'
df.to_csv(cleaned_file_path, index=False)

print("Le dataframe a été nettoyé et sauvegardé avec succès.")
