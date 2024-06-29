import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV nettoyé
file_path = '4cleaned_data.csv'
df = pd.read_csv(file_path)

# Convertir la colonne de date en format datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Créer une nouvelle colonne pour le mois-année
df['month_year'] = df['date'].dt.to_period('M')

# Calculer la moyenne des ratings par mois
monthly_avg_rating = df.groupby('month_year')['rating'].mean().reset_index()
monthly_avg_rating['month_year'] = monthly_avg_rating['month_year'].dt.to_timestamp()

# Compter le nombre de commentaires par mois
monthly_comments = df.groupby('month_year').size().reset_index(name='comment_count')
monthly_comments['month_year'] = monthly_comments['month_year'].dt.to_timestamp()

# Tracer les courbes
plt.figure(figsize=(12, 6))

# Courbe de la moyenne des ratings par mois
plt.plot(monthly_avg_rating['month_year'], monthly_avg_rating['rating'], marker='o', linestyle='-', label='Moyenne des Ratings par Mois')

# Courbe du nombre de commentaires par mois
plt.plot(monthly_comments['month_year'], monthly_comments['comment_count'], marker='x', linestyle='-', label='Nombre de Commentaires par Mois', color='orange')

plt.title('Évolution des Ratings et Nombre de Commentaires')
plt.xlabel('Temps')
plt.ylabel('Valeurs')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('evolution_ratings_comments.png')  # Enregistrer le graphique comme fichier image
plt.show()
