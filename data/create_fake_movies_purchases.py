import pandas as pd
import random

# Load the movies and ratings data from the two CSV files
movies = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_clean.csv')
ratings = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_ratings.csv')

# Fonction pour déterminer le nombre d'achats basé sur la note

def is_eligible(rating):
    return rating > 3  # Film éligible si la note est supérieure à 3

# Filtrer les évaluations pour obtenir uniquement les films éligibles
ratings['is_eligible'] = ratings['rating'].apply(is_eligible)
eligible_ratings = ratings[ratings['is_eligible']]

# Fonction pour générer des achats aléatoires
def generate_random_purchases(user_id, group):
    num_purchases = random.randint(1, 10)  # Nombre d'achats aléatoire entre 1 et 10
    purchases = group.sample(n=min(num_purchases, len(group)), replace=False)  # Échantillonner sans remplacement
    purchases['total_purchases'] = 1  # Fixer le nombre d'achats à 1 pour chaque film sélectionné
    return purchases

# Appliquer la fonction de génération d'achats aléatoires pour chaque utilisateur
limited_sales = eligible_ratings.groupby('user_id').apply(generate_random_purchases, group=eligible_ratings).reset_index(drop=True)

# Trier le DataFrame par user_id
limited_sales.sort_values(by='user_id', inplace=True)

# Sélectionner uniquement les colonnes pertinentes
purchases = limited_sales[['user_id', 'work_id', 'total_purchases']]

#Filter based on clean list of movies
df_filtered = purchases[purchases['work_id'].isin(movies['work_id'])]

# Save the DataFrame as a CSV file
df_filtered.to_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_users_purchases.csv', index=False)
