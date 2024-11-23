import sys
import os

sys.path.append('//application/utils')
from tools import *

import solara

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

#Stopwords dir
stopwords_relative_path = '../../data/stopwords'
stopwords_dir = os.path.abspath(os.path.join(current_dir, stopwords_relative_path))


import nltk
nltk.download('stopwords', download_dir=stopwords_dir)
# Construct the path to the French stopwords file
stopwords_french_path = os.path.join(stopwords_dir, 'corpora', 'stopwords', 'french')

# Load French stopwords manually
with open(stopwords_french_path, 'r', encoding='utf-8') as file:
    stopwords_french = [line.strip() for line in file]

#from nltk.corpus import stopwords



######## SHOWS
relative_path_shows_items = "../../data/shows-23-24-items.csv"
absolute_path_shows_items = os.path.abspath(os.path.join(current_dir, relative_path_shows_items))

relative_path_shows_users = "../../data/shows-23-24-users.csv"
absolute_path_shows_users = os.path.abspath(os.path.join(current_dir, relative_path_shows_users))

relative_path_shows_purchases = "../../data/shows-23-24-purchase.csv"
absolute_path_shows_purchases = os.path.abspath(os.path.join(current_dir, relative_path_shows_purchases))

relative_path_shows_users_items_views = "../../data/shows-23-24-users-items-views.csv"
absolute_path_shows_users_items_views = os.path.abspath(os.path.join(current_dir, relative_path_shows_users_items_views))

######## MOVIES
relative_path_movies_items = "../../data/movies_clean.csv"
absolute_path_movies_items = os.path.abspath(os.path.join(current_dir, relative_path_movies_items))

print("absolute_path_movies_items==>",absolute_path_movies_items)

relative_path_movies_users = "../../data/movies_users.csv"
absolute_path_movies_users = os.path.abspath(os.path.join(current_dir, relative_path_movies_users))

relative_path_movies_users_purchases = "../../data/movies_users_purchases.csv"
absolute_path_movies_users_purchases = os.path.abspath(os.path.join(current_dir, relative_path_movies_users_purchases))

relative_path_movies_users_page_views = "../../data/movies_users_page_views.csv"
absolute_path_movies_users_page_views = os.path.abspath(os.path.join(current_dir, relative_path_movies_users_page_views))

relative_path_movies_users_ratings = "../../data/movies_ratings.csv"
absolute_path_movies_users_ratings = os.path.abspath(os.path.join(current_dir, relative_path_movies_users_ratings))

relative_path_movies_users_tags = "../../data/movies_tags.csv"
absolute_path_movies_users_tags = os.path.abspath(os.path.join(current_dir, relative_path_movies_users_tags))


def display_data(df):
    solara.DataFrame(df, items_per_page=5)

DATA_WORK = 'movies'
#DATA_WORK = 'shows'

if DATA_WORK == 'shows':
    def get_data(product_id, count):
        #Dataframe of items
        data = pd.read_csv(absolute_path_shows_items)
        data = data[['work_id', 'title', 'description', 'genre_1', 'auteur', 'url']]

        if product_id is not None:
            if (data['work_id'] == product_id).any():
                data = data[data['work_id'] == product_id]
            else:
                data_message = {
                    'message_error': 'This product ID does not exist',
                }
                data = pd.DataFrame(data)

        if count is not None:
            data = data.iloc[:count]

        return data

    def get_data_users():
        data_users = pd.read_csv(absolute_path_shows_users)
        data_users['firstlastname_user'] = data_users['firstname_user'] + ' ' + data_users['lastname_user']
        return data_users

    def get_data_purchase():
        data_purchase = pd.read_csv(absolute_path_shows_purchases)
        return data_purchase
    def get_data_views():
        data_views = pd.read_csv(absolute_path_shows_users_items_views)
        return data_views

    def get_work_default():
        return 'Lohengrin'

elif DATA_WORK == 'movies':

    def get_data(product_id, count):
        # Dataframe of items
        data = pd.read_csv(absolute_path_movies_items)
        data = data[['work_id', 'title', 'description', 'genre_1', 'auteur', 'year', 'url']]
        data['work_id'] = data['work_id'].astype(int)

        if product_id is not None:
            if (data['work_id'] == product_id).any():
                data = data[data['work_id'] == product_id]
            else:
                data_message = {
                    'message_error': 'This product ID does not exist',
                }
                data = pd.DataFrame(data)

        if count is not None:
            data = data.iloc[:count]

        return data

    def get_data_users():
        data_users = pd.read_csv(absolute_path_movies_users)
        data_users['user_firstlastname'] = data_users['user_firstname'] + ' ' + data_users['user_lastname']
        return data_users

    def get_data_purchase():
        data_purchase = pd.read_csv(absolute_path_movies_users_purchases)
        return data_purchase

    def get_data_views():
        data_views = pd.read_csv(absolute_path_movies_users_page_views)
        return data_views

    def get_data_users_ratings():
        data_users_ratings = pd.read_csv(absolute_path_movies_users_ratings)
        return data_users_ratings

    def get_data_users_tags():
        data_users_tags = pd.read_csv(absolute_path_movies_users_tags)
        return data_users_tags

    def get_work_default():
        #return 'Die Hard'
        #return 'Jurassic Park'
        #return 'Braveheart'
        return 'Toy Story'


def get_data_similarities(data):
    # Création du bags of Words et de la répétition des mots sur certaines caractéristiques des spectacles
    # If no description is given in the dataset, we cannot recommand it
    data = data.dropna(subset=['description'], how='any')

    # Set the display options to show the full content
    pd.set_option('display.max_colwidth', None)

    # To improve model, add bag_of_words with auteur genre or venue repeating to give more weight
    data['genre_improved_multi'] = data['genre_1'].apply(lambda x: repeat_word(x, 3))
    data['genre_improved_multi'] = data['genre_improved_multi'].apply(lambda x: ' '.join(map(str, x)))
    data.insert(4, 'genre_improved', data['genre_improved_multi'])

    # Create a bag of words composed with different features
    data['bag_of_words'] = data[data.columns[1:6]].apply(lambda x: ' '.join(x), axis=1)
    # Clean HTML of bag of words
    data['bag_of_words'] = data['bag_of_words'].apply(lambda x: cleanHTML(x))

    return data


def get_data_ratings(data_purchase):

    if DATA_WORK == 'shows':
        data_ratings = pd.DataFrame(data_purchase)
        data_ratings['rating'] = data_ratings['total_purchases'].apply(rating_to_show)

    elif DATA_WORK == 'movies':
        data_ratings = pd.DataFrame(data_purchase)
        data_ratings['rating'] = data_ratings['total_purchases'].apply(rating_to_allpurchases_of_movies)

    print(df_info(data_ratings,'data_ratings'))

    return data_ratings


def get_data_item_score(data, data_ratings):
    data_items = pd.DataFrame()
    data_items['rating_count'] = data_ratings['work_id'].value_counts()
    data_items['rating_average'] = data_ratings.groupby('work_id')['rating'].mean()
    data_items_merge = data_items.merge(data, on='work_id', how='left')

    # Calculate the number of votes garnered by the 80th percentile show
    m = data_items_merge['rating_count'].quantile(0.80)
    # print(m)
    C = data_items_merge['rating_average'].mean()
    # print(C)

    # Compute the score using the weighted_rating function defined above
    data_items_merge['score'] = data_items_merge.apply(lambda data_items_merge: weighted_rating(data_items_merge, m, C),
                                                   axis=1)
    data_items_merge = data_items_merge.sort_values('score', ascending=False)
    data_items_merge = data_items_merge.reset_index()

    return data_items_merge


def get_data_with_score(data, data_items_merge):
    data_with_score = data.merge(data_items_merge, on='work_id', how='left')

    display_data(data_with_score)

    data_with_score = data_with_score[['work_id', 'title_x', 'description_x', 'genre_x', 'genre_improved_x',
                                       'auteur_x', 'url_x', 'genre_improved_multi_x', 'bag_of_words_x', 'score']]
    data_with_score.rename(columns={'title_x': 'title', 'description_x': 'description', 'genre_x': 'genre_1',
                                    'genre_improved_x': 'genre_improved', 'auteur_x': 'auteur',
                                    'url_x': 'url', 'genre_improved_multi_x': 'genre_improved_multi',
                                    'bag_of_words_x': 'bag_of_words'}, inplace=True)
    return data_with_score

@solara.component
def exploreData():

    data = get_data(product_id=None, count=None)
    display_data(data)

    solara.Markdown(
        f"""
         ## Les oeuvres - spectacles ou films - création d'un ensemble de mots pour caractériser les oeuvres
     """
    )

    data_similarities = get_data_similarities(data)
    # we'll try to find similarities based on the Description Words of a show
    data_similarities = data_similarities['bag_of_words']

    X = np.array(data_similarities)

    data_for_display_only = pd.DataFrame(data_similarities)
    display_data(data_for_display_only)

    solara.Markdown(
        f"""
        ## Les ventes des oeuvres spectacles ou films (fictif)
    """
    )

    data_purchase = get_data_purchase()
    display_data(data_purchase)

    solara.Markdown(
        f"""
        ## Les oeuvres spectacles ou films vus sur le site (fictif)
    """
    )

    data_views = get_data_views()
    display_data(data_views)

    solara.Markdown(
        f"""
        ## Les utilisateurs avec données démographiques et géographiques du site (fictif)
    """
    )

    data_users = get_data_users()
    display_data(data_users)


    solara.Markdown(
        f"""
        ## Les scores calculés en fonction du rating qui est basé sur les ventes des oeuvres spectacles ou films (fictif)
    """
    )

    # Application du Rating sur les spectacles (fictif)
    data_ratings = get_data_ratings(data_purchase)

    data_items_merge = get_data_item_score(data, data_ratings)
    display_data(data_items_merge)

    #data_with_score = get_data_with_score(data, data_items_merge)
    #display_data(data_with_score)

    #data_for_display_only = pd.DataFrame(data_with_score['score'])
    #display_data(data_for_display_only)
