import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), 'application/utils'))
from tools import *
from exploreData import *

import solara

import matplotlib.pyplot as plt
import numpy as np
#from sentence_transformers import SentenceTransformer
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.

#Get default work title
work_default_title = get_work_default()

#select_default_shows = solara.reactive("Lohengrin")
select_default_shows = solara.reactive(work_default_title)

@solara.component
def Page():

    solara.Style(
        """
        .v-application--wrap {
            padding:1em; !important
        }
        """
    )

    solara.Markdown(
        f"""
        # Machine Learning - Recommandation d'oeuvres - spectacles ou films
        # 1 > Exploration des données
    """
    )

    solara.Markdown(
        f"""
        ## Les spectacles (saison 23-24) ou films (année jusqu'à 1996)
    """
    )

    # Define nltk stopwords in french
    stopwords_french = stopwords.words('french')

    # Dataframe of items
    data = get_data(product_id=None, count=None)

    solara.DataFrame(data, items_per_page=5)

    solara.Markdown(
        f"""
         ## Les oeuvres - création d'un ensemble de mots pour caractériser les oeuvres
     """
    )
    # Création du bags of Words et de la répétition des mots sur certaines caractéristiques des oeuvres
    # If no description is given in the dataset, we cannot recommand it
    data = data.dropna(subset=['description'], how='any')

    # Set the display options to show the full content
    pd.set_option('display.max_colwidth', None)

    # To improve model, add bag_of_words with auteur genre or venue repeating to give more weight
    data['genre_improved_multi'] = data['genre_1'].apply(lambda x: repeat_word(x, 2))
    data['genre_improved_multi'] = data['genre_improved_multi'].apply(lambda x: ' '.join(map(str, x)))
    data.insert(4, 'genre_improved', data['genre_improved_multi'])

    # Create a bag of words composed with different features
    data['bag_of_words'] = data[data.columns[1:6]].apply(lambda x: ' '.join(x), axis=1)
    # Clean HTML of bag of words
    data['bag_of_words'] = data['bag_of_words'].apply(lambda x: cleanHTML(x))

    # we'll try to find similarities based on the Description Words of a work
    data_similarities = data['bag_of_words']

    X = np.array(data_similarities)

    data_for_display_only = pd.DataFrame(data['bag_of_words'])
    solara.DataFrame(data_for_display_only, items_per_page=5)

    solara.Markdown(
        f"""
        ## Les achats des oeuvres par utilisateur (fictif)
    """
    )
    data_purchase = get_data_purchase()
    solara.DataFrame(data_purchase, items_per_page=5)

    solara.Markdown(
        f"""
        ## Les oeuvres consultées sur le site web (fictif)
    """
    )

    data_views = get_data_views()
    solara.DataFrame(data_views, items_per_page=5)

    solara.Markdown(
        f"""
        ## Les utilisateurs avec données démographiques et géographiques du site web (fictif)
    """
    )

    data_users = get_data_users()
    solara.DataFrame(data_users, items_per_page=5)

    solara.Markdown(
        f"""
        ## Les scores calculés en fonction du rating qui est basé sur les ventes des oeuvres (fictif)
    """
    )

    # Application du Rating sur les oeuvres (fictif)
    data_ratings = pd.DataFrame(data_purchase)
    data_ratings_sorted = data_ratings.sort_values(by='work_id')

    data_ratings_2 = pd.DataFrame(data_purchase)
    data_ratings_2_sorted = data_ratings_2.sort_values(by='work_id')

    #data_ratings['rating'] = data_ratings['total_purchases'].apply(rating_to_show)
    data_ratings_2_sorted['rating'] = data_ratings_sorted['total_purchases'].apply(rating_to_movie)

    print(df_info(data_ratings_sorted, 'data_ratings_sorted'))
    print(df_info(data_ratings_2_sorted, 'data_ratings_2_sorted'))


    data_items = pd.DataFrame()
    #data_items['work_id'] = data_ratings['work_id']
    data_items = data_ratings_sorted.groupby('work_id').size().reset_index(name='rating_count')
    #data_items['rating_count'] = data_ratings['work_id'].value_counts()
    data_items['rating_average'] = data_ratings_2_sorted.groupby('work_id')['rating'].transform('mean')

    print(df_info(data_items,'data_items'))
    #print(data_items.info())
    #print(data.info())
    #print(data_items['work_id'].dtype)

    data_items_merge = data_items.merge(data, on='work_id', how='left')

    data_items_merge['rating_average'] = data_items_merge['rating_count'].apply(rating_to_allpurchases_of_movies)

    print(df_info(data_items_merge, 'data_items_merge'))

    # Calculate the number of votes garnered by the 80th percentile show
    m = data_items_merge['rating_count'].quantile(0.80)
    # print(m)
    C = data_items_merge['rating_average'].mean()
    # print(C)

    # Compute the score using the weighted_rating function defined above
    data_items_merge['score'] = data_items_merge.apply(lambda data_items_merge: weighted_rating(data_items_merge, m, C),
                                                       axis=1)

    data_items_merge = data_items_merge.sort_values('work_id', ascending=True)
    data_items_merge = data_items_merge.reset_index()

    print(df_info(data, 'data'))
    #print(data_items_merge.head())
    #print(data_items_merge.info())

    solara.DataFrame(data_items_merge, items_per_page=5)

    data_with_score = data.merge(data_items_merge, on='work_id', how='left')

    data_with_score = data_with_score[['work_id', 'title_x', 'description_x', 'genre_1_x', 'genre_improved_x',
                                       'auteur_x', 'url_x', 'genre_improved_multi_x', 'bag_of_words_x', 'score']]
    data_with_score.rename(columns={'title_x': 'title', 'description_x': 'description', 'genre_1_x': 'genre_1',
                                    'genre_improved_x': 'genre_improved', 'auteur_x': 'auteur',
                                    'url_x': 'url', 'genre_improved_multi_x': 'genre_improved_multi',
                                    'bag_of_words_x': 'bag_of_words'}, inplace=True)

    data_with_score = data_with_score.sort_values('score', ascending=False)

    print(df_info(data_with_score, 'data_with_score'))

    solara.DataFrame(data_with_score, items_per_page=5)

    #data_for_display_only = pd.DataFrame(data_with_score['score'])
    #solara.DataFrame(data_for_display_only, items_per_page=5)

    ############################################################################
    ## Vectorisation des mots avec 2 méthodes CountVectorizer ou TfIdfVectorizer
    ############################################################################

    # Import CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # Define a CV Vectorizer Object. Remove all english stopwords
    cv = CountVectorizer(stop_words=stopwords_french)

    # Construct the required CV matrix by applying the fit_transform method on the overview feature
    cv_matrix = cv.fit_transform(data_similarities)

    # Output the shape of tfidf_matrix
    # cv_matrix.shape

    # Import TfIdfVectorizer from the scikit-learn library
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words=stopwords_french)

    # Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(data_similarities)

    # Output the shape of tfidf_matrix
    # tfidf_matrix.shape

    ############################################################################
    ## Création de la matrice avec utilisation de Cosine Similarity
    ############################################################################

    # Reset index of your df and construct reverse mapping again
    # Import cosine_score
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(cv_matrix, cv_matrix)
    indices = pd.Series(data.index, index=data['title'])

    ############################################################################
    ## Création de la matrice avec utilisation de Cosine Similarity
    ############################################################################

    # Function that takes in shows title as input and gives recommendations
    def content_recommender(title, cosine_sim=cosine_sim, df=data, indices=indices, limit=4,
                            with_score=False):
        # Obtain the index of the show that matches the title
        idx = indices[title]

        # Get the pairwise similarity scores of all shows with that show
        # And convert it into a list of tuples as described above
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the cosine similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar shows. Ignore the first movie.
        sim_scores = sim_scores[1:limit]

        # Get the show indices
        show_indices = [i[0] for i in sim_scores]

        # Return the top most similar show
        rec = df.iloc[show_indices]

        # Sort by score based on purchase - popularity
        if (with_score):
            rec = rec.sort_values('score', ascending=False)

        # Return the top 10 most similar movies
        return rec

    # return df['title'].iloc[show_indices]

    solara.Markdown(
        f"""
        ## Moteur de recommandations des oeuvres - démarrage à froid - étape 1
        ### Sans prise en compte des interactions utilisateurs
    """
    )
    solara.Markdown(
        f"""
        #### Si l'utilisateur est sur l'oeuvre *{work_default_title}*
    """
    )
    # Get recommendations for a work
    rec_df_search = data[data['title'] == work_default_title]
    # solara.DataFrame(rec_df_search, items_per_page=5)

    # Get recommendations for a work
    rec_df = content_recommender(work_default_title, with_score=False)
    # solara.DataFrame(rec_df, items_per_page=5)

    ############################################################################
    ## Moteur de recommandation
    ############################################################################

    select_data_shows = np.array(data['title']).tolist()
    select_data_shows_id = np.array(data['work_id']).tolist()

    solara.Select(label="Oeuvre", value=select_default_shows, values=select_data_shows)

    if select_default_shows.value != 'Choisir...':
        sv = select_default_shows.value
    else:
        sv = work_default_title

    rec_df_search_select = data[data['title'] == sv]


    rec_df_from_select = content_recommender(sv, df=data, with_score=False)

    solara.Markdown(f"####Les recommandations pour l'oeuvre sélectionnée : *{select_default_shows.value}*")

    with solara.Row(gap="10px", justify="space-around"):

        for index, row in rec_df_from_select.head(3).iterrows():

            show_title = row['title']
            show_genre = row['genre_1']
            #show_venue = row['venue']
            show_description = row['description']

            with solara.Card(title=show_title, subtitle=show_genre):

                #solara.Markdown(f"{show_title}")

                if pd.notna(row['url']):
                    image_url = row['url']
                else:
                    image_url = 'https://placehold.co/300x400?text=No%20Image'

                solara.Image(
                   image=image_url,
                    width="30%"
                )

                solara.Markdown(f"{show_description}")

    #solara.DataFrame(rec_df_search_select, items_per_page=5)
    solara.DataFrame(rec_df_from_select, items_per_page=5)

    solara.Markdown(
        f"####Les recommandations en fonction des ventes pour l'oeuvre sélectionnée : {select_default_shows.value}")

    rec_df_rating_from_select = content_recommender(sv, df=data_with_score, with_score=True)

    rec_df_rating_from_select_display_only = pd.DataFrame(rec_df_rating_from_select['score'])
    solara.DataFrame(rec_df_rating_from_select_display_only, items_per_page=5)

    with solara.Row(gap="10px", justify="space-around"):

        for index, row in rec_df_rating_from_select.head(3).iterrows():

            show_title = row['title']
            show_genre = row['genre_1']
            #show_venue = row['venue']

            with solara.Card(title=show_title, subtitle=show_genre):

                solara.Markdown(f"{show_title}")

                if pd.notna(row['url']):
                    image_url = row['url']
                else:
                    image_url = 'https://placehold.co/300x400?text=No%20Image'

                solara.Image(image_url)


# The following line is required only when running the code in a Jupyter notebook:
Page()
