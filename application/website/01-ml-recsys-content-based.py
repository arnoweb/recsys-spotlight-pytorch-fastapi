import sys
import os
from icecream import ic
from pathlib import Path

#sys.path.append(os.path.join(os.getcwd(), 'application/utils'))

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_utils = "../utils"
absolute_path_utils = os.path.abspath(os.path.join(current_dir, relative_path_utils))
sys.path.insert(0, absolute_path_utils)

from tools import *
from exploreData import *

import solara

import matplotlib.pyplot as plt
import numpy as np
#from sentence_transformers import SentenceTransformer
import pandas as pd

from collections import OrderedDict, Counter

class DataFrameLRUCache:
    def __init__(self, maxsize=64):
        self.maxsize = maxsize
        self.store = OrderedDict()

    def get(self, key, compute_fn):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key].copy()

        value = compute_fn()
        if len(self.store) >= self.maxsize:
            self.store.popitem(last=False)
        self.store[key] = value
        return value.copy()

    def clear(self):
        self.store.clear()

#######################################################################
# stopwords_terms already populated when exploreData is imported above
#######################################################################

from wordcloud import WordCloud, STOPWORDS

# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
products_types = ["shoes", "movies", "books"]
products_type = solara.reactive("movies")

# Get default work title
work_default_title = get_work_default(products_type)
select_default_shows = solara.reactive(work_default_title)
show_raw_tables = solara.reactive(False)
content_rec_cache = DataFrameLRUCache(maxsize=64)
popularity_rec_cache = DataFrameLRUCache(maxsize=64)
cache_cleared = solara.reactive(False)


def cb_on_toggle_change(new_products_type):
    #Assign the default value of the reactive select_default_shows
    #to get the Work title changed into the select of products after changing the type of the product
    select_default_shows.value = get_work_default(new_products_type)


@solara.component
def Page():

    #solara.Markdown(f"**Current:** {select_default_shows}")


    solara.Style(
        """
        .v-application--wrap {
            padding:1em; !important
        }
        .product-select {
            max-width: 320px;
        }
        """
    )

    solara.Markdown(
        f"""
        # Machine Learning - Recommandation de produits
    """
    )

    solara.Markdown(
        f"""
        ## Suggestions personnalisées en fonction du produit sélectionné
    """
    )

    solara.Markdown(
        f"""L’objectif est de fournir une expérience de découverte et d'exploration de produits
        e-commerce ou oeuvres culturelles à l'utilisateur, en croisant analyse du contenu et intéractions. 
        La solution présentée ici permet de proposer à ses utilisateurs des suggestions de façon automatisée, 
        évolutive et très qualifiée. Lest techniques de Machine Learning sont utilisées pour répondre à cet objectif.
    """
    )
    solara.Markdown(
        f"""Nous utilisons une techniques sur cette page de démonstration, qui pourrait être la page Produit de votre site web
        et cela sans que votre utilisateur soit encore connecté, c'est ce que l'on appelle le démarrage à froid (ou Cold start).
        Un bloc 'Vous aimerez aussi' et un bloc 'Les utilisateurs ont aussi aimé' affichent des produits sur un niveau
        de pertinence très élévé et sur un matching des données intelligents, bien plus évolué que les techniques habituelles
        de requête SQL par exemple présent généralement sur un site web standard. 
    """
    )
    solara.Markdown(
        f"""
    La méthode de recommandation utilisée ici est le content-based filtering (ou filtrage basé sur le contenu).
    Ce système analyse les caractéristiques propres des produits/oeuvres et suggère des alternatives dont le profil est le plus proche.
    Nous avons couplé ce système à un filtre supplémentaire qui est celui de la popularité basée sur des ventes fictives.
    """
    )

    solara.Markdown(
        f"""Ainsi, même sans profil utilisateur, on peut déjà suggérer des produits pertinents.
    """
    )
    solara.Markdown(
        """
        **Accès API** : les mêmes algorithmes sont disponibles via notre [API](/recsys-api/docs). Consultez cette documentation pour tester les endpoints et intégrer rapidement les recommandations dans vos produits.
        """
    )

    if show_raw_tables.value:
        solara.Markdown(
            f"""
            # 1 > Exploration des données
        """
        )

    solara.Markdown(
        f"""
        ## Sélectionner un jeu de données
    """
    )

    # Define nltk stopwords (loaded via exploreData)
    #stopwords_terms = stopwords.words('english')

    solara.ToggleButtonsSingle(value=products_type, values=products_types, on_value=cb_on_toggle_change)
    solara.Switch(label="Afficher les tableaux bruts", value=show_raw_tables)
    with solara.Row(gap="8px", justify="start"):
        solara.Button(
            "Vider le cache",
            on_click=lambda: (
                content_rec_cache.clear(),
                popularity_rec_cache.clear(),
                cache_cleared.set(True) if hasattr(cache_cleared, "set") else setattr(cache_cleared, "value", True),
            ),
            color="grey",
            text=False,
            classes=["ma-2"],
        )
        if cache_cleared.value:
            solara.Markdown("Cache vidé")
    #solara.Markdown(f"**Selected**: {products_type.value}")

    products_type_selected = products_type.value

    # Dataframe of items
    data = get_data(product_type=products_type_selected, product_id=None, count=None)

    if show_raw_tables.value:
        solara.DataFrame(data, items_per_page=5)

    if show_raw_tables.value:
        solara.Markdown(
            f"""
             ## Création d'un ensemble de mots pour caractériser les produits
         """
        )
    # Création du bags of Words et de la répétition des mots sur certaines caractéristiques des oeuvres
    # If no description is given in the dataset, we cannot recommand it
    data = data.dropna(subset=['description'], how='any')

    # Set the display options to show the full content
    pd.set_option('display.max_colwidth', None)

    # To improve model, add bag_of_words with author genre or venue repeating to give more weight
    data['genre_improved_multi'] = data['genre_1'].apply(lambda x: repeat_word(x, 2))
    data['genre_improved_multi'] = data['genre_improved_multi'].apply(lambda x: ' '.join(map(str, x)))
    data.insert(4, 'genre_improved', data['genre_improved_multi'])

    # Create a bag of words composed with different features
    data['bag_of_words'] = data[data.columns[1:6]].apply(lambda x: ' '.join(x), axis=1)

    if products_type_selected == "shoes" and 'price' in data.columns:
        def bucketize_price(value):
            try:
                price = float(value)
            except (TypeError, ValueError):
                return ""

            if price < 80:
                return "price_low"
            if price < 130:
                return "price_mid"
            if price < 180:
                return "price_high"
            return "price_premium"

        data['price_bucket'] = data['price'].apply(bucketize_price)
        data['bag_of_words'] = data['bag_of_words'] + ' ' + data['price_bucket']
    # Clean HTML of bag of words
    data['bag_of_words'] = data['bag_of_words'].apply(lambda x: cleanHTML(x))

    # we'll try to find similarities based on the Description Words of a work
    data_similarities = data['bag_of_words']

    X = np.array(data_similarities)

    data_for_display_only = pd.DataFrame(data['bag_of_words'])
    if show_raw_tables.value:
        solara.DataFrame(data_for_display_only, items_per_page=5)

    if show_raw_tables.value:
        solara.Markdown(
            f"""
            ## Les achats des produits par utilisateur (fictif)
        """
        )
    data_purchase = get_data_users_purchases(product_type=products_type_selected, user_id = None, count = None)
    if show_raw_tables.value:
        solara.DataFrame(data_purchase, items_per_page=5)

    data_purchase_last_month = get_data_users_purchases(
        product_type=products_type_selected,
        user_id=None,
        count=None,
        since_days=30
    )
    if data_purchase_last_month.empty:
        data_purchase_last_month = data_purchase.copy()

    if show_raw_tables.value:
        solara.Markdown(
            f"""
            ## Les produits consultées sur le site web (fictif)
        """
        )

    data_views = get_data_users_page_views(product_type=products_type_selected, user_id = None, count = None)
    if show_raw_tables.value:
        solara.DataFrame(data_views, items_per_page=5)

    if show_raw_tables.value:
        solara.Markdown(
            f"""
            ## Les utilisateurs avec données démographiques et géographiques du site web (fictif)
        """
        )

    data_users = get_data_users(product_type=products_type_selected, user_id = None, count = None)
    if show_raw_tables.value:
        solara.DataFrame(data_users, items_per_page=5)

    if show_raw_tables.value:
        solara.Markdown(
            f"""
            ## Les scores calculés en fonction du rating qui est basé sur les ventes des produits (fictif)
        """
        )

    # Application du Rating sur les produits (fictif)
    data_ratings = pd.DataFrame(data_purchase_last_month)
    data_ratings_sorted = data_ratings.sort_values(by='work_id')

    data_ratings_2 = pd.DataFrame(data_purchase_last_month)
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

    ##todo: merge data_similarities instead of data below
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

    if show_raw_tables.value:
        solara.DataFrame(data_items_merge, items_per_page=5)

    data_with_score = data.merge(data_items_merge, on='work_id', how='left')

    data_with_score = data_with_score[['work_id', 'title_x', 'description_x', 'genre_1_x', 'genre_improved_x',
                                       'author_x', 'url_x', 'genre_improved_multi_x', 'bag_of_words_x', 'price_x', 'score']]
    data_with_score.rename(columns={'title_x': 'title', 'description_x': 'description', 'genre_1_x': 'genre_1',
                                    'genre_improved_x': 'genre_improved', 'author_x': 'author',
                                    'url_x': 'url', 'genre_improved_multi_x': 'genre_improved_multi','price_x': 'price',
                                    'bag_of_words_x': 'bag_of_words'}, inplace=True)

    data_with_score = data_with_score.sort_values('score', ascending=False)

    print(df_info(data_with_score, 'data_with_score'))

    if show_raw_tables.value:
        solara.DataFrame(data_with_score, items_per_page=5)

    #data_for_display_only = pd.DataFrame(data_with_score['score'])
    #solara.DataFrame(data_for_display_only, items_per_page=5)

    ############################################################################
    ## Vectorisation des mots avec 2 méthodes CountVectorizer ou TfIdfVectorizer
    ############################################################################

    # Import CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # Define a CV Vectorizer Object. Remove all french stopwords
    cv = CountVectorizer(stop_words=stopwords_terms)

    # Construct the required CV matrix by applying the fit_transform method on the overview feature
    cv_matrix = cv.fit_transform(data_similarities)

    # Output the shape of tfidf_matrix
    # cv_matrix.shape

    # Import TfIdfVectorizer from the scikit-learn library
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Define a TF-IDF Vectorizer Object. Remove all french stopwords
    tfidf = TfidfVectorizer(stop_words=stopwords_terms)

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

        # Get the scores of the X most similar shows. Ignore the first movie.
        sim_scores = sim_scores[1:limit]

        # Get the show indices
        show_indices = [i[0] for i in sim_scores]

        # Get the show indices
        show_scores = [i[1] for i in sim_scores]

        # Return the top most similar show
        rec = df.iloc[show_indices]
        rec['score'] = show_scores

        # Sort by score based on purchase - popularity
        if (with_score):
            rec = rec.sort_values('score', ascending=False)

        # Return the top 10 most similar movies
        return rec

    # return df['title'].iloc[show_indices]

    solara.Markdown(
        f"""
        ## Moteur de recommandations des produits - démarrage à froid - étape 1
        ### Sans prise en compte des interactions utilisateurs
    """
    )

    # Get recommendations for a work
    # rec_df_search = data[data['title'] == work_default_title]
    # solara.DataFrame(rec_df_search, items_per_page=5)

    # Get recommendations for a work
    # rec_df = content_recommender(work_default_title, with_score=False)
    # solara.DataFrame(rec_df, items_per_page=5)

    ############################################################################
    ## Moteur de recommandation
    ############################################################################

    select_data_shows = np.array(data['title']).tolist()
    select_data_shows_id = np.array(data['work_id']).tolist()
    select_data_shows_genre = np.array(data['genre_1']).tolist()

    select_options = []
    option_to_title = {}
    for work_id, title, genre in zip(select_data_shows_id, select_data_shows, select_data_shows_genre):
        label = f"{work_id} - {title} ({genre})"
        select_options.append(label)
        option_to_title[label] = title

    current_value = select_default_shows.value
    if current_value in option_to_title.values():
        for label, title in option_to_title.items():
            if title == current_value:
                select_default_shows.value = label
                break
    elif current_value not in option_to_title:
        select_default_shows.value = select_options[0]

    solara.Select(label="Produit", value=select_default_shows, values=select_options, dense=True, classes=["product-select"])

    selected_label = select_default_shows.value or select_options[0]
    sv = option_to_title.get(selected_label, data['title'].iloc[0])

    def compute_content_rec():
        return content_recommender(sv, df=data, with_score=False).reset_index(drop=True)

    rec_df_from_select = content_rec_cache.get(
        (products_type_selected, sv),
        compute_content_rec,
    )

    solara.Markdown(f"####Si l'utilisateur se rend sur le Produit **<span style='font-size:1.3em;'>{sv}</span>** alors il aura les recommandations proposées suivantes:")

    with solara.Row(gap="10px", justify="space-around"):

        for index, row in rec_df_from_select.head(3).iterrows():

            show_title = row['title']
            show_genre = row['genre_1']
            show_year = str(row['year'])
            show_score = f"{row['score']:.5f}" if pd.notna(row['score']) else "-"
            show_info = f"{show_year} - {show_genre} "
            show_price_raw = row['price'] if 'price' in row.index else None
            price_text = None
            if pd.notna(show_price_raw):
                try:
                    price_value = float(show_price_raw)
                    price_text = f"{price_value:,.2f} €"
                except (TypeError, ValueError):
                    price_text = f"{show_price_raw} €"
            show_description = row['description']

            with solara.Card(
                title=show_title,
                subtitle=show_info,
                style="text-align: center; width: 320px; margin: auto;",
                classes=["d-flex", "flex-column", "align-center"],
                elevation=4,
            ):

                #solara.Markdown(f"{show_year}")

                if price_text:
                    solara.Markdown(f"Prix : {price_text}", style="text-align: center; font-weight: 600;")

                if pd.notna(row['url']):
                    image_url = row['url']
                else:
                    image_url = 'https://placehold.co/300x400?text=No%20Image'

                solara.Column([
                    solara.Image(
                       image=image_url,
                        width="220px"
                    )
                ], align="center")

                solara.Markdown(f"Score : {show_score}", style="text-align: center; font-weight: bold;")

    #solara.DataFrame(rec_df_search_select, items_per_page=5)
    if show_raw_tables.value:
        solara.DataFrame(rec_df_from_select, items_per_page=5)

    solara.Markdown(
        f"####Les recommandations en fonction des ventes du mois dernier du produit sélectionné : **<span style='font-size:1.3em;'>{sv}</span>**")

    def compute_popularity_rec():
        return content_recommender(sv, df=data_with_score, with_score=True).reset_index(drop=True)

    rec_df_rating_from_select = popularity_rec_cache.get(
        (products_type_selected, sv, "popularity"),
        compute_popularity_rec,
    )

    rec_df_rating_from_select_display_only = pd.DataFrame(rec_df_rating_from_select['score'])
    if show_raw_tables.value:
        solara.DataFrame(rec_df_rating_from_select_display_only, items_per_page=5)

    with solara.Row(gap="10px", justify="space-around"):

        for index, row in rec_df_rating_from_select.head(3).iterrows():

            show_title = row['title']
            show_genre = row['genre_1']
            show_score = f"{row['score']:.5f}" if pd.notna(row['score']) else "-"
            show_price_raw = row['price'] if 'price' in row.index else None
            price_text = None
            if pd.notna(show_price_raw):
                try:
                    price_value = float(show_price_raw)
                    price_text = f"{price_value:,.2f} €"
                except (TypeError, ValueError):
                    price_text = f"{show_price_raw} €"
            #show_venue = row['venue']

            with solara.Card(
                title=show_title,
                subtitle=show_genre,
                style="text-align: center; width: 320px; margin: auto;",
                classes=["d-flex", "flex-column", "align-center"],
                elevation=4,
            ):

                #solara.Markdown(f"{show_title}")

                if price_text:
                    solara.Markdown(f"Prix : {price_text}", style="text-align: center; font-weight: 600;")

                if pd.notna(row['url']):
                    image_url = row['url']
                else:
                    image_url = 'https://placehold.co/300x400?text=No%20Image'

                solara.Column([
                    solara.Image(
                       image=image_url,
                        width="220px"
                    )
                ], align="center")

                solara.Markdown(f"Score : {show_score}", style="text-align: center; font-weight: bold;")

# The following line is required only when running the code in a Jupyter notebook:
Page()
