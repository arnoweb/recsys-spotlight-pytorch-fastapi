import sys
import os

sys.path.append('//application/utils')
from tools import *

import solara

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

# Stopwords dir within the repository so we never rely on remote downloads at runtime.
stopwords_root_relative = '../../data/stopwords'
stopwords_root_dir = os.path.abspath(os.path.join(current_dir, stopwords_root_relative))
stopwords_dir = stopwords_root_dir  # backward compatibility for other modules
stopwords_corpus_dir = os.path.join(stopwords_root_dir, 'corpora', 'stopwords')
os.makedirs(stopwords_corpus_dir, exist_ok=True)

STOPWORDS_LANGUAGE = os.environ.get("STOPWORDS_LANGUAGE", "english").lower()
stopwords_file_path = os.path.join(stopwords_corpus_dir, STOPWORDS_LANGUAGE)

FALLBACK_STOPWORDS = {
    'english': [
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
        'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
        "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't",
        'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have',
        "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him',
        'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',
        "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
        'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
        'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
        "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've",
        'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we',
        "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
        "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'will', 'with', "won't",
        'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
        'yourselves'
    ],
    'french': [
        'ai', 'aie', 'aient', 'aies', 'ainsi', 'allaient', 'aller', 'allons', 'alors', 'après', 'as',
        'assez', 'attendu', 'au', 'aucun', 'aucune', 'aujourd', 'auquel', 'aura', 'aurai', 'auraient',
        'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aussi', 'autre',
        'autres', 'aux', 'auxquelles', 'auxquels', 'avaient', 'avais', 'avait', 'avant', 'avec', 'avez',
        'aviez', 'avions', 'avoir', 'avons', 'ayant', 'ayez', 'ayons', 'bah', 'bas', 'bien', 'bigre',
        'bonjour', 'bravo', 'brrr', 'c', 'ça', 'car', 'ce', 'ceci', 'cela', 'celle', 'celle-ci',
        'celle-là', 'celles', 'celles-ci', 'celles-là', 'celui', 'celui-ci', 'celui-là', 'cent', 'cependant',
        'certain', 'certaine', 'certaines', 'certains', 'ces', 'cet', 'cette', 'ceux', 'ceux-ci', 'ceux-là',
        'chacun', 'chaque', 'cher', 'chère', 'chères', 'chers', 'chez', 'chiche', 'chut', 'ci', 'cinq',
        'clac', 'clic', 'combien', 'comme', 'comment', 'compris', 'concernant', 'contre', 'couic', 'crac',
        'd', 'da', 'dans', 'de', 'debout', 'dedans', 'dehors', 'delà', 'depuis', 'derrière', 'des',
        'dès', 'désormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'deux', 'deuxième', 'deuxièmement',
        'devant', 'devers', 'devoir', 'devra', 'devrait', 'diable', 'différent', 'différente', 'différentes',
        'différents', 'dire', 'divers', 'diverse', 'diverses', 'dix', 'dix-huit', 'dix-neuf', 'dix-sept',
        'dois', 'doit', 'donc', 'dont', 'douze', 'dring', 'du', 'duquel', 'durant', 'dès', 'dés',
        'effet', 'eh', 'elle', 'elle-même', 'elles', 'elles-mêmes', 'en', 'encore', 'entre', 'envers',
        'environ', 'es', 'ès', 'est', 'et', 'étaient', 'étais', 'était', 'étant', 'été', 'étée',
        'étées', 'étés', 'êtes', 'être', 'eu', 'eue', 'eues', 'eus', 'eusse', 'eussent', 'eusses',
        'eussiez', 'eussions', 'eut', 'eux', 'eux-mêmes', 'excepté', 'façon', 'fais', 'faisaient', 'faisant',
        'fait', 'faite', 'faites', 'faits', 'faut', 'feront', 'fi', 'flac', 'floc', 'font', 'gens',
        'ha', 'hé', 'hein', 'hélas', 'hem', 'hep', 'hi', 'ho', 'holà', 'hop', 'hormis', 'hors',
        'hou', 'houp', 'hue', 'hui', 'hurrah', 'il', 'ils', 'importe', 'j', 'je', 'jusqu', 'jusque',
        'l', 'la', 'laisser', 'laquelle', 'las', 'le', 'lequel', 'les', 'lès', 'lesquelles', 'lesquels',
        'leur', 'leurs', 'longtemps', 'lors', 'lorsque', 'lui', 'lui-même', 'là', 'lès', 'ma', 'maint',
        'mais', 'malgré', 'me', 'même', 'mêmes', 'merci', 'mes', 'mien', 'mienne', 'miennes', 'miens',
        'mille', 'mince', 'moi', 'moi-même', 'moins', 'mon', 'moyennant', 'multiple', 'multiples', 'même',
        'n', 'na', 'ne', 'néanmoins', 'neuf', 'neuvième', 'ni', 'nombreuses', 'nombreux', 'non', 'nos',
        'notamment', 'notre', 'nôtre', 'nôtres', 'nous', 'nous-mêmes', 'nul', 'nulle', 'o', 'ô', 'oh',
        'ohé', 'olé', 'ollé', 'on', 'ont', 'onze', 'onzième', 'ore', 'ou', 'où', 'ouf', 'ouias',
        'oust', 'ouste', 'outre', 'paf', 'pan', 'par', 'parce', 'parmi', 'partant', 'particulier', 'particulière',
        'particulièrement', 'pas', 'passé', 'pendant', 'personne', 'peu', 'peut', 'peuvent', 'peux', 'pff',
        'pfft', 'pfut', 'pif', 'plein', 'plouf', 'plus', 'plusieurs', 'plutôt', 'pouah', 'pour', 'pourquoi',
        'premier', 'première', 'premièrement', 'près', 'proche', 'psitt', 'pu', 'puis', 'puisque', 'pur',
        'pure', 'qu', 'quadrillion', 'quand', 'quant', 'quanta', 'quant-à-soi', 'quarante', 'quatorze',
        'quatre', 'quatre-vingt', 'quatrième', 'quatrièmement', 'que', 'quel', 'quelconque', 'quelle',
        'quelles', 'quelque', 'quelques', 'quelquun', "quelqu'un", 'quels', 'qui', 'quiconque', 'quinze',
        'quoi', 'quoique', 'rien', 's', 'sa', 'sans', 'sapristi', 'sauf', 'se', 'seize', 'selon',
        'sept', 'septième', 'sera', 'serai', 'seraient', 'serais', 'serait', 'seras', 'serez', 'seriez',
        'serions', 'serons', 'seront', 'ses', 'seul', 'seule', 'seulement', 'si', 'sien', 'sienne',
        'siennes', 'siens', 'sinon', 'six', 'sixième', 'soi', 'soi-même', 'soient', 'sois', 'soit',
        'soixante', 'sommes', 'son', 'sont', 'sous', 'souvent', 'soyez', 'soyons', 'suis', 'suivant',
        'sur', 'surtout', 't', 'ta', 'tac', 'tandis', 'tant', 'te', 'tel', 'telle', 'tellement',
        'telles', 'tels', 'tenant', 'tes', 'tic', 'tien', 'tienne', 'tiennes', 'tiens', 'toc', 'toi',
        'toi-même', 'ton', 'touchant', 'toujours', 'tous', 'tout', 'toute', 'toutes', 'treize', 'trente',
        'très', 'trois', 'troisième', 'troisièmement', 'trop', 'tsoin', 'tsouin', 'tu', 'un', 'une',
        'unes', 'uns', 'va', 'vais', 'vas', 'vé', 'vers', 'via', 'vif', 'vifs', 'vingt', 'vivat',
        'vive', 'vives', 'vlan', 'voici', 'voilà', 'vont', 'vos', 'votre', 'vous', 'vous-mêmes',
        'vu', 'wagons', 'zut'
    ]
}

fallback_words = FALLBACK_STOPWORDS.get(STOPWORDS_LANGUAGE, FALLBACK_STOPWORDS['english'])

if not os.path.exists(stopwords_file_path):
    with open(stopwords_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(sorted(set(fallback_words))))

with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    stopwords_terms = [line.strip() for line in file if line.strip()]

######## PRODUCTS

def get_data(product_type, product_id, count):

    relative_path_products_items = f"../../data/{product_type}/products.csv"
    absolute_path_products_items = os.path.abspath(os.path.join(current_dir, relative_path_products_items))

    data = pd.read_csv(absolute_path_products_items)
    base_columns = ['work_id', 'title', 'description', 'genre_1', 'author', 'year', 'url']
    optional_columns = []
    if 'price' in data.columns:
        optional_columns.append('price')

    data = data[base_columns + optional_columns]
    data['work_id'] = data['work_id'].astype(int)

    if 'price' not in data.columns:
        data['price'] = pd.NA

    if product_id is not None:
        if (data['work_id'] == product_id).any():
            data = data[data['work_id'] == product_id]
        else:
            data_error = pd.DataFrame({
                "Error": [f"This product ID {product_id} does not exist"]
            })
            return data_error

    if count is not None:
        data = data.iloc[:count]

    return data


def get_data_users(product_type, user_id, count):

    relative_path_products_users = f"../../data/{product_type}/products_users.csv"
    absolute_path_products_users = os.path.abspath(os.path.join(current_dir, relative_path_products_users))

    data = pd.read_csv(absolute_path_products_users)
    data['user_firstlastname'] = data['user_firstname'] + ' ' + data['user_lastname']

    if user_id is not None:
        if (data['user_id'] == user_id).any():
            data = data[data['user_id'] == user_id]
        else:
            data_error = pd.DataFrame({
                "Error": [f"This user ID {user_id} does not exist"]
            })
            return data_error

    if count is not None:
        data = data.iloc[:count]

    return data

def get_data_users_ratings(product_type, user_id, count):

    relative_path_products_users_ratings = f"../../data/{product_type}/products_ratings.csv"
    absolute_path_products_users_ratings = os.path.abspath(
        os.path.join(current_dir, relative_path_products_users_ratings))

    data = pd.read_csv(absolute_path_products_users_ratings)

    if user_id is not None:
        if (data['user_id'] == user_id).any():
            data = data[data['user_id'] == user_id]
        else:
            data_error = pd.DataFrame({
                "Error": [f"This user ID {user_id} does not have a rating"]
            })
            return data_error

    if count is not None:
        data = data.iloc[:count]

    return data


def get_data_users_purchases(product_type, user_id, count, since_days=None):

    relative_path_products_users_purchases = f"../../data/{product_type}/products_users_purchases.csv"
    absolute_path_products_users_purchases = os.path.abspath(
        os.path.join(current_dir, relative_path_products_users_purchases))

    data = pd.read_csv(absolute_path_products_users_purchases)

    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        if since_days is not None:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=since_days)
            data = data[data['timestamp'] >= cutoff]
        data = data.sort_values('timestamp', ascending=False).reset_index(drop=True)

    if user_id is not None:
        if (data['user_id'] == user_id).any():
            data = data[data['user_id'] == user_id]
        else:
            data_error = pd.DataFrame({
                "Error": [f"This user ID {user_id} does not have a purchase"]
            })
            return data_error

    if count is not None:
        data = data.iloc[:count]

    return data


def get_data_users_page_views(product_type, user_id, count):

    relative_path_products_users_page_views = f"../../data/{product_type}/products_users_page_views.csv"
    absolute_path_products_users_page_views = os.path.abspath(
        os.path.join(current_dir, relative_path_products_users_page_views))

    data = pd.read_csv(absolute_path_products_users_page_views)

    if user_id is not None:
        if (data['user_id'] == user_id).any():
            data = data[data['user_id'] == user_id]
        else:
            data_error = pd.DataFrame({
                "Error": [f"This user ID {user_id} does not have a page view"]
            })
            return data_error

    if count is not None:
        data = data.iloc[:count]

    return data


def display_data(df):
    solara.DataFrame(df, items_per_page=5)

def get_work_default(product_type='movies'):
    if product_type == 'movies':
        return 'Interstellar'
    elif product_type == 'books':
        return '1984'
    elif product_type == 'shoes':
        return 'Nike Air Zoom Pegasus 41'
    else:
        return 'Interstellar'


def get_data_similarities(data):
    # Création du bags of Words et de la répétition des mots sur certaines caractéristiques des spectacles
    # If no description is given in the dataset, we cannot recommand it
    data = data.dropna(subset=['description'], how='any')

    # Set the display options to show the full content
    pd.set_option('display.max_colwidth', None)

    # To improve model, add bag_of_words with author genre or venue repeating to give more weight
    data['genre_improved_multi'] = data['genre_1'].apply(lambda x: repeat_word(x, 2))
    data['genre_improved_multi'] = data['genre_improved_multi'].apply(lambda x: ' '.join(map(str, x)))
    data.insert(4, 'genre_improved', data['genre_improved_multi'])

    # Clean HTML of bag of words
    data[data.columns[1:6]] = data[data.columns[1:6]].applymap(cleanHTML)

    # Create a bag of words composed with different features
    data['bag_of_words'] = (
             data[data.columns[1:6]].astype(str).apply(lambda x: ' '.join(x), axis=1)
            + ' ' + data['year'].astype(int).astype(str)
    )

    # Clean HTML of bag of words
    #data['bag_of_words'] = data['bag_of_words'].apply(lambda x: cleanHTML(x))

    return data


def get_data_ratings(data_purchase, product_type):

    if product_type == 'shows':
        data_ratings = pd.DataFrame(data_purchase)
        data_ratings['rating'] = data_ratings['total_purchases'].apply(rating_to_show)

    elif product_type in ('movies', 'books', 'shoes'):
        data_ratings = pd.DataFrame(data_purchase)
        data_ratings['rating'] = data_ratings['total_purchases'].apply(rating_to_allpurchases_of_movies)
    else:
        data_ratings = pd.DataFrame(data_purchase)
        data_ratings['rating'] = data_ratings['total_purchases']


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

    #display_data(data_with_score)

    data_with_score = data_with_score[['work_id', 'title_x', 'description_x', 'genre_x', 'genre_improved_x',
                                       'author_x', 'url_x', 'genre_improved_multi_x', 'bag_of_words_x', 'price_x', 'score']]
    data_with_score.rename(columns={'title_x': 'title', 'description_x': 'description', 'genre_x': 'genre_1',
                                    'genre_improved_x': 'genre_improved', 'author_x': 'author',
                                    'url_x': 'url', 'genre_improved_multi_x': 'genre_improved_multi',
                                    'price_x': 'price',
                                    'bag_of_words_x': 'bag_of_words'}, inplace=True)
    return data_with_score

@solara.component
def exploreData(product_type='movies'):

    data = get_data(product_type=product_type, product_id=None, count=None)
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

    data_purchase = get_data_users_purchases(product_type=product_type, user_id=None, count=None)
    display_data(data_purchase)

    solara.Markdown(
        f"""
        ## Les oeuvres spectacles ou films vus sur le site (fictif)
    """
    )

    data_views = get_data_users_page_views(product_type=product_type, user_id=None, count=None)
    display_data(data_views)

    solara.Markdown(
        f"""
        ## Les utilisateurs avec données démographiques et géographiques du site (fictif)
    """
    )

    data_users = get_data_users(product_type=product_type, user_id=None, count=None)
    display_data(data_users)


    solara.Markdown(
        f"""
        ## Les scores calculés en fonction du rating qui est basé sur les ventes des oeuvres spectacles ou films (fictif)
    """
    )

    # Application du Rating sur les spectacles (fictif)
    data_ratings = get_data_ratings(data_purchase, product_type)

    data_items_merge = get_data_item_score(data, data_ratings)
    display_data(data_items_merge)

    #data_with_score = get_data_with_score(data, data_items_merge)
    #display_data(data_with_score)

    #data_for_display_only = pd.DataFrame(data_with_score['score'])
    #display_data(data_for_display_only)
