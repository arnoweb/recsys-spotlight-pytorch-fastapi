from spotlight.cross_validation import user_based_train_test_split
from spotlight.interactions import Interactions
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import subprocess
import time


import sys
import os

import torch

from application.utils.exploreData import DATA_WORK

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def dvc_push(file_path=None):

    try:
        if file_path:
            # Add and Push a specific file
            subprocess.run(['dvc', 'add', file_path], check=True)
            subprocess.run(['dvc', 'push', file_path], check=True)
        else:
            # Add and Push all DVC-tracked files
            subprocess.run(['dvc', 'add'], check=True)
            subprocess.run(['dvc', 'push'], check=True)
        print("DVC push successful.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pushing with DVC: {e}", file=sys.stderr)

def load_model(data_work_type):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path_model = "../../model/" + data_work_type + '_users_rating_model.pth'
    absolute_path_model = os.path.abspath(os.path.join(current_dir, relative_path_model))
    sys.path.insert(0, absolute_path_model)

    try:
        if os.path.exists(absolute_path_model):
            rec_model = torch.load(absolute_path_model)
            #, weights_only=True
        else:
            rec_model = 'Path error: model can not be loaded'
    except Exception as e:
        # Gérer les erreurs potentielles
        return f"Error of model file path : {e}"

    return rec_model

def create_dataset_interactions(data_ratings):

    user_ids = data_ratings['user_id'].values.astype(np.int32)
    item_ids = data_ratings['work_id'].values.astype(np.int32)
    ratings_values = data_ratings['rating'].values.astype(np.int32)

    # Create Spotlight Interactions
    dataset = Interactions(user_ids, item_ids, ratings=ratings_values)

    return dataset

def get_best_iteration_for_model(dataset):

    # Define a range of iteration on model to get best Scores or lowest RMSE
    #n_iter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_iter_values = [1, 2, 3, 4, 5]

    # Track the best n_iter and corresponding RMSE
    best_rmse = np.inf  # Best RMSE for this specific n_iter
    patience_counter = 0  # Reset patience counter

    # Paramètres pour le suivi du RMSE et l'arrêt anticipé
    # n_epochs = 100
    patience = 5  # Nombre d'itérations à attendre avant d'arrêter si aucune amélioration
    tolerance = 1e-4  # Tolérance minimale d'amélioration du RMSE - 0,0001

    best_epochs = []
    rmse_scores = []

    for n_iter in n_iter_values:

        print(f"Testing model with maximum n_iter={n_iter}")

        train, test, model = create_model(dataset, type="explicit", test_percentage=0.2, n_iter=n_iter,
                                          with_model_fit=1)

        rmse = score_model(model, test)
        rmse_scores.append(rmse)
        best_epochs.append(n_iter)

        # On s'assure que toute première valeur de RMSE calculée sera inférieure à best_rmse, car tout nombre fini est inférieur à l'infini
        if rmse < best_rmse - tolerance:
            best_rmse = rmse
            patience_counter = 0  # Réinitialiser le compteur si amélioration
            print(f"New best RMSE for n_iter={n_iter}: {best_rmse:.4f}")
        else:
            patience_counter += 1  # Incrémenter si pas d'amélioration
            print(f"No improvement for {patience_counter} epochs.")

        # Arrêter si aucune amélioration après "patience" itérations
        if patience_counter >= patience:
            print(f'Early stopping at epoch {n_iter + 1}, best RMSE: {best_rmse}')
            break

    # Get the best RMSE score (lowest)
    data_rmse = {'Nombre Iterations - Epoch': best_epochs, 'RMSE': rmse_scores}
    df_rmse = pd.DataFrame(data_rmse)
    best_rmse = df_rmse['RMSE'].min()
    best_rmse_row = df_rmse[df_rmse['RMSE'] == best_rmse]
    best_iteration_number = int(best_rmse_row['Nombre Iterations - Epoch'])

    return best_iteration_number


def save_model(users_rating_model, data_work_type):

    #if Gdrive Auth Error, delete files in /Library/Caches/pydrive2fs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path_model = "../../model/" + data_work_type + '_users_rating_model.pth'
    absolute_path_model = os.path.abspath(os.path.join(current_dir, relative_path_model))
    sys.path.insert(0, absolute_path_model)


    relative_path_model_dvc = "../../model/" + data_work_type + '_users_rating_model.pth.dvc'
    absolute_path_model_dvc = os.path.abspath(os.path.join(current_dir, relative_path_model_dvc))
    sys.path.insert(1, absolute_path_model_dvc)

    #try:
    #    # Pull model from DVC
    #    dvc_result = subprocess.run(["dvc", "pull", "--force", absolute_path_model], check=True)
    #    process_msg = f"OK"
    #except subprocess.CalledProcessError as e:
    #    process_msg = f"Default Machine Learning model load failed with error:{e}"
    #except FileNotFoundError as e:
    #    process_msg = f"Default Machine Learning model load failed with:{e}"
    #print(process_msg)

    #if process_msg == "OKKO":
    #    return pd.DataFrame({'message': ['The existing model was successfully loaded from DVC - google drive, no need to regenerate one model']})
    #else:

    try:

        # Pull model from DVC
        dvc_result = subprocess.run(["dvc", "remove", absolute_path_model_dvc], check=True)
        process_msg = f"Default Machine Learning model has been removed and can now be updated"

        # Save model
        torch.save(users_rating_model, absolute_path_model)

        # Add and Push it to DVC - Google Drive
        dvc_push(absolute_path_model)

    except subprocess.CalledProcessError as e:
        process_msg = f"Default Machine Learning model removed failed with error:{e}"
    except FileNotFoundError as e:
        process_msg = f"Default Machine Learning model removed failed with:{e}"

    print(process_msg)

    return pd.DataFrame({'message': ['The model was successfully saved locally and push via DVC to google drive']})


def create_model(dataset, type='explicit', test_percentage=0.2, n_iter=1, with_model_fit=0):
    # Split the dataset into train and test sets
    train, test = user_based_train_test_split(dataset, test_percentage=test_percentage)

    # Training an implicit factorization model, use of Adam optimizer
    if type == "implicit":
        model = ImplicitFactorizationModel(n_iter=n_iter, loss='bpr')
    elif type == "explicit":
        model = ExplicitFactorizationModel(n_iter=n_iter, loss='regression')
    else:
        model = ExplicitFactorizationModel(n_iter=n_iter, loss='regression')

    if with_model_fit == 1:
        model.fit(train, verbose=True)

    return train, test, model


def score_model(model, test):
    rmse = rmse_score(model, test)
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse


def predict_items_from_user(model, data, user_id, count: int = 3):
    # Predicting X items for user by apply model
    scores = model.predict(user_id)
    print('Scores:', scores)

    predict_top_items_id_works = np.argsort(scores)[-count:][::-1]
    print('predict_top_items_id_works:', predict_top_items_id_works)

    # Filter the data to get the top predicted items
    rec_predict_works = data[data['work_id'].isin(predict_top_items_id_works)].copy()
    print('rec_predict_works:', rec_predict_works)

    # Add the scores to the DataFrame
    rec_predict_works['score'] = rec_predict_works['work_id'].map(lambda x: scores[x])

    # Sort the DataFrame by higher score (descending order)
    rec_predict_works_sorted = rec_predict_works.sort_values(by='score', ascending=False)

    return rec_predict_works_sorted


def predict_items_from_user_api(data_work_type, data_works, data_purchases, user_id, count: int = 3):

    # We don't want to propose works that the user has already buy
    data_works_purchased_by_user = data_purchases[data_purchases['user_id'] == user_id]

    if not data_works_purchased_by_user.empty:

        data_ids_works_purchased_by_user = data_works_purchased_by_user['work_id'].unique()
        data_works_filtered = data_works[~data_works['work_id'].isin(data_ids_works_purchased_by_user)]

        #example: we display 5 shows with count=5, if user bought 2 shows, we increment count+2 shows coz data less of 2 shows
        #if data_shows_purchased_by_user.shape[0] > 0:
        #    count += data_shows_purchased_by_user.shape[0]

    else:
        data_works_filtered = data_works

    # Use of the Torch model loaded
    rec_model = load_model(data_work_type)

    rec_df_rating = predict_items_from_user(rec_model, data_works_filtered, int(user_id), count)

    ## Transform to JSON
    rec_df_rating_json = rec_df_rating.to_json(orient='records')

    return rec_df_rating_json


def add_ratings_from_purchases(data, data_purchase):

    # Application du Rating sur les oeuvres (fictif)
    data_ratings = pd.DataFrame(data_purchase)
    data_ratings_sorted = data_ratings.sort_values(by='work_id')

    data_ratings_2 = pd.DataFrame(data_purchase)
    data_ratings_2_sorted = data_ratings_2.sort_values(by='work_id')

    #Apply of a rating regarding purchases of the products
    data_ratings_2_sorted['rating'] = data_ratings_sorted['total_purchases'].apply(rating_to_movie)

    #print(df_info(data_ratings_sorted, 'data_ratings_sorted'))
    #print(df_info(data_ratings_2_sorted, 'data_ratings_2_sorted'))

    data_items = pd.DataFrame()
    data_items = data_ratings_sorted.groupby('work_id').size().reset_index(name='rating_count')
    data_items['rating_average'] = data_ratings_2_sorted.groupby('work_id')['rating'].transform('mean')

    #print(df_info(data_items, 'data_items'))
    # print(data_items.info())
    # print(data.info())
    # print(data_items['work_id'].dtype)

    data_items_merge = data_items.merge(data, on='work_id', how='left')

    data_items_merge['rating_average'] = data_items_merge['rating_count'].apply(rating_to_allpurchases_of_movies)

    #print(df_info(data_items_merge, 'data_items_merge'))

    # Calculate the number of votes garnered by the 80th percentile show
    m = data_items_merge['rating_count'].quantile(0.80)
    C = data_items_merge['rating_average'].mean()

    # Compute the score using the weighted_rating function defined above
    data_items_merge['score'] = data_items_merge.apply(lambda data_items_merge: weighted_rating(data_items_merge, m, C),
                                                       axis=1)

    data_items_merge = data_items_merge.sort_values('work_id', ascending=True)
    data_items_merge = data_items_merge.reset_index()

    return data_items_merge

def model_vectorization_cosine_similarities(data_works, data_similarities, stopwords_french, typeOfVec = "CountVec"):

    if(typeOfVec == "Tfidf"):

        # Define a TF-IDF Vectorizer Object. Remove all french stopwords
        tfidf = TfidfVectorizer(stop_words=stopwords_french)

        # Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
        matrix = tfidf.fit_transform(data_similarities)

        # Output the shape of tfidf_matrix
        # matrix.shape

    else:

        # Define a CV Vectorizer Object. Remove all french stopwords
        cv = CountVectorizer(stop_words=stopwords_french)

        # Construct the required CV matrix by applying the fit_transform method on the overview feature
        matrix = cv.fit_transform(data_similarities)

        # Output the shape of tfidf_matrix
        # matrix.shape

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(matrix, matrix)
    indices = pd.Series(data_works.index, index=data_works['title'])

    return cosine_sim, indices



# Function that takes in shows title as input and gives recommendations
def model_content_recommender(title, cosine_sim, df, indices, limit=4,
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


def model_vector_db_init():

    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    return pc

def model_vector_create_index():

    pc = model_vector_db_init()

    index_name = DATA_WORK

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # Select Index
    index = pc.Index(index_name)
    return index

def model_vector_getModel():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2').to(device)
    return model

def model_generate_embeddings(data_similarities):

    model = model_vector_getModel()

    sentences = [x["bag_of_words"] for x in data_similarities]
    embeddings = model.encode(sentences)

    print("embeddings shape: ", embeddings.shape)

    return model, embeddings


def model_vector_indexing(data_works, data_similarities_prepared_for_vectors):

    # Select Index
    index = model_vector_create_index()

    # Create embeddings
    model, embeddings = model_generate_embeddings(data_similarities_prepared_for_vectors)

    vectors = []
    for ds, dv, e in zip(data_works, data_similarities_prepared_for_vectors, embeddings):
        vectors.append({
            "id": dv['id'],
            "values": e,
            "metadata": {
                "title": ds["title"],
                "description": ds["description"],
                "genre": ds["genre_1"],
                "auteur": ds["auteur"],
                "year": ds["year"]
            }
        })

    index.upsert(
        vectors=vectors,
        namespace="recsys_movies"
    )

    time.sleep(10)  # Wait for the upserted vectors to be indexed

    index_info = index.describe_index_stats()
    print(index_info)
    total_vectors = index_info.get('total_vector_count')
    print(total_vectors)

    return index, model, total_vectors


def model_content_recommender_vectors(data_works, data_similarities, title, count):

    # Select Index
    index = model_vector_create_index()

    # Get Model
    model = model_vector_getModel()

    #index, model = model_vector_indexing(data_works, data_similarities)

    query = title

    query_embedding = model.encode(query).tolist()
    #print(query_embedding)

    results = index.query(
        namespace="recsys_movies",
        vector=query_embedding,
        top_k=count,
        include_values=False,
        include_metadata=True,
        filter={"title": {"$ne": title}}
    )

    filtered_results = [
        {"work_id": match["id"], "title": match["metadata"]["title"]}
        for match in results["matches"]
    ]

    return filtered_results