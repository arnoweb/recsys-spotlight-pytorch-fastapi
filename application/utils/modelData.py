from spotlight.cross_validation import user_based_train_test_split
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
import numpy as np

import sys
import os

import torch


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


def load_model(rec_type, work_type):
    if rec_type == "content":
        model_path = '/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/model/' + work_type + 'content_based_model.pth'
    elif rec_type == "collaborative":
        model_path = '/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/model/' + work_type + 'users_rating_model.pth'
    else:
        model_path = '/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/model/' + work_type + 'content_based_model.pth'

    try:
        if os.path.exists(model_path):
            rec_model = torch.load(model_path)
        else:
            rec_model = 'Path error: model can not be loaded'
    except Exception as e:
        # Gérer les erreurs potentielles
        return f"Error of model file path : {e}"

    return rec_model


def predict_items_from_user_api(rec_type, data_shows, data_purchases, user_id, count: int = 3):
    # We don't want to propose shows that the user has already buy
    data_shows_purchased_by_user = data_purchases[data_purchases['id_user'] == user_id]

    if not data_shows_purchased_by_user.empty:
        data_ids_shows_purchased_by_user = data_shows_purchased_by_user['id_show'].unique()
        data_shows_filtered = data_shows[~data_shows['id_show'].isin(data_ids_shows_purchased_by_user)]

        #example: we display 5 shows with count=5, if user bought 2 shows, we increment count+2 shows coz data less of 2 shows
        #if data_shows_purchased_by_user.shape[0] > 0:
        #    count += data_shows_purchased_by_user.shape[0]

    else:
        data_shows_filtered = data_shows

    # Use of the Torch model loaded
    rec_model = load_model(rec_type, DATA_WORK)

    rec_df_rating = predict_items_from_user(rec_model, data_shows_filtered, int(user_id), count)

    ## Transform to JSON
    rec_df_rating_json = rec_df_rating.to_json(orient='records')

    return rec_df_rating_json
