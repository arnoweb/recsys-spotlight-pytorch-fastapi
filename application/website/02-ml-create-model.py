import sys
import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_utils = "../utils"
absolute_path_utils = os.path.abspath(os.path.join(current_dir, relative_path_utils))
sys.path.insert(0, absolute_path_utils)


from exploreData import *
from modelData import *

import solara
import plotly.express as px

import numpy as np
import pandas as pd
import torch
from spotlight.interactions import Interactions


# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
sentence = solara.reactive("Solara makes our team more productive.")
word_limit = solara.reactive(10)


#default value for iteration of data evaluation model
slider_rmse_iteration_default = solara.reactive(0)

@solara.component
def Page():
    solara.Markdown(
        f"""
        # Machine Learning - Recommandation d'oeuvres
        # 2 > Création du modèle
    """
    )

    solara.Markdown(
        f"""
        ## Les oeuvres - spectacles ou films
    """
    )

    ############################################################################
    ## Exploration et chargement des données
    ############################################################################

    exploreData()

    data = get_data(product_id = None, count = None)
    display_data(data)

    data_users = get_data_users()
    display_data(data_users)

    data_purchase = get_data_purchase()
    display_data(data_purchase)

    print(DATA_WORK)

    if DATA_WORK == 'shows':
        data_ratings = get_data_ratings(data_purchase)
    elif DATA_WORK == 'movies':
        data_ratings = get_data_users_ratings()
    else:
        data_ratings = get_data_ratings(data_purchase)

    display_data(data_ratings)

    user_ids = data_ratings['user_id'].values.astype(np.int32)
    item_ids = data_ratings['work_id'].values.astype(np.int32)
    ratings_values = data_ratings['rating'].values.astype(np.int32)

    #df_info(data_ratings,'data_ratings')
    #print(user_ids)
    #print(item_ids)
    #print(ratings_values)

    #quit()

    # weight_values_views = data_views['views'].values.astype(np.int32)
    # weight_values_plays = data_plays['views'].values.astype(np.int32)

    ############################################################################
    ## Vérification existance du model torch
    ############################################################################
    current_dir = os.getcwd() + '/model/'
    relative_path_model = DATA_WORK + '_users_rating_model.pth'
    model_path = os.path.join(current_dir, relative_path_model)
    #model_path = '/Users/a.breton/digital_projects/machine-learning/recsys-spotlight-pytorch-fastapi/model/' + DATA_WORK + '_users_rating_model.pth'


    if not os.path.exists(model_path):

        # Create Spotlight Interactions
        dataset = Interactions(user_ids, item_ids, ratings=ratings_values)

        # dataset = Interactions(user_ids, item_ids, ratings=None, weights=weight_values_views)

        #Define a range of iteration on model to get best Scores or lowest RMSE
        n_iter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Track the best n_iter and corresponding RMSE
        best_rmse = np.inf  # Best RMSE for this specific n_iter
        patience_counter = 0  # Reset patience counter

        # Paramètres pour le suivi du RMSE et l'arrêt anticipé
        #n_epochs = 100
        patience = 5  # Nombre d'itérations à attendre avant d'arrêter si aucune amélioration
        tolerance = 1e-4  # Tolérance minimale d'amélioration du RMSE - 0,0001

        best_epochs = []
        rmse_scores = []

        #for epoch in range(n_epochs):
        for n_iter in n_iter_values:

            print(f"Testing model with maximum n_iter={n_iter}")

            train, test, model = create_model(dataset, type="explicit", test_percentage=0.2, n_iter=n_iter, with_model_fit=1)

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


        # Plotting
        solara.Markdown(
            f"""
            ## Evaluation du model de prediction des ratings des oeuvres par utilisateur
            #### en fonction du nombre d'itérations réalisées sur le model
        """
        )

        data_rmse = {'Nombre Iterations - Epoch': best_epochs, 'RMSE': rmse_scores}
        df_rmse = pd.DataFrame(data_rmse)
        fig = px.scatter(df_rmse, x='Nombre Iterations - Epoch', y='RMSE')
        solara.FigurePlotly(fig)


        # Get the best RMSE score (lowest)
        best_rmse = df_rmse['RMSE'].min()
        best_rmse_row = df_rmse[df_rmse['RMSE'] == best_rmse]
        best_iteration_number = int(best_rmse_row['Nombre Iterations - Epoch'])

        solara.Markdown(
            f"""
            ## Best RMSE iteration number:{best_rmse} {best_iteration_number}
        """
        )

        # Create the model with best RMSE
        users_rating_train, users_rating_test, users_rating_model = create_model(dataset, type="explicit", test_percentage=0.2, n_iter=best_iteration_number, with_model_fit=1)

        # Save model
        torch.save(users_rating_model, model_path)

    else:

        try:
         # Pull model from DVC
            dvc_result = subprocess.run(["dvc", "pull", "model/movies_users_rating_model.pth"], check=True)
            process_msg = f"OK"
        except subprocess.CalledProcessError as e:
            process_msg = f"Default Machine Learning model load failed with error:{e}"
        except FileNotFoundError as e:
            process_msg = f"Default Machine Learning model load failed with:{e}"

        solara.Markdown(
            f"""
            ### Default Machine Learning model has been loaded via DVC (Google Drive) and has not been built during that step: {process_msg}
        """
        )


# The following line is required only when running the code in a Jupyter notebook:
Page()
