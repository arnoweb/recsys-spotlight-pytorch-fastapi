import sys
import os

current_dir = os.getcwd()
utils_path = os.path.abspath(os.path.join(current_dir, '../utils'))
sys.path.append(utils_path)
from exploreData import *
from modelData import *

import solara

import numpy as np
import pandas as pd
import torch


# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
word_limit = solara.reactive(10)

# default value for select of rec select
select_default_user = solara.reactive("Arnaud Breton")

#default value for iteration of data evaluation model
slider_rmse_iteration_default = solara.reactive(0)

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
    """
    )

    solara.Markdown(
        f"""
        ## Les spectacles ou films recommandés pour un utilisateur en fonction de son évaluation
    """
    )

    data = get_data(product_id=None, count=None)
    #display_data(data)

    data_users = get_data_users()
    #display_data(data_users)


    ############################################################################
    ## Check of torch model file
    ############################################################################
    current_dir = os.getcwd() + '/model/'
    relative_path_model = DATA_WORK + '_users_rating_model.pth'
    model_path = os.path.join(current_dir, relative_path_model)

    if os.path.exists(model_path):

        # Load Model
        #load_model(rec_type, work_type)
        users_rating_model = torch.load(model_path)

        ############################################################################
        ## Rec Sys
        ############################################################################

        select_data_users = np.array(data_users['user_firstlastname']).tolist()
        select_data_users_id = np.array(data_users['user_id']).tolist()

        solara.Select(label="Utilisateurs", value=select_default_user, values=select_data_users)

        if select_default_user.value != 'Choisir...':
            sv = select_default_user.value
        else:
            sv = 'Arnaud Breton'

        rec_df_search_select = data_users[data_users['user_firstlastname'] == sv]
        #display_data(rec_df_search_select)

        # Use of the Torch model loaded
        rec_df_rating_from_select = predict_items_from_user(users_rating_model, data, int(rec_df_search_select['user_id']), 3)
        display_data(rec_df_rating_from_select)

        #score_model(users_rating_model, onp_users_rating_test)

        #solara.Markdown(
        #    f"""
        #    ## Best RMSE iteration number verified:{best_rmse}
        #"""
        #)

        with solara.Row(gap="10px", justify="space-around"):

            for index, row in rec_df_rating_from_select.head(3).iterrows():

                show_title = row['title']
                show_genre = row['genre_1']
                #show_venue = row['venue']

                with solara.Card(title=show_title, subtitle=show_genre):

                    #solara.Markdown(f"{show_venue}")

                    if pd.notna(row['url']):
                        image_url = row['url']
                    else:
                        image_url = 'https://placehold.co/300x400?text=No%20Image'

                    solara.Image(image_url)
    else:

        solara.Markdown(
            f"""
            # Please, create the Machine Learning Model first
        """
        )

Page()
