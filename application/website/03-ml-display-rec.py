import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_utils = "../utils"
absolute_path_utils = os.path.abspath(os.path.join(current_dir, relative_path_utils))
sys.path.insert(0, absolute_path_utils)


from exploreData import *
from modelData import *

import solara

import numpy as np
import pandas as pd
import torch


# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
products_types = ["shoes", "movies", "books"]
products_type = solara.reactive("movies")

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
        # Machine Learning - Recommandation de produits
    """
    )

    solara.Markdown(
        f"""
        ## Les produits recommandés pour un utilisateur en fonction de son évaluation
    """
    )



    solara.ToggleButtonsSingle(value=products_type, values=products_types)
    solara.Markdown(f"**Selected**: {products_type.value}")

    products_type_selected = products_type.value

    data = get_data(product_type=products_type_selected, product_id=None, count=None)
    #display_data(data)

    data_users = get_data_users(product_type=products_type_selected,user_id=None, count=None)
    #display_data(data_users)


    ############################################################################
    ## Check of torch model file
    ############################################################################
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path_model = "../../model/" + products_type_selected + '_users_rating_model.pth'
    model_path = os.path.abspath(os.path.join(current_dir,     relative_path_model))
    sys.path.insert(0,  model_path)

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
                show_year = str(row['year'])
                show_score = str(row['score'])
                show_info = f"{show_year} - {show_genre} "
                #show_venue = row['venue']

                with solara.Card(title=show_title, subtitle=show_info):

                    #solara.Markdown(f"{show_venue}")

                    if pd.notna(row['url']):
                        image_url = row['url']
                    else:
                        image_url = 'https://placehold.co/300x400?text=No%20Image'

                    solara.Image(
                        image=image_url,
                        width="70%"
                    )

                    solara.Markdown(f"Score : {show_score}")
    else:

        solara.Markdown(
            f"""
            # Please, create the Machine Learning Model first
        """
        )

Page()
