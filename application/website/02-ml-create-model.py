import sys
import os
import subprocess
import json
from datetime import datetime

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
from spotlight.cross_validation import user_based_train_test_split
from spotlight.factorization.explicit import ExplicitFactorizationModel
import torch.serialization
torch.serialization.add_safe_globals([ExplicitFactorizationModel])


# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
products_types = ["shoes", "movies", "books"]
products_type = solara.reactive("movies")

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

    ############################################################################
    ## Exploration et chargement des données
    ############################################################################

    solara.ToggleButtonsSingle(
        value=products_type,
        values=products_types,
        on_value=lambda val: products_type.set(val)
    )
    # solara.Markdown(f"**Selected**: {products_type.value}")

    products_type_selected = products_type.value

    result = train_model_for_dataset(products_type_selected, render=True)


def train_model_for_dataset(product_type_selected, render=False):
    print(f"\n=== Processing dataset: {product_type_selected} ===")
    data = get_data(product_type=product_type_selected, product_id=None, count=None)
    data_users = get_data_users(product_type=product_type_selected, user_id=None, count=None)
    data_purchase = get_data_users_purchases(product_type=product_type_selected, user_id=None, count=None)

    if product_type_selected in ('movies', 'shoes', 'books'):
        data_ratings = get_data_users_ratings(product_type=product_type_selected, user_id=None, count=None)
    else:
        data_ratings = get_data_ratings(data_purchase, product_type=product_type_selected)

    user_ids = data_ratings['user_id'].values.astype(np.int32)
    item_ids = data_ratings['work_id'].values.astype(np.int32)
    ratings_values = data_ratings['rating'].values.astype(np.int32)

    dataset = Interactions(user_ids, item_ids, ratings=ratings_values)

    model_dir = os.path.join(os.getcwd(), 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{product_type_selected}_users_rating_model.pth")

    if not os.path.exists(model_path):
        n_iter_values = [1, 10, 20, 30, 40, 50]
        best_rmse = np.inf
        patience_counter = 0
        patience = 5
        tolerance = 1e-4
        best_epochs = []
        rmse_scores = []

        for n_iter in n_iter_values:
            print(f"[{product_type_selected}] Testing model with n_iter={n_iter}")
            train, test, model = create_model(dataset, type="explicit", test_percentage=0.2, n_iter=n_iter, with_model_fit=1)
            rmse = score_model(model, test)
            rmse_scores.append(rmse)
            best_epochs.append(n_iter)

            if rmse < best_rmse - tolerance:
                best_rmse = rmse
                patience_counter = 0
                print(f"[{product_type_selected}] New best RMSE: {best_rmse:.4f}")
            else:
                patience_counter += 1
                print(f"[{product_type_selected}] No improvement for {patience_counter} iterations")

            if patience_counter >= patience:
                print(f"[{product_type_selected}] Early stopping at n_iter={n_iter}")
                break

        best_rmse_idx = int(np.argmin(rmse_scores))
        best_iteration_number = best_epochs[best_rmse_idx]
        best_rmse_val = rmse_scores[best_rmse_idx]
        print(f"[{product_type_selected}] Best RMSE {best_rmse_val:.4f} at n_iter={best_iteration_number}")

        users_rating_train, users_rating_test, users_rating_model = create_model(
            dataset, type="explicit", test_percentage=0.2, n_iter=best_iteration_number, with_model_fit=1
        )
        torch.save(users_rating_model, model_path)
        print(f"[{product_type_selected}] Model saved to {model_path}")
        push_target = f"model/{product_type_selected}_users_rating_model.pth"
        try:
            subprocess.run(["dvc", "add", push_target], check=True)
        except subprocess.CalledProcessError:
            pass
        try:
            subprocess.run(["dvc", "push", push_target], check=True)
            print(f"[{product_type_selected}] Model pushed to DVC remote.")
        except subprocess.CalledProcessError as e:
            print(f"[{product_type_selected}] DVC push failed: {e}")

        metrics_path = os.path.join(model_dir, f"{product_type_selected}_users_rating_metrics.json")
        history = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    previous_payload = json.load(f)
                    history = previous_payload.get("history", [])
            except Exception:
                history = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        history.append({
            "timestamp": timestamp,
            "best_rmse": best_rmse_val,
            "best_iter": best_iteration_number
        })
        metrics_payload = {
            "dataset": product_type_selected,
            "epochs": best_epochs[:len(rmse_scores)],
            "rmse_scores": rmse_scores,
            "best_rmse": best_rmse_val,
            "best_iter": best_iteration_number,
            "timestamp": timestamp,
            "history": history,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        subprocess.run(["dvc", "add", metrics_path], check=False)
        subprocess.run(["dvc", "push", metrics_path], check=False)

        previous_entry = history[-2] if len(history) > 1 else None

        if render:
            render_model_heading(product_type_selected, model_path)
            df_rmse = pd.DataFrame(
                {'Nombre Iterations - Epoch': best_epochs[:len(rmse_scores)], 'RMSE': rmse_scores}
            )
            render_evaluation(df_rmse, best_rmse_val, best_iteration_number, product_type_selected, previous_entry)
    else:
        push_target = f"model/{product_type_selected}_users_rating_model.pth"
        try:
            subprocess.run(["dvc", "add", push_target], check=True)
        except subprocess.CalledProcessError:
            pass
        try:
            subprocess.run(["dvc", "push", push_target], check=True)
            print(f"[{product_type_selected}] Existing model synced to DVC remote.")
        except subprocess.CalledProcessError as e:
            print(f"[{product_type_selected}] Unable to push existing model to DVC: {e}")
        print(f"[{product_type_selected}] Model already exists locally. Skipping retrain.")
        metrics_path = os.path.join(model_dir, f"{product_type_selected}_users_rating_metrics.json")
        if render:
            render_model_heading(product_type_selected, model_path)
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_payload = json.load(f)
                df_rmse = pd.DataFrame(
                    {'Nombre Iterations - Epoch': metrics_payload['epochs'],
                     'RMSE': metrics_payload['rmse_scores']}
                )
                history = metrics_payload.get("history", [])
                previous_entry = history[-2] if len(history) > 1 else None
                render_evaluation(df_rmse, metrics_payload['best_rmse'], metrics_payload['best_iter'], product_type_selected, previous_entry)
            else:
                solara.Markdown("Aucun historique RMSE disponible pour ce modèle.")
        return None


def train_models_for_all():
    for dataset in products_types:
        train_model_for_dataset(dataset)


if __name__ == "__main__":
    train_models_for_all()
else:
    # The following line is required only when running the code in a Jupyter notebook:
    Page()
def render_evaluation(df_rmse, best_rmse_val, best_iteration_number, product_type_selected, previous_entry=None):
    solara.Markdown(
        f"### Évaluation du modèle - {product_type_selected.title()}"
    )
    solara.Markdown(
        f"**Meilleur RMSE** : {best_rmse_val:.4f} obtenu avec `n_iter = {best_iteration_number}`"
    )
    if previous_entry:
        solara.Markdown(
            f"**Modèle précédent du {previous_entry['timestamp']}** — Meilleur RMSE : {previous_entry['best_rmse']:.4f} avec `n_iter = {previous_entry['best_iter']}`"
        )
    solara.Markdown("### Technique de création du modèle")
    solara.Markdown(
        "Matrix factorization explicite (Spotlight) entraînée sur un split utilisateur basé "
        "sur les ratings synthétiques. Le modèle teste plusieurs valeurs de `n_iter`, mesure le RMSE "
        "sur un jeu de test utilisateur et retient l’itération offrant la meilleure précision."
    )
    fig = px.line(df_rmse, x='Nombre Iterations - Epoch', y='RMSE', markers=True)
    solara.FigurePlotly(fig)


def render_model_heading(product_type_selected, model_path):
    try:
        modified_ts = datetime.fromtimestamp(os.path.getmtime(model_path))
        modified_str = modified_ts.strftime('%Y-%m-%d %H:%M')
    except FileNotFoundError:
        modified_str = "date inconnue"
    solara.Markdown(f"### Modèle {product_type_selected.title()}")
    solara.Markdown(f"##### Date de création : {modified_str}")
    solara.Markdown(f"##### Nom du modèle : {os.path.basename(model_path)}")


def render_existing_model_metrics(product_type_selected, model_path):
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        solara.Markdown(f"#### Modèle chargé depuis `{model_path}`")
        return model
    except Exception as exc:
        solara.Markdown(f"#### Impossible de charger le modèle existant : {exc}")
        return None


def render_rmse_for_model(model, dataset, product_type_selected, title_suffix="(évaluation)"):
    train, test = user_based_train_test_split(dataset, test_percentage=0.2)
    rmse = score_model(model, test)
    solara.Markdown(
        f"**RMSE {title_suffix} pour {product_type_selected.title()}** : {rmse:.4f}"
    )
