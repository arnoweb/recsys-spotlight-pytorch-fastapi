from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.templating import Jinja2Templates
import json

from pydantic import BaseModel, Field, Json
from typing import Any
from typing import List
from enum import Enum

import sys
import os
import uvicorn


current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_utils = "../utils"
absolute_path_utils = os.path.abspath(os.path.join(current_dir, relative_path_utils))
sys.path.insert(0, absolute_path_utils)

#print(sys.path)

from exploreData import *
from modelData import *

#Stopwords dir
stopwords_relative_path = '../../data/stopwords'
stopwords_dir = os.path.abspath(os.path.join(current_dir, stopwords_relative_path))


#class ItemType(str, Enum):
#    type1 = "Content"
#    type2 = "Collaborative"

tags_metadata = [
    {
        "name": "getProducts",
        "description": "List of products or one product",
    },
    {
        "name": "getUsers",
        "description": "List of users",
    },
    {
        "name": "getUsersPurchases",
        "description": "List of users purchases",
    },
    {
        "name": "getUsersRatings",
        "description": "List of users ratings",
    },
    {
        "name": "getUsersPageViews",
        "description": "List of users page views",
    },
    {
        "name": "getUsersTags",
        "description": "List of users tags on works",
    },
    {
        "name": "generateModel",
        "description": "Create the Machine Learning Model based on ratings of the users. Be patient, it takes time...",
    },
    {
        "name": "getRecContent",
        "description": "Get a list of recommendated works based on similar features of products - Content-based Filtering",
    },
    {
        "name": "getRecContentVectorCreateIndex",
        "description": "Create an Index with vectors from data - Pinecone tool - https://app.pinecone.io/",
    },
    {
        "name": "getRecContentVectorDb",
        "description": "Get a list of recommendated works based on similar features of products - based on a Vector DB - Content-based Filtering",
    },
    {
        "name": "getRecCollaborative",
        "description": "Get a list of recommendated works for a user based on others users ratings/purchase - User-based Collaborative Filtering",
    },
]

# Function to call the generate model endpoint
def update_model():
    with httpx.Client() as client:
        #local
        #response = client.get("http://127.0.0.1:8765/generateModel?data_work_type=movies")
        #distant docker
        response = client.get("http://127.0.0.1:8080/recsys-api/generateModel?data_work_type=movies")
        print("Task executed with response:", response.json())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add the scheduler
    scheduler = BackgroundScheduler()
    ####CRON LIKE ######
    scheduler.add_job(update_model, 'cron', minute='*/5')
    scheduler.start()
    print("Scheduler started for create ML model.")

    yield

    scheduler.shutdown()
    print("Scheduler shutdown for create ML model.")


#app = FastAPI(lifespan=lifespan, title="ML API - Predict Rec Products", description="Get predicted recommendated products", version="0.0.1", openapi_tags=tags_metadata)
app = FastAPI(title="ML API - Predict Rec Products",
              description="Get predicted recommendated products",
              version="0.0.1",
              openapi_tags=tags_metadata,
              openapi_url="/recsys-api/openapi.json",
              docs_url="/recsys-api/docs"
              )


class typeOfDataWork(str, Enum):
    value1 = "movies"
    value2 = "shows"

class Item(BaseModel):
    count: int = Field(default='3')
    user_name: int = Field(default='Arnaud Breton')

    #df: Json[Any] = Field(default='{"count": 3}')

@app.get("/")
async def root():
    return {"message": "API- Recommandation System based either on Machine Learning Model Spotlight TfidfVectorizer or Pinecone with encoding model all-MiniLM-L12-v2"}

@app.get("/products", tags=["getProducts"])
async def getProducts(product_id: Optional[int] = None, count: Optional[int] = None):
    # List of products (works, movies, shows)
    data = get_data(product_id, count).to_json(orient='records')
    return data

@app.get("/users", tags=["getUsers"])
async def getUsers(user_id: Optional[int] = None, count: Optional[int] = None):
    # List of users
    data = get_data_users(user_id, count).to_json(orient='records')
    return data

@app.get("/usersPurchases", tags=["getUsersPurchases"])
async def getUsersPurchases(user_id: Optional[int] = None, count: Optional[int] = None):
    # List of users purchases
    data = get_data_users_purchases(user_id, count).to_json(orient='records')
    return data

@app.get("/usersRatings", tags=["getUsersRatings"])
async def usersRatings(user_id: Optional[int] = None, count: Optional[int] = None):
    # List of users purchases
    data = get_data_users_ratings(user_id, count).to_json(orient='records')
    return data

@app.get("/usersPageViews", tags=["getUsersPageViews"])
async def usersPageViews(user_id: Optional[int] = None, count: Optional[int] = None):
    # List of users purchases
    data = get_data_users_page_views(user_id, count).to_json(orient='records')
    return data

@app.get("/usersTags", tags=["getUsersTags"])
async def usersTags(user_id: Optional[int] = None, count: Optional[int] = None):
    # List of users purchases
    data = get_data_users_tags(user_id, count).to_json(orient='records')
    return data

@app.get("/generateModel", tags=["generateModel"])
async def generate_model(data_work_type:typeOfDataWork):

    if data_work_type == 'shows':
        data_purchase = get_data_users_purchases(user_id=None, count=None)
        data_ratings = get_data_ratings(data_purchase)
    elif data_work_type == 'movies':
        data_purchase = get_data_users_purchases(user_id=None, count=None)
        data_ratings = get_data_users_ratings(user_id = None, count = None)
    else:
        data_purchase = get_data_users_purchases(user_id=None, count=None)
        data_ratings = get_data_ratings(data_purchase)

    #create dataset interaction with Spotlight
    dataset = create_dataset_interactions(data_ratings)

    #get best iteration - best score - patience...it takes time
    best_iteration_number = get_best_iteration_for_model(dataset)

    #create of the model
    users_rating_train, users_rating_test, users_rating_model = create_model(dataset, type="explicit",
                                                                             test_percentage=0.2,
                                                                             n_iter=best_iteration_number,
                                                                             with_model_fit=1)

    #save the model
    df_message = save_model(users_rating_model,data_work_type)

    return df_message

@app.get("/getRec/content/{product_id}/{count}", tags=["getRecContent"])
async def get_rec_content(data_work_type:typeOfDataWork, product_id: int, count: int):
    #try:
        # List of works
        data_works = get_data(product_id=None, count=None)
        # Get Work Title from the ID
        title = data_works.loc[data_works['work_id'] == product_id, 'title'].iloc[0]
        # create bags of words
        data_similarities = get_data_similarities(data_works)

        # List of purchases of works
        # data_purchases = get_data_users_purchases(user_id=None, count=None)
        # add ratings/weight regarding the purchases of the products
        #data_with_ratings = add_ratings_from_purchases(data_works, data_purchases)
        # add score to data
        #rename_data_with_score = get_data_with_score(data_similarities, data_with_ratings)

        # create cosine similarities matrix
        cosine_sim, indices = model_vectorization_cosine_similarities(data_works, data_similarities, stopwords_french, typeOfVec = "CountVec")

        predictOutput = model_content_recommender(title, cosine_sim, data_works, indices, limit=4, with_score=False)

        ## Transform to JSON
        rec_df_rating_json = predictOutput.to_json(orient='records')

        return rec_df_rating_json

    #except Exception as error:
    #    return {'error': error}

@app.get("/getRec/contentVec/createIndex", tags=["getRecContentVectorCreateIndex"])
async def get_rec_content_vectordb_init(data_work_type:typeOfDataWork):
    #try:
        # List of works
        data_works = get_data(product_id=None, count=None)
        # create bag of words
        data_similarities = get_data_similarities(data_works)
        # get only bag of words and convert it to dictionnary
        data_similarities["id"] = data_similarities["work_id"].astype(str)
        # Transform list of works into dictionnary
        data_similarities_dict = data_similarities.to_dict(orient='records')
        data_similarities_prepared_for_vectors = data_similarities[["id","bag_of_words"]].to_dict(orient='records')

        index, model, total_vectors = model_vector_indexing(data_similarities_dict, data_similarities_prepared_for_vectors)

        endpoint_response = {
            "message": f"OK, Vector Index was created and total of {total_vectors} vectors were added to the index"
        }

        return endpoint_response

    #except Exception as error:
    #    return {'error': error}

@app.get("/getRec/contentVec/{product_id}/{count}", tags=["getRecContentVectorDb"])
async def get_rec_content_vectordb(data_work_type:typeOfDataWork, product_id: int, count: int):
    #try:
        # List of works
        data_works = get_data(product_id=None, count=None)
        # Get Work Title from the ID
        title = data_works.loc[data_works['work_id'] == product_id, 'title'].iloc[0]
        # create bag of words
        data_similarities = get_data_similarities(data_works)
        # get only bag of words and convert it to dictionnary
        data_similarities["id"] = data_similarities["work_id"].astype(str)
        # Transform list of works into dictionnary
        data_similarities_dict = data_similarities.to_dict(orient='records')
        data_similarities_prepared_for_vectors = data_similarities[["id","bag_of_words"]].to_dict(orient='records')

        print("title:",title)
        #print(data_works[:3])
        #print(data_similarities_prepared_for_vectors[:3])

        predictOutput = model_content_recommender_vectors(data_similarities_dict, data_similarities_prepared_for_vectors, title, count)

        ## Transform to JSON
        rec_df_json = json.dumps(predictOutput)

        return rec_df_json

    #except Exception as error:
    #    return {'error': error}


@app.get("/getRec/collaborative/{data_work_type}/{user_id}/{count}", tags=["getRecCollaborative"])
async def get_rec_collaborative(data_work_type:typeOfDataWork, user_id: int, count: int):
    #try:
        # List of works
        data_works = get_data(product_id=None, count=None)

        # List of works purchased by users - exclusion of works products buy previously by the user - to not rec those to him
        data_purchases = get_data_users_purchases(user_id=None, count=None)

        predictOutput = predict_items_from_user_api(data_work_type, data_works, data_purchases, user_id, count)

        return predictOutput



origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    #uvicorn.run('app:app', host='127.0.0.1', port=80)
    uvicorn.run('app:app', host='0.0.0.0', port=6061)

