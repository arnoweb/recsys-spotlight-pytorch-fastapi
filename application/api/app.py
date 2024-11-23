from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.templating import Jinja2Templates

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
        "name": "createModel",
        "description": "Create the model based on data and ML with Spotlight implicit features",
    },
    {
        "name": "getModel",
        "description": "Operations with users. The **login** logic is also here.",
    },
    {
        "name": "getRecContent",
        "description": "Get a list of recommendated shows based on similar features of products - Content-based Filtering",
    },
    {
        "name": "getRecCollaborative",
        "description": "Get a list of recommendated shows for a user based on others users ratings/purchase - User-based Collaborative Filtering",
    },
]

app = FastAPI(title="ML API - Predict Rec Products", description="Get predicted recommendated products", version="0.0.1", openapi_tags=tags_metadata)


class Item(BaseModel):
    count: int = Field(default='3')
    user_name: int = Field(default='Arnaud Breton')

    #df: Json[Any] = Field(default='{"count": 3}')

@app.get("/")
async def root():
    return {"message": "Fastapi - Rec Sys based on Spotlight"}

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
async def getUsersPurchases():
    # List of users purchases
    data = get_data_users_purchases().to_json(orient='records')
    return data

@app.get("/usersRatings", tags=["getUsersRatings"])
async def usersRatings():
    # List of users purchases
    data = get_data_users-ratings().to_json(orient='records')
    return data

@app.get("/createModel", tags=["createModel"])
def create_model(request: Request):

    return

@app.get("/getModel", tags=["getModel"])
def get_model(request: Request):

    return

@app.get("/getRec/content/{product_id}/{count}", tags=["getRecContent"])
async def get_rec_content(product_id: int, count: int):
    try:
        # Rec type
        rec_type = 'content'

        # List of shows
        data_shows = get_data()

        predictOutput = predict_items_from_user_api(rec_type, data_shows, data_purchases, user_id, count)

        return predictOutput

    except Exception as error:
        return {'error': error}

@app.get("/getRec/collaborative/{user_id}/{count}", tags=["getRecCollaborative"])
async def get_rec_collaborative(user_id: int, count: int):
    try:
        # Rec type
        rec_type = 'collaborative'

        # List of shows
        data_shows = get_data()

        # List of shows purchased by users
        data_purchases = get_data_purchase()

        predictOutput = predict_items_from_user_api(rec_type, data_shows, data_purchases, user_id, count)

        return predictOutput

    except Exception as error:
        return {'error': error}



origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run('app:app', host='0.0.0.0', port=8765)
