# Products Recommendation System API

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1%2B-lightgrey)
![FastAPI Version](https://img.shields.io/badge/fastapi-0.115%2B-green)
![Solara Version](https://img.shields.io/badge/solara-1.41%2B-orange)
![Spotlight Version](https://img.shields.io/badge/spotlight-0.1.6%2B-red)
## Project Overview

This project aims to build a product recommendation system that predicts relevant products for users based on their 
previous interactions - COLLABORATIVE FILTERING - and without interactions - ITEM BASED FILTERING -. 

The recommendations are served via a REST API, with a Web UI interface to test and visualize results. 
The project leverages various machine learning libraries and frameworks, including PyTorch and Spotlight, to train and deploy 
recommendation models.

## Features

- **Product Recommendations**: Predicts and provides personalized product recommendations for users.
- **REST API**: An API built with FastAPI to serve predictions, allowing easy integration with other applications.
- **Interactive Web UI**: A web interface built with Solara (React-based) for testing and exploring recommendations.

## Technologies Used

- **Python**: Core language for model development, API, and backend logic.
- **PyTorch**: Used as the backbone framework for building and training the recommendation model.
- **Spotlight**: A recommendation library based on PyTorch, used to simplify building and training collaborative filtering models.
- **FastAPI**: A high-performance API framework used to expose the recommendation model for external access.
- **Solara**: A React-based framework for building a Web UI, allowing users to interact with the recommendation model and visualize results.

## How to use the Recommendation System

### Manage Data
- **Exploration**
- **Preparation**
- **Evaluation**

![Rec Sys Explore Data](readmeAssets/ml-recsys-explore.jpg)

### Recommend Products based on similarities of Products - Content based Filtering
#### 2 methods availables for vectorization of bags of words:
- **CountVectorizer**
- **TfIdfVectorizer**

#### Use of the cosine similarities matrix

```solara run application/website/01-ml-recsys-content-based.py```

![Rec Sys Content Based - Cold Start](readmeAssets/ml-recsys-coldstart.jpg)

### Create the Machine Learning Model - PyTorch Model
```solara run application/website/02-ml-create-model.py```

Path of the model created
```model/movies_users_rating_model.pth```


### Display the recommendation products to the user - Collaborative Filtering
#### Based on the machine learning model - PyTorch Model
```solara run application/website/03-ml-display-rec.py```

![Rec Sys User Based](readmeAssets/ml-recsys-users-ratings.jpg)

### To Come!!

- API for the Rec Sys
- Use of a Vector Database instead of a static Pytorch model