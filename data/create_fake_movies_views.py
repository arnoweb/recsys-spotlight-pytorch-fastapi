import pandas as pd
import random

# Load the movies and ratings data from the two CSV files
movies = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_clean.csv')
movies_users = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_users.csv')

# Assuming the user IDs and movie IDs are in the first column
user_ids = movies_users.iloc[:, 0].tolist()  # Get user IDs as a list
movie_ids = movies.iloc[:, 0].tolist()  # Get movie IDs as a list

# Parameters
num_records = 1000  # Change this to the desired number of records

# Generate data
data = {
    'user_id': [random.choice(user_ids) for _ in range(num_records)],
    'work_id': [random.choice(movie_ids) for _ in range(num_records)],
    'total_page_views': [random.randint(1, 30) for _ in range(num_records)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Group by user_id and movie_id, and sum total_page_views
grouped_df = df.groupby(['user_id', 'work_id'], as_index=False).agg({'total_page_views': 'sum'})

# Order the DataFrame by user_id
ordered_df = grouped_df.sort_values(by='user_id')

# Save to CSV
ordered_df.to_csv("/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_users_page_views.csv", index=False)