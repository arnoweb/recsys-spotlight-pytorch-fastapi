import pandas as pd
import re

#https://www.kaggle.com/datasets/snehal1409/movielens
#it contains 100004 ratings and 1296 tag applications across 9125 movies. These data were created by 671 users between
# January 09, 1995 and October 16, 2016. This dataset was generated on October 17, 2016.

#https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset

# Load the CSV file
movies_df = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies.csv')
movies_with_summary_df = pd.read_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_imdb_summary.csv')

# Define a function to split 'movie_title' into 'title' and 'date'
def split_title_and_date(movie_title):
    # Use regex to capture the title and the year (YYYY)
    match = re.match(r'^(.*)\s\((\d{4})\)$', movie_title)
    if match:
        title, year = match.groups()
        return pd.Series([title, year])
    else:
        return pd.Series([movie_title, None])

# Apply the function to the 'movie_title' column
movies_df[['title', 'year']] = movies_df['title'].apply(split_title_and_date)

# Merge the two datasets on the 'title' column
movies_df = pd.merge(movies_df, movies_with_summary_df[['title', 'description','Poster_Link','Director']], on='title', how='left')

# Split the 'genres' column into at most 3 columns based on the '|' separator
genres_split = movies_df['genres'].str.split('|', n=2, expand=True)

# Rename the columns to 'genre_1', 'genre_2', 'genre_3'
genres_split.columns = ['genre_1', 'genre_2', 'genre_3']

# Concatenate the split genre columns back with the original dataframe (excluding the original 'genres' column)
movies_df = pd.concat([movies_df.drop(columns=['genres']), genres_split], axis=1)

# Move the 'Director' column to the end
date_col = movies_df.pop('Director')
movies_df['author'] = date_col

# Move the 'year' column to the end
date_col = movies_df.pop('year')
movies_df['year'] = date_col

# Move the 'Poster_Link' column to the end
date_col = movies_df.pop('Poster_Link')
movies_df['url'] = date_col

# Remove rows where the 'title' column contains a comma
#movies_df = movies_df[~movies_df['title'].str.contains(',', na=False)]

# Remove rows where the 'url' is empty
#movies_df = movies_df[movies_df['url'].ne('') & movies_df['url'].notna()]

# Save the result to a new CSV file
movies_df.to_csv('/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_clean.csv', index=False)