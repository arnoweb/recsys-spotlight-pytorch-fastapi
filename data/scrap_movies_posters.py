import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# Path to your CSV file
file_path = '/Users/a.breton/digital_projects/machine-learning/rec-shows-spotlight-fastapi/data/movies_clean.csv'
movie_titles = pd.read_csv(file_path, usecols=['title'])['title'].tolist()

#print(movie_titles)

# Base URL format
base_url = "https://www.mposter.com/"

session = requests.Session()

session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

cookies = {
    'PHPSESSID': '624912e002b65d4aceb8476f8c96b180'  # Replace with your actual PHPSESSID
}

# Function to format the movie title into the URL format
def format_movie_title(title):
    return title.lower().replace(" ", "-") + "-movie-poster.html"


# Loop through each movie title
for title in movie_titles:
    # Construct the URL
    url = base_url + format_movie_title(title)
    print(f"Processing: {url}")

    try:
        # Send a GET request to the URL
        response = session.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the image tag with the appropriate attribute (adjust if necessary)
            container = soup.find('div', class_='icerikbox')
            image = container.find('img')

            # Check if the image tag was found and has a 'src' attribute
            if image and 'src' in image.attrs:
                image_url = image['src']
                image_url_https = image_url.replace('http://', 'https://')
                print(f"Found image for '{title}': {image_url_https}")

                # Download the image
                img_data = session.get(image_url_https, stream=True)

                if img_data.status_code == 200:
                    filename = f"{title.replace(' ', '_').lower()}_poster.jpg"
                    with open('movies_posters/' + filename, "wb") as file:
                        for chunk in img_data.iter_content(1024):  # Write data in chunks
                            file.write(chunk)
                    print(f"Image saved as '{filename}'")
                else:
                    print("Failed to download image. Status code:", img_data.status_code)
            else:
                print(f"No poster image found for '{title}'")
        else:
            print(f"Failed to retrieve page for '{title}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred for '{title}': {e}")