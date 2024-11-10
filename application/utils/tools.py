from bs4 import BeautifulSoup
import re

# Function to get info about the DataFrame and display its name
def df_info(df, df_name):
    print("=" * 40)  # Separator line
    print(f"DataFrame Name: {df_name}")
    print("=" * 40)  # Separator line
    df.info()  # Display DataFrame info
    print("\nFirst 10 Rows:")
    print(df.head(10))  # Display the first 10 rows
    #print("\nDescriptive Statistics:")
    #print(df.describe(include='all'))  # Display descriptive statistics
    print("\nColumns:")
    print(df.columns.tolist())  # List of column names
    print("\nData Types:")
    print(df.dtypes)  # Data types of each column
    print("\nMissing Values:")
    print(df.isnull().sum())  # Count of missing values



# Clean HTML tags into Description
def cleanHTML(data_html):
    clean_desc = BeautifulSoup(data_html, "lxml").text
    clean_desc = re.sub(r'[\'|\"|«|»|€|-|:|\?|3|5|16|10|224]', '', clean_desc)
    return clean_desc


# Function to repeat a word X times in a list
def repeat_word(word, times):
    words = [word] * times
    return words


# Création d'un système de notation en fonction des ventes de spectacles
def rating_to_show(total_purchase_by_product):
    if total_purchase_by_product in list(range(0, 2)):
        return 1
    elif total_purchase_by_product in list(range(2, 3)):
        return 2
    elif total_purchase_by_product in list(range(3, 4)):
        return 3
    elif total_purchase_by_product in list(range(4, 5)):
        return 4
    elif total_purchase_by_product >= 6:
        return 5
    else:
        return 0

# Création d'un système de notation en fonction des ventes de spectacles
def rating_to_movie(total_purchase_by_product):
    if total_purchase_by_product == 1:
        return 5
    else:
        return 0

# Création d'un système de notation en fonction des ventes de spectacles
def rating_to_allpurchases_of_movies(total_purchase_by_product):
    if total_purchase_by_product in list(range(0, 2)):
        return 1
    elif total_purchase_by_product in list(range(2, 3)):
        return 2
    elif total_purchase_by_product in list(range(3, 4)):
        return 3
    elif total_purchase_by_product in list(range(4, 6)):
        return 4
    elif total_purchase_by_product >= 6:
        return 5
    else:
        return 0

# Function to compute the shows weighted rating for each show
def weighted_rating(x, m, C):
    if m and C:
        v = x['rating_count']
        R = x['rating_average']
        # Compute the weighted score
        w = (v / (v + m) * R) + (m / (m + v) * C)
    else:
        w = False
    return w
