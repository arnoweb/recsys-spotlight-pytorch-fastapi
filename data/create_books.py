import pandas as pd

# Define the list of famous books
books = [
    {
        "work_id": 1,
        "title": "To Kill a Mockingbird",
        "description": "A novel by Harper Lee published in 1960.",
        "genre_1": "Fiction",
        "genre_2": "Classic",
        "genre_3": "Southern Gothic",
        "author": "Harper Lee",
        "year": 1960,
        "url": "https://covers.openlibrary.org/b/isbn/9780060935467-L.jpg"
    },
    {
        "work_id": 2,
        "title": "1984",
        "description": "A dystopian novel by George Orwell published in 1949.",
        "genre_1": "Dystopian",
        "genre_2": "Science Fiction",
        "genre_3": "Political Fiction",
        "author": "George Orwell",
        "year": 1949,
        "url": "https://covers.openlibrary.org/b/isbn/9780451524935-L.jpg"
    },
    {
        "work_id": 3,
        "title": "The Great Gatsby",
        "description": "A novel by F. Scott Fitzgerald published in 1925.",
        "genre_1": "Fiction",
        "genre_2": "Classic",
        "genre_3": "Tragedy",
        "author": "F. Scott Fitzgerald",
        "year": 1925,
        "url": "https://covers.openlibrary.org/b/isbn/9780743273565-L.jpg"
    },
    {
        "work_id": 4,
        "title": "The Catcher in the Rye",
        "description": "A novel by J.D. Salinger published in 1951.",
        "genre_1": "Fiction",
        "genre_2": "Classic",
        "genre_3": "Coming-of-age",
        "author": "J.D. Salinger",
        "year": 1951,
        "url": "https://covers.openlibrary.org/b/isbn/9780316769488-L.jpg"
    },
    {
        "work_id": 5,
        "title": "The Hobbit",
        "description": "A fantasy novel by J.R.R. Tolkien published in 1937.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Children's Literature",
        "author": "J.R.R. Tolkien",
        "year": 1937,
        "url": "https://covers.openlibrary.org/b/isbn/9780618002214-L.jpg"
    },
    {
        "work_id": 6,
        "title": "Pride and Prejudice",
        "description": "A novel by Jane Austen published in 1813.",
        "genre_1": "Romance",
        "genre_2": "Classic",
        "genre_3": "Social Commentary",
        "author": "Jane Austen",
        "year": 1813,
        "url": "https://covers.openlibrary.org/b/isbn/9780199535569-L.jpg"
    },
    {
        "work_id": 7,
        "title": "Moby-Dick",
        "description": "A novel by Herman Melville published in 1851.",
        "genre_1": "Adventure",
        "genre_2": "Classic",
        "genre_3": "Epic",
        "author": "Herman Melville",
        "year": 1851,
        "url": "https://covers.openlibrary.org/b/isbn/9780199535569-L.jpg"
    },
    {
        "work_id": 8,
        "title": "War and Peace",
        "description": "A novel by Leo Tolstoy published in 1869.",
        "genre_1": "Historical Fiction",
        "genre_2": "Classic",
        "genre_3": "Epic",
        "author": "Leo Tolstoy",
        "year": 1869,
        "url": "https://covers.openlibrary.org/b/isbn/9780140447938-L.jpg"
    },
    {
        "work_id": 9,
        "title": "The Odyssey",
        "description": "An epic poem by Homer.",
        "genre_1": "Epic",
        "genre_2": "Classic",
        "genre_3": "Adventure",
        "author": "Homer",
        "year": -800,
        "url": "https://covers.openlibrary.org/b/isbn/9780140449112-L.jpg"
    },
    {
        "work_id": 10,
        "title": "Crime and Punishment",
        "description": "A novel by Fyodor Dostoevsky published in 1866.",
        "genre_1": "Crime",
        "genre_2": "Classic",
        "genre_3": "Psychological Fiction",
        "author": "Fyodor Dostoevsky",
        "year": 1866,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 11,
        "title": "The Adventures of Huckleberry Finn",
        "description": "A novel by Mark Twain published in 1884.",
        "genre_1": "Adventure",
        "genre_2": "Classic",
        "genre_3": "Satire",
        "author": "Mark Twain",
        "year": 1884,
        "url": "https://covers.openlibrary.org/b/isbn/9780486280615-L.jpg"
    },
    {
        "work_id": 12,
        "title": "The Grapes of Wrath",
        "description": "A novel by John Steinbeck published in 1939.",
        "genre_1": "Historical Fiction",
        "genre_2": "Classic",
        "genre_3": "Social Commentary",
        "author": "John Steinbeck",
        "year": 1939,
        "url": "https://covers.openlibrary.org/b/isbn/9780140186473-L.jpg"
    },
    {
        "work_id": 13,
        "title": "The Lord of the Rings",
        "description": "A fantasy novel by J.R.R. Tolkien published in 1954.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "J.R.R. Tolkien",
        "year": 1954,
        "url": "https://covers.openlibrary.org/b/isbn/9780618002252-L.jpg"
    },
    {
        "work_id": 14,
        "title": "The Divine Comedy",
        "description": "An epic poem by Dante Alighieri.",
        "genre_1": "Epic",
        "genre_2": "Classic",
        "genre_3": "Religious",
        "author": "Dante Alighieri",
        "year": 1320,
        "url": "https://covers.openlibrary.org/b/isbn/9780199540365-L.jpg"
    },
    {
        "work_id": 15,
        "title": "The Iliad",
        "description": "An epic poem by Homer.",
        "genre_1": "Epic",
        "genre_2": "Classic",
        "genre_3": "War",
        "author": "Homer",
        "year": -800,
        "url": "https://covers.openlibrary.org/b/isbn/9780140445928-L.jpg"
    },
    {
        "work_id": 16,
        "title": "The Brothers Karamazov",
        "description": "A novel by Fyodor Dostoevsky published in 1880.",
        "genre_1": "Philosophical",
        "genre_2": "Classic",
        "genre_3": "Psychological Fiction",
        "author": "Fyodor Dostoevsky",
        "year": 1880,
        "url": "https://covers.openlibrary.org/b/isbn/9780374528379-L.jpg"
    },
    {
        "work_id": 17,
        "title": "The Stranger",
        "description": "A novel by Albert Camus published in 1942.",
        "genre_1": "Existential",
        "genre_2": "Classic",
        "genre_3": "Philosophical",
        "author": "Albert Camus",
        "year": 1942,
        "url": "https://covers.openlibrary.org/b/isbn/9780679720201-L.jpg"
    },
    {
        "work_id": 18,
        "title": "The Metamorphosis",
        "description": "A novella by Franz Kafka published in 1915.",
        "genre_1": "Absurdist",
        "genre_2": "Classic",
        "genre_3": "Existential",
        "author": "Franz Kafka",
        "year": 1915,
        "url": "https://covers.openlibrary.org/b/isbn/9780553213690-L.jpg"
    },
    {
        "work_id": 19,
        "title": "The Picture of Dorian Gray",
        "description": "A novel by Oscar Wilde published in 1890.",
        "genre_1": "Gothic",
        "genre_2": "Classic",
        "genre_3": "Philosophical",
        "author": "Oscar Wilde",
        "year": 1890,
        "url": "https://covers.openlibrary.org/b/isbn/9780199535590-L.jpg"
    },
    {
        "work_id": 20,
        "title": "The Sun Also Rises",
        "description": "A novel by Ernest Hemingway published in 1926.",
        "genre_1": "Modernist",
        "genre_2": "Classic",
        "genre_3": "Expatriate",
        "author": "Ernest Hemingway",
        "year": 1926,
        "url": "https://covers.openlibrary.org/b/isbn/9780743297332-L.jpg"
    },
    {
        "work_id": 21,
        "title": "Brave New World",
        "description": "A dystopian novel by Aldous Huxley published in 1932.",
        "genre_1": "Dystopian",
        "genre_2": "Science Fiction",
        "genre_3": "Social Commentary",
        "author": "Aldous Huxley",
        "year": 1932,
        "url": "https://covers.openlibrary.org/b/isbn/9780060932149-L.jpg"
    },
    {
        "work_id": 22,
        "title": "Fahrenheit 451",
        "description": "A dystopian novel by Ray Bradbury published in 1953.",
        "genre_1": "Dystopian",
        "genre_2": "Science Fiction",
        "genre_3": "Social Commentary",
        "author": "Ray Bradbury",
        "year": 1953,
        "url": "https://covers.openlibrary.org/b/isbn/9781451673319-L.jpg"
    },
    {
        "work_id": 23,
        "title": "The Alchemist",
        "description": "A novel by Paulo Coelho published in 1988.",
        "genre_1": "Fiction",
        "genre_2": "Philosophical",
        "genre_3": "Adventure",
        "author": "Paulo Coelho",
        "year": 1988,
        "url": "https://covers.openlibrary.org/b/isbn/9780062315007-L.jpg"
    },
    {
        "work_id": 24,
        "title": "The Da Vinci Code",
        "description": "A mystery thriller by Dan Brown published in 2003.",
        "genre_1": "Mystery",
        "genre_2": "Thriller",
        "genre_3": "Conspiracy",
        "author": "Dan Brown",
        "year": 2003,
        "url": "https://covers.openlibrary.org/b/isbn/9780307277671-L.jpg"
    },
    {
        "work_id": 25,
        "title": "The Hunger Games",
        "description": "A dystopian novel by Suzanne Collins published in 2008.",
        "genre_1": "Dystopian",
        "genre_2": "Young Adult",
        "genre_3": "Science Fiction",
        "author": "Suzanne Collins",
        "year": 2008,
        "url": "https://covers.openlibrary.org/b/isbn/9780439023528-L.jpg"
    },
    {
        "work_id": 26,
        "title": "The Girl with the Dragon Tattoo",
        "description": "A crime novel by Stieg Larsson published in 2005.",
        "genre_1": "Crime",
        "genre_2": "Mystery",
        "genre_3": "Thriller",
        "author": "Stieg Larsson",
        "year": 2005,
        "url": "https://covers.openlibrary.org/b/isbn/9780307454546-L.jpg"
    },
    {
        "work_id": 27,
        "title": "The Kite Runner",
        "description": "A novel by Khaled Hosseini published in 2003.",
        "genre_1": "Fiction",
        "genre_2": "Historical",
        "genre_3": "Drama",
        "author": "Khaled Hosseini",
        "year": 2003,
        "url": "https://covers.openlibrary.org/b/isbn/9781594480003-L.jpg"
    },
    {
        "work_id": 28,
        "title": "The Night Circus",
        "description": "A fantasy novel by Erin Morgenstern published in 2011.",
        "genre_1": "Fantasy",
        "genre_2": "Magical Realism",
        "genre_3": "Romance",
        "author": "Erin Morgenstern",
        "year": 2011,
        "url": "https://covers.openlibrary.org/b/isbn/9780385534635-L.jpg"
    },
    {
        "work_id": 29,
        "title": "The Fault in Our Stars",
        "description": "A young adult novel by John Green published in 2012.",
        "genre_1": "Young Adult",
        "genre_2": "Romance",
        "genre_3": "Drama",
        "author": "John Green",
        "year": 2012,
        "url": "https://covers.openlibrary.org/b/isbn/9780142424175-L.jpg"
    },
    {
        "work_id": 30,
        "title": "The Goldfinch",
        "description": "A novel by Donna Tartt published in 2013.",
        "genre_1": "Fiction",
        "genre_2": "Literary",
        "genre_3": "Drama",
        "author": "Donna Tartt",
        "year": 2013,
        "url": "https://covers.openlibrary.org/b/isbn/9780316055437-L.jpg"
    },
    {
        "work_id": 31,
        "title": "The Name of the Wind",
        "description": "A fantasy novel by Patrick Rothfuss published in 2007.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Patrick Rothfuss",
        "year": 2007,
        "url": "https://covers.openlibrary.org/b/isbn/9780756404741-L.jpg"
    },
    {
        "work_id": 32,
        "title": "The Way of Kings",
        "description": "A fantasy novel by Brandon Sanderson published in 2010.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Brandon Sanderson",
        "year": 2010,
        "url": "https://covers.openlibrary.org/b/isbn/9780765326355-L.jpg"
    },
    {
        "work_id": 33,
        "title": "The Lies of Locke Lamora",
        "description": "A fantasy novel by Scott Lynch published in 2006.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Heist",
        "author": "Scott Lynch",
        "year": 2006,
        "url": "https://covers.openlibrary.org/b/isbn/9780553588941-L.jpg"
    },
    {
        "work_id": 34,
        "title": "The First Law Trilogy",
        "description": "A fantasy series by Joe Abercrombie published in 2006.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Joe Abercrombie",
        "year": 2006,
        "url": "https://covers.openlibrary.org/b/isbn/9781591025948-L.jpg"
    },
    {
        "work_id": 35,
        "title": "The Blade Itself",
        "description": "A fantasy novel by Joe Abercrombie published in 2006.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Joe Abercrombie",
        "year": 2006,
        "url": "https://covers.openlibrary.org/b/isbn/9781591025948-L.jpg"
    },
    {
        "work_id": 36,
        "title": "The Final Empire",
        "description": "A fantasy novel by Brandon Sanderson published in 2006.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Brandon Sanderson",
        "year": 2006,
        "url": "https://covers.openlibrary.org/b/isbn/9780765311788-L.jpg"
    },
    {
        "work_id": 37,
        "title": "The Eye of the World",
        "description": "A fantasy novel by Robert Jordan published in 1990.",
        "genre_1": "Fantasy",
        "genre_2": "Adventure",
        "genre_3": "Epic",
        "author": "Robert Jordan",
        "year": 1990,
        "url": "https://covers.openlibrary.org/b/isbn/9780812511819-L.jpg"
    },
    {
        "work_id": 38,
        "title": "The Shadow of the Torturer",
        "description": "A science fiction novel by Gene Wolfe published in 1980.",
        "genre_1": "Science Fiction",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Gene Wolfe",
        "year": 1980,
        "url": "https://covers.openlibrary.org/b/isbn/9780671571439-L.jpg"
    },
    {
        "work_id": 39,
        "title": "The Left Hand of Darkness",
        "description": "A science fiction novel by Ursula K. Le Guin published in 1969.",
        "genre_1": "Science Fiction",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Ursula K. Le Guin",
        "year": 1969,
        "url": "https://covers.openlibrary.org/b/isbn/9780441478125-L.jpg"
    },
    {
        "work_id": 40,
        "title": "The Dispossessed",
        "description": "A science fiction novel by Ursula K. Le Guin published in 1974.",
        "genre_1": "Science Fiction",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Ursula K. Le Guin",
        "year": 1974,
        "url": "https://covers.openlibrary.org/b/isbn/9780060936288-L.jpg"
    },
    {
        "work_id": 41,
        "title": "The Wind-Up Bird Chronicle",
        "description": "A novel by Haruki Murakami published in 1994.",
        "genre_1": "Fiction",
        "genre_2": "Magical Realism",
        "genre_3": "Mystery",
        "author": "Haruki Murakami",
        "year": 1994,
        "url": "https://covers.openlibrary.org/b/isbn/9780679775430-L.jpg"
    },
    {
        "work_id": 42,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 43,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 44,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 45,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 46,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 47,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 48,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 49,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    },
    {
        "work_id": 50,
        "title": "The Wind in the Willows",
        "description": "A children's novel by Kenneth Grahame published in 1908.",
        "genre_1": "Children's Literature",
        "genre_2": "Fantasy",
        "genre_3": "Adventure",
        "author": "Kenneth Grahame",
        "year": 1908,
        "url": "https://covers.openlibrary.org/b/isbn/9780486415871-L.jpg"
    }
]

# Create a DataFrame from the list of books
df = pd.DataFrame(books)

# Save the DataFrame to a CSV file
df.to_csv('books/products.csv', index=False)

print("CSV file 'products.csv' created successfully.")
