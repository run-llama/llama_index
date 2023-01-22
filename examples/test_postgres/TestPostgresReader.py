from __future__ import absolute_import

# My OpenAI Key
import os
os.environ['OPENAI_API_KEY'] = ""

from gpt_index.readers.postgresql import PostgreSQLReader

documents = PostgreSQLReader(
    host = "localhost", # My Database Host
    port = "5432", # My Database Port
    user = "postgres", # My Database User
    password = "FakeExamplePassoword", # My Database Password
    dbname = "postgres", # My Database Name
    max_docs = 1000 # Maximum number of documents to load
).load_data(
    # The bellow example query returns a list of Document objects
    # with concatenated text from the name and age columns
    # from the users table where the age is greater than or equal to 18
    query = f"""
        SELECT
            CONCAT(name, ' is ', age, ' years old.') AS text
        FROM public.users
        WHERE age >= 18
        """
)

# Print type(documents) and documents
# type(documents) must return <class 'list'>
print(type(documents))

# documents must return a list of Document objects
print(documents)