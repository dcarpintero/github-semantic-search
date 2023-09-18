from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from dotenv import load_dotenv
import os
import logging
import pandas as pd
import weaviate
import json

def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""
    
    load_dotenv()
    openapi_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not openapi_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    if not weaviate_url:
        raise EnvironmentError("WEAVIATE_URL environment variable not set.")
    
    if not weaviate_api_key:
        raise EnvironmentError("WEAVIATE_API_KEY environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"OPENAI_API_KEY": openapi_key, "WEAVIATE_URL": weaviate_url, "WEAVIATE_API_KEY": weaviate_api_key}


def index_data():
    """Index Data"""
    #file_name = "langchain-github-issues-2021-09-22.jsonl"
    #df = pd.read_json(file_name, orient="records", lines=True)
    df = pd.read_pickle("./data-pipeline/langchain-github-issues-2023-09-18.pkl")

    #print(df)

    #logging.info("Indexing data from Documents...")
    #embeddings = OpenAIEmbeddings()
    #vector_store = Weaviate.from_documents(documents=docs, embedding=embeddings)
    #logging.info("Indexing data from Documents - (OK)")
    
    #return vector_store
    openapi_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key), 
        additional_headers={"X-OpenAI-Api-Key": openapi_key})
    
    client.schema.delete_class("GitHubIssue")
    
    class_obj = {
    "class": "GitHubIssue",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        }
    }
    }

    client.schema.create_class(class_obj)

    current_issue = None

    # Iterate through each row of data
    try:
        with client.batch as batch:  # Initialize a batch process
            batch.batch_size = 100
            for issue in df.itertuples():
                current_issue = issue

                properties = {
                    "title": issue.title,
                    "url": issue.url,
                    "labels": issue.labels,
                    "description": issue.description,
                }

                batch.add_data_object(data_object=properties, class_name="GitHubIssue")
    except Exception as e:
        print(f"something happened {e}. Failure at {current_issue}")

    #some_objects = client.data_object.get()
    #print(json.dumps(some_objects))

    # https://weaviate.io/developers/weaviate/tutorials/query
    response = (
        client.query
        .get("GitHubIssue", ["description", "title", "labels"])
        .with_near_text({"concepts": ["documentation"]})
        .with_limit(2)
        .do()
    )

    print(json.dumps(response, indent=4))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()

        index_data()
    except EnvironmentError as ee:
        logging.error(f"Environment Error: {ee}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected Error: {ex}")
        raise