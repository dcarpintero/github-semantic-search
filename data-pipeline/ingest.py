from langchain.document_loaders import GitHubIssuesLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from dotenv import load_dotenv
import os
import logging

def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""
    
    load_dotenv()
    
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    api_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN environment variable not set.")
    
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    if not weaviate_url:
        raise EnvironmentError("WEAVIATE_URL environment variable not set.")
    
    if not weaviate_api_key:
        raise EnvironmentError("WEAVIATE_API_KEY environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"GITHUB_TOKEN": github_token, "OPENAI_API_KEY": api_key, "WEAVIATE_URL": weaviate_url, "WEAVIATE_API_KEY": weaviate_api_key}


def initialize_github_loader() -> GitHubIssuesLoader:
    """Load GitHub issues with comments and labels from the langchain-ai/langchain repository."""	

    loader = GitHubIssuesLoader(
        repo="langchain-ai/langchain",
        include_prs=False,
        include_comments=False,
        include_labels=True,
        creator="RoderickVM",
    )

    return loader


def load_and_index_data(loader):
    """Load and Index Data from GitHub Repository"""

    docs = load_data(loader)
    
    logging.info("Loaded %s documents from Github.", len(docs))
    logging.info("*******************************************")
    logging.info(docs[0].page_content)
    logging.info("*******************************************")
    logging.info(docs[0].metadata)
    logging.info("*******************************************")
    logging.info(docs[0].metadata["labels"])
    logging.info("*******************************************")
    logging.info(docs[0].metadata["state"])

    index = index_data(docs)


def load_data(loader: GitHubIssuesLoader) -> []:
    """Load Data from GitHub Repository"""

    logging.info("Loading data from Github: %s", loader.repo)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logging.info("Loading data from Github - (OK)")
        
    return docs


def index_data(docs: []) -> Weaviate:
    """Index Data from Documents"""

    logging.info("Indexing data from Documents...")
    embeddings = OpenAIEmbeddings()
    vector_store = Weaviate.from_documents(documents=docs, embedding=embeddings)
    logging.info("Indexing data from Documents - (OK)")
    
    return vector_store

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()

        loader = initialize_github_loader()
        load_and_index_data(loader)
    except Exception as ex:
        logging.error("Unexpected Error: %s", ex)
        raise ex