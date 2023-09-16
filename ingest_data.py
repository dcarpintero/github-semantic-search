from langchain.document_loaders import GitHubIssuesLoader
from dotenv import load_dotenv
import os
import logging


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""
    
    load_dotenv()
    
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    if not github_token:
        raise EnvironmentError("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")
    
    logging.info("Environment variables loaded.")
    return {"GITHUB_PERSONAL_ACCESS_TOKEN": github_token}


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


def load_data(loader: GitHubIssuesLoader) -> []:
    """Load Data from GitHub Repository"""

    logging.info("Loading data from Github: %s", loader.repo)
    docs = loader.load()
        
    return docs


# def index_data(docs: []) -> VectorStoreIndex:
#    """Index Data from Documents"""
#
#    logging.info("Indexing data from Github.")
#    index = VectorStoreIndex()
#    index.index_documents(docs)
#    return index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()

        loader = initialize_github_loader()
        load_and_index_data(loader)
    except Exception as ex:
        logging.error("Unexpected Error: %s", ex)
        raise ex