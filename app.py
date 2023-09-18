"""
"""
import streamlit as st
import pandas as pd
import weaviate
import logging
import os

from dotenv import load_dotenv


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


@st.cache_resource(show_spinner=False)
def weaviate_client(openai_key: str, weaviate_url: str, weaviate_api_key: str):

    logging.info(f"Initializing Weaviate Client: '{weaviate_url}'")
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key), 
        additional_headers={"X-OpenAI-Api-Key": openai_key})
    
    return client


@st.cache_data
def search_github_issues(_w_client: weaviate.Client, query, max_results=10) -> pd.DataFrame:
    """Search GitHub Issues in Weaviate."""

    response = (
        _w_client.query
        .get("GitHubIssue", ["title", "labels", "description"])
        .with_near_text({"concepts": [query]})
        .with_limit(max_results)
        .do()
    )

    data = response["data"]["Get"]["GitHubIssue"]
    return  pd.DataFrame.from_dict(data, orient='columns')


env_vars = load_environment_vars()
w_client = weaviate_client(env_vars["OPENAI_API_KEY"], env_vars["WEAVIATE_URL"], env_vars["WEAVIATE_API_KEY"])

st.header("🦜 Github Semantic Search with Weviate 🔍")

with st.sidebar.expander("🐙 GITHUB-REPOSITORY", expanded=True):
    st.text_input(label='GITHUB-REPOSITORY', key='github_repo', label_visibility='hidden', value='langchain-ai/langchain', disabled=True)

with st.sidebar.expander("🔧 WEAVIATE-SETTINGS", expanded=True):
     max_results = st.sidebar.slider('Max Results', min_value=0, max_value=100, value=2, step=1)


query = st.text_input("Search in 'langchain-ai/langchain'", '')
if query:
    df = search_github_issues(w_client, query, max_results)
    st.dataframe(df, hide_index=True)