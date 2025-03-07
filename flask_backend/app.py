import base64
import io
import json
import os
import sys
import threading
import time
import logging
import boto3
from typing import Tuple, Dict, Any
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import altair as alt
from bokeh.plotting import figure
from bokeh.embed import json_item
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from jupyter_client import KernelManager
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import re

# Import fetch_s3_file from your s3_utils module
# from s3_utils import fetch_s3_file
from queue import Empty
import uvicorn
from queue import Empty
from datetime import datetime, date
import ast

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def fetch_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key):
    """
    Fetches a file from the specified S3 bucket, loads it into both Pandas and DuckDB.

    :param bucket_name: Name of the S3 bucket
    :param file_key: Key of the file in the S3 bucket
    :param aws_access_key_id: AWS access key ID
    :param aws_secret_access_key: AWS secret access key
    :param session_manager: The session manager that holds the DuckDB connection
    :param session_id: The session ID to get the DuckDB connection
    :param table_name: The name of the table to register the data in DuckDB
    :return: A tuple of download_path, pandas dataframe (as pandas code string), and success message for DuckDB
    """
    try:
        # Create an S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Define the local path to download the file
        download_path = os.path.join(os.getcwd(), file_key.split("/")[-1])

        # Download the file from S3
        s3.download_file(bucket_name, file_key, download_path)
        print(f"File downloaded successfully to {download_path}")

        # Initialize an empty variable to store the Pandas code string
        pandas_code = ""

        # retreive the session_id for this session
        session_id = session_manager.get_session

        # Get the DuckDB connection for this session
        duckdb_conn = session_manager.get_duckdb_connection(session_id)

        # Load into Pandas and register in DuckDB based on file extension
        file_extension = os.path.splitext(file_key)[1].lower()
        file_name = os.path.basename(file_key)

        # Replace special characters (including hyphens and periods) with underscores
        file_name = re.sub(r"[^a-zA-Z0-9_]", "_", file_name)

        # if file_extension == '.csv':
        #     pandas_code = f"{file_name} = pd.read_csv(r'''{download_path}''')"
        #     duckdb_conn.execute(f"CREATE TABLE {file_name} AS SELECT * FROM read_csv_auto('{download_path}')")
        # elif file_extension == '.parquet':
        #     pandas_code = f"{file_name} = pd.read_parquet(r'''{download_path}''')"
        #     duckdb_conn.execute(f"CREATE TABLE {file_name} AS SELECT * FROM parquet_scan('{download_path}')")
        # elif file_extension == '.json':
        #     pandas_code = f"{file_name} = pd.read_json(r'''{download_path}''')"
        #     duckdb_conn.execute(f"CREATE TABLE {file_name} AS SELECT * FROM read_json_auto('{download_path}')")
        # elif file_extension in ['.xls', '.xlsx']:
        #     pandas_code = f"{file_name} = pd.read_excel(r'''{download_path}''')"
        #     duckdb_conn.execute(f"CREATE TABLE {file_name} AS SELECT * FROM read_excel('{download_path}')")
        # else:
        #     raise ValueError(f"Unsupported file type: {file_extension}")

        # print(f"Table '{file_name}' successfully created in DuckDB from the file '{download_path}'")

        # Return both the pandas code (as a string) and the success message from DuckDB
        return download_path, pandas_code

    except Exception as e:
        print(f"Failed to fetch the file and register it in DuckDB and Pandas: {e}")
        raise


def extract_variable_from_code(code: str, variable_name: str) -> str:
    pattern = rf'{variable_name}\s*=\s*[\'"](.+?)[\'"]'
    match = re.search(pattern, code)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract {variable_name} from the code")


async def send_output(output_type: str, content: Any, block_id: str):
    message = {
        "type": "execute_response",
        "output_type": output_type,
        "content": content,
        "blockId": block_id,
    }
    return message


def send_error(error_message: str, block_id: str):
    message = {
        "type": "execute_response",
        "output_type": "error",
        "content": error_message,
        "blockId": block_id,
    }
    return message


@app.get("/test")
async def test_route():
    return {"message": "Let's see how it works!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
