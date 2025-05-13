from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
api = HfApi()

api.upload_large_folder(
    repo_id=os.getenv("REPO_ID"), # Change to your own repo ID
    repo_type="dataset", # Change to either a model or dataset, depending on your use case
    folder_path="data/small", # Change to your own folder path
)