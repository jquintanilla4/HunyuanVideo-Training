from huggingface_hub import HfApi
api = HfApi()

api.upload_large_folder(
    repo_id="xxx/xxx", # Change to your own repo ID
    repo_type="dataset", # Change to either a model or dataset, depending on your use case
    folder_path="data/small", # Change to your own folder path
)