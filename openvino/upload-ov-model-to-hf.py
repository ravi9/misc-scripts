import argparse
import os
from huggingface_hub import HfApi, HfFolder, create_repo
from huggingface_hub.utils import HfHubHTTPError

def main():
    parser = argparse.ArgumentParser(description="Upload OpenVINO model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local OpenVINO model directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository ID in format 'username/modelname'")
    parser.add_argument("--commit_message", type=str, default="Upload OpenVINO model",
                        help="Commit message for upload")
    parser.add_argument("--private", action="store_true",
                        help="Create private repository (default: public)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token (optional)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.isdir(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist or is not a directory")
    
    # Check for required files
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_path, file)):
            print(f"Warning: Missing file: {file} in model directory")
    
    # Get token
    token = args.token if args.token else HfFolder.get_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login`")
    
    api = HfApi(token=token)
    repo_exists = False
    
    # Try to create repository
    try:
        print(f"Attempting to create repository: {args.repo_id}")
        create_repo(
            repo_id=args.repo_id,
            private=args.private,
            exist_ok=False,  # Force exception if repo exists
            token=token
        )
        print(f"Successfully created {'private' if args.private else 'public'} repository")
    except HfHubHTTPError as e:
        if e.response.status_code == 409:  # Repository already exists
            repo_exists = True
            print(f"Repository already exists: {args.repo_id}")
            print("Will overwrite existing files...")
        else:
            print(f"Error creating repository: {e}")
            return
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload files using HfApi (overwrites existing files)
    try:
        print("Uploading model files...")
        api.upload_folder(
            folder_path=args.model_path,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            allow_patterns=["*"],  # Upload all files
        )
        
        print("\nUpload completed successfully!")
        print(f"Repository URL: https://huggingface.co/{args.repo_id}")
        print(f"Visibility: {'Private' if args.private else 'Public'}")
        if repo_exists:
            print("Existing repository was updated (files overwritten)")
        else:
            print("New repository created with model files")
    except Exception as e:
        print(f"Error uploading model: {e}")
        return

if __name__ == "__main__":
    main()
