import os
import shutil
import kagglehub

DATASET_ID = "ayushmandatta1/deepdetect-2025"

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models/swin/v1/checkpoints", exist_ok=True)

def download_dataset():
    print("Downloading dataset from KaggleHub...")
    src_path = kagglehub.dataset_download(DATASET_ID)
    print("Downloaded to cache:", src_path)

    target_dir = os.path.join("data", "deepdetect-2025_dddata")

    if os.path.exists(target_dir):
        print("Dataset already exists at:", target_dir)
        return

    print("Copying dataset to project data folder...")
    shutil.copytree(src_path, target_dir)

    print("Dataset copied to:", target_dir)
    print("Top-level folders:", os.listdir(target_dir))

def main():
    ensure_dirs()
    download_dataset()
    print("Setup complete.")

if __name__ == "__main__":
    main()