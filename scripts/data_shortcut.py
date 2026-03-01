import os
import shutil
import kagglehub

kaggle_id = "ayushmandatta1/deepdetect-2025"

src_path = kagglehub.dataset_download(kaggle_id)
print("Kagglehub cache path:", src_path)

dst_root = os.path.join(os.getcwd(), "data", "deepdetect-2025")

if os.path.exists(dst_root):
    print("Already exists:", dst_root)
else:
    shutil.copytree(src_path, dst_root)
    print("Copied dataset to:", dst_root)

print("Top-level files:", os.listdir(dst_root)[:50])