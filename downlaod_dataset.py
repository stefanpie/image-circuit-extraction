import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://osf.io/3qx7m/download"
DATA_DIR = Path("./data")

zip_path, _ = urllib.request.urlretrieve(DOWNLOAD_URL)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(DATA_DIR)
