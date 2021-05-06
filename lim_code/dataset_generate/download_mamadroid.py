import requests
import pandas as pd
from glob import glob
from pathlib import Path
import configparser
from concurrent.futures import ThreadPoolExecutor

CONCURRENT_DOWNLOADS = 20
DATASET_DIR = Path("dataset_mamadroid")
OUTPUT_DIR = DATASET_DIR / "raw"
config = configparser.ConfigParser()
config.read(Path("~/.az").expanduser())
APIKEY = config["default"]["apikey"]


def main():
    for _list_file in glob("dataset_mamadroid/SamplesHash/*.txt"):
        download_list(_list_file)


def download_list(_list_file):
    _dir = OUTPUT_DIR / Path(_list_file).stem
    if not _dir.exists():
        _dir.mkdir(parents=True)

    df = pd.read_csv(_list_file, names=["sample", "sha256"])
    df["sha256"] = df["sha256"].str.strip()
    with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
        for _, (sample, sha256) in df.iterrows():
            _file = _dir / sample
            executor.submit(download_apk, _file, sha256)


def download_apk(_file, sha256):
    # curl -O --remote-header-name -G -d apikey=${APIKEY} -d sha256=${SHA256} https://androzoo.uni.lu/api/download
    r = requests.get(
        url="https://androzoo.uni.lu/api/download",
        params={"apikey": APIKEY, "sha256": sha256},
        stream=True)
    with _file.open(mode="wb") as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)


if __name__ == "__main__":
    main()
