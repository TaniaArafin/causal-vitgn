"""Download Higgs Twitter dataset from Stanford SNAP.

Run:
    python data/download.py
"""

import os
import sys
import urllib.request

URLS = {
    "higgs-social_network.edgelist.gz":
        "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz",
    "higgs-retweet_network.edgelist.gz":
        "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz",
    "higgs-reply_network.edgelist.gz":
        "https://snap.stanford.edu/data/higgs-reply_network.edgelist.gz",
    "higgs-mention_network.edgelist.gz":
        "https://snap.stanford.edu/data/higgs-mention_network.edgelist.gz",
    "higgs-activity_time.txt.gz":
        "https://snap.stanford.edu/data/higgs-activity_time.txt.gz",
}


def report_progress(blocknum: int, blocksize: int, totalsize: int):
    if totalsize <= 0:
        return
    bytes_done = blocknum * blocksize
    pct = min(100, bytes_done * 100 / totalsize)
    sys.stdout.write(f"\r  -> {pct:5.1f}%  ({bytes_done / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_file(url: str, dest: str):
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()


def main(out_dir: str = "./data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    for fname, url in URLS.items():
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f"Skipping {fname} (already exists)")
            continue
        try:
            download_file(url, path)
        except Exception as e:
            print(f"  ! Failed to download {fname}: {e}")

    print("\nDone. Files saved to:", out_dir)
    print("Next step: python data/preprocess.py")


if __name__ == "__main__":
    main()
