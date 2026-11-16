"""
Build the vector store from data/processed/products_clean.csv.

Run:  python src/build_index.py

Embeddings are computed locally (no API key). The build is atomic and stamps
the collection with the CSV hash, row count and embedding model, which the app
verifies at startup. See src/index.py.
"""

import shutil
import time

from tqdm import tqdm

import config
import index


def main() -> None:
    df = index.read_products()
    # Start from an empty store so the shipped database carries no dead rows from
    # earlier builds (the SQLite file must stay under GitHub's 100 MB limit).
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    print(f"Indexing {len(df)} products with {config.EMBED_MODEL}")
    bar = tqdm(total=len(df), unit="product")
    t0 = time.time()
    collection = index.build(progress=lambda done, total: bar.update(done - bar.n))
    bar.close()
    print(f"Done in {time.time() - t0:.0f}s. Collection '{config.COLLECTION_NAME}' holds "
          f"{collection.count()} vectors; manifest: {collection.metadata}")


if __name__ == "__main__":
    main()
