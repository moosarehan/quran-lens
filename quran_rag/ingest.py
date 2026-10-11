import argparse
from .config import DATA
from .loader import TanzilLoader
from .store import VerseStore
from .retriever import HybridRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    documents = TanzilLoader(DATA).load()
    store = VerseStore()
    store.ingest(documents)
    print(f"Stored {len(documents)} ayahs in SQLite", flush=True)
    if not args.sql_only:
        retriever = HybridRetriever(store)
        try:
            retriever.index(args.batch_size)
        finally:
            retriever.close()

if __name__ == "__main__":
    main()
