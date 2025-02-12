# Run this before you launch the script: export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import argparse
from pathlib import Path
import ollama
import sqlite3
import logging
import logging
import chromadb
import uvicorn

from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


ALLOWED_EXTENSIONS = [".jpg", ".jpeg"]


def setup_logger(name, log_file, level=logging.DEBUG):
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)  # Add console output

    return logger


class Descriptor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.p.resolve()
        self.output = args.o or self.root
        self.db_desc = self.output / "descriptions.sql"
        self.chroma_client = chromadb.PersistentClient(path=str(self.output))
        self.collection = self.chroma_client.get_or_create_collection(name="image_descriptions")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger = setup_logger("Descriptor", "descriptor.log", level=logging.DEBUG)
        self.logger.debug(f"Root: {self.root}")
        self.output.mkdir(exist_ok=True)
        self.init_db()

    def process_images(self):
        self.iterate_folder(self.root)

    def query_interactive(self, top_k=5):
        self.logger.debug("Number of images in db: " + str(self.collection.count()))

        while True:
            text = input("Query: ").strip()
            query_embedding = self.model.encode(text).tolist()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
            metadatas = results.get("metadatas", [[]])[0]
            similarities = results.get("distances", [[]])[0]

            if metadatas:
                for i, (meta, similarity) in enumerate(zip(metadatas, similarities), start=1):
                    image_path = str(self.root / meta["path"])
                    print(f"{i}. similarity: {similarity:.4f}, {image_path}")

    def query(self, query, top_k=5):
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        metadatas = results.get("metadatas", [[]])[0]
        similarities = results.get("distances", [[]])[0]

        if not metadatas:
            return {"results": []}

        response = {
            "results": [
                {
                    "image_path": str(self.root / meta["path"]),
                    "distance": similarity,
                }
                for meta, similarity in zip(metadatas, similarities)
            ]
        }
        return response

    def create_embeddings(self, overwrite=False):
        collection = self.chroma_client.get_collection(name="image_descriptions")
        entries = self.get_all_entries()

        existing = collection.get(include=["metadatas"])
        existing_ids = (
            {meta["path"] for meta in existing["metadatas"] if meta}
            if existing and "metadatas" in existing
            else set()
        )

        for entry in entries:
            if not overwrite and entry["path"] in existing_ids:
                self.logger.debug(f"Skipping {entry['path']}, embedding already exists.")
                continue

            self.logger.debug(f"Computing embedding for {entry['path']}")
            embedding = self.model.encode(entry["description"]).tolist()

            # If overwriting, remove existing before adding new embedding
            if overwrite and entry["path"] in existing_ids:
                collection.delete(where={"path": entry["path"]})

            collection.add(
                ids=[entry["path"]],
                embeddings=[embedding],
                metadatas=[{"path": entry["path"], "description": entry["description"]}],
            )
        self.logger.debug("Embeddings created")

    def init_db(self):
        conn = sqlite3.connect(self.db_desc)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS files (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            path TEXT NOT NULL UNIQUE,
                            description TEXT
                        )""")
        conn.commit()
        conn.close()

    def get_all_entries(self):
        conn = sqlite3.connect(self.db_desc)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def path_exists(self, path: Path):
        conn = sqlite3.connect(self.db_desc)
        cursor = conn.cursor()
        path_rel = str(self.relative_path(path))
        cursor.execute("SELECT 1 FROM files WHERE path = ?", (path_rel,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def add_path(self, path: Path, description: str):
        if self.path_exists(path):
            return False
        conn = sqlite3.connect(self.db_desc)
        cursor = conn.cursor()
        path_rel = str(self.relative_path(path))
        cursor.execute(
            "INSERT INTO files (path, description) VALUES (?, ?)", (path_rel, description)
        )
        conn.commit()
        conn.close()
        return True

    def relative_path(self, path: Path):
        return path.relative_to(self.root)

    def iterate_folder(self, path: Path):
        for item in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            if item.is_dir():
                self.logger.debug(f"Processing folder {self.relative_path(item)}")
                self.iterate_folder(item)
            else:
                if item.suffix.lower() in ALLOWED_EXTENSIONS:
                    self.logger.debug(f"Processing image {self.relative_path(item)}")
                    if not self.path_exists(item):
                        self.process_image(item)

    def process_image(self, image: Path):
        try:
            response = ollama.chat(
                model="llama3.2-vision:11b",
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image in great detail. Don't do lists, just create a neutral but precise description.",
                        "images": [image],
                    }
                ],
            )
        except Exception as e:
            self.logger.error(
                f"Couldn't compute description for image {self.relative_path(image)}: {e}"
            )
            return
        desc = response["message"]["content"]
        self.add_path(image, desc)
        self.logger.debug(f"Description: {desc}")

    def create_server(self):
        app = FastAPI()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/search")
        def search_images(
            query: str = Query(..., description="Search query"),
            top_k: int = Query(15, description="Number of results"),
        ):
            return self.query(query, top_k)

        return app

    def launch_server(self, port: int = 8000):
        app = self.create_server()
        uvicorn.run(app, host="127.0.0.1", port=port)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Image Search")
    parser.add_argument(
        "mode",
        choices=["descriptions", "embeddings", "query", "server"],
        help="Mode of operation. descriptions: process images and create descriptions, embeddings: create embeddings, query: query the database interactively, server: launch a server",
    )
    parser.add_argument(
        "-p", type=Path, default=Path("."), help="Path to input directory with images"
    )
    parser.add_argument(
        "-o",
        type=Path,
        default=None,
        help="Output directory to save embeddings, defaults to input directory",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    d = Descriptor(args)
    if args.mode == "descriptions":
        d.process_images()
    elif args.mode == "embeddings":
        d.create_embeddings(overwrite=False)
    elif args.mode == "query":
        d.query_interactive()
    elif args.mode == "server":
        d.launch_server()
    else:
        d.logger.error("Invalid mode!")


if __name__ == "__main__":
    main()
