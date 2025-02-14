import argparse
import logging
from pathlib import Path
import sqlite3
import glob
from typing import Optional

import ollama
import chromadb
import uvicorn
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class SemanticPipeline:
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg"]

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.p.resolve()  # All paths in db are relative to this
        self.output = args.o or self.root  # Where to save semantic information
        self.glob_pattern = args.g
        self.db_desc = self.output / "descriptions.sql"
        self.chroma_client = chromadb.PersistentClient(path=str(self.output))
        self.collection = self.chroma_client.get_or_create_collection(name="image_descriptions")
        self.model_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        self.model_img_text = "llama3.2-vision:11b"  # ollama model name
        self.logger = setup_logger("", "log.log", level=logging.INFO)
        self.output.mkdir(exist_ok=True)
        self.init_db()

    def show_stats(self):
        self.logger.info("Number of embeddings in db: " + str(self.collection.count()))

    def query_interactive(self, top_k=5):
        self.show_stats()

        while True:
            text = input("Query: ").strip()
            query_embedding = self.model_embeddings.encode(text).tolist()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            metadatas = results.get("metadatas", [[]])[0]
            similarities = results.get("distances", [[]])[0]

            if metadatas:
                for i, (meta, similarity) in enumerate(zip(metadatas, similarities), start=1):
                    image_path = str(self.root / meta["path"])
                    print(f"{i}. similarity: {similarity:.4f}, {image_path}")

    def query(self, query: str, top_k=5):
        query_embedding = self.model_embeddings.encode(query).tolist()
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

    def get_similar_images(self, path: Path, top_k=5):
        desc = self.get_description(path)
        self.logger.info(f"Querying for path: {path}")
        self.logger.info(f"Description: {desc}")

        if not desc:
            return {"results": []}

        return self.query(desc, top_k)

    def create_embeddings(self, overwrite=False):
        collection = self.chroma_client.get_collection(name="image_descriptions")
        entries = self.get_all_entries()

        existing = collection.get(include=["metadatas"])
        existing_ids = (
            {meta["path"] for meta in existing["metadatas"] if meta}
            if existing and "metadatas" in existing
            else set()
        )

        n_added = 0
        for entry in entries:
            if not overwrite and entry["path"] in existing_ids:
                continue

            self.logger.info(f"Computing embedding for {entry['path']}")
            embedding = self.model_embeddings.encode(entry["description"]).tolist()

            if overwrite and entry["path"] in existing_ids:
                collection.delete(where={"path": entry["path"]})

            collection.add(
                ids=[entry["path"]],
                embeddings=[embedding],
                metadatas=[{"path": entry["path"], "description": entry["description"]}],
            )
            n_added += 1

        self.logger.info(f"Embeddings created: {n_added}")
        self.show_stats()

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

    def path_exists_in_db(self, path: Path):
        conn = sqlite3.connect(self.db_desc)
        cursor = conn.cursor()
        path_rel = str(self.relative_path(path))
        cursor.execute("SELECT 1 FROM files WHERE path = ?", (path_rel,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def get_description(self, path: Path) -> str | None:
        conn = sqlite3.connect(self.db_desc)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        path_rel = str(self.relative_path(path))
        cursor.execute("SELECT description FROM files WHERE path = ?", (path_rel,))
        row = cursor.fetchone()
        conn.close()
        return row["description"] if row else None

    def save_description(self, path: Path, description: str):
        if self.path_exists_in_db(path):
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

    def process_images(self):
        pattern = str(self.root / self.glob_pattern) if self.glob_pattern else None
        self.iterate_folder(self.root, pattern)

    def iterate_folder(self, path: Path, pattern: Optional[str] = None):
        if not pattern:
            pattern = str(self.root / "**/*")

        # We use iglob here to not load all file paths into memory
        items = glob.iglob(pattern, recursive=True)
        for path_str in items:
            item = Path(path_str)
            if item.is_file() and item.suffix.lower() in self.ALLOWED_EXTENSIONS:
                if not self.path_exists_in_db(item):
                    self.logger.info(f"Processing {self.relative_path(item)}")
                    self.process_image(item)

    def process_image(self, image: Path):
        try:
            response = ollama.chat(
                model=self.model_img_text,
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image in great detail. "
                        "Don't do lists, just create a neutral but precise description.",
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
        self.save_description(image, desc)
        # self.logger.info(f"Description: {desc}")

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

        @app.get("/similar")
        def similar_images(
            path: str = Query(..., description="Path to image"),
            top_k: int = Query(15, description="Number of results"),
        ):
            return self.get_similar_images(Path(path), top_k)

        return app

    def launch_server(self, port: int = 8000):
        self.show_stats()
        app = self.create_server()
        uvicorn.run(app, host="127.0.0.1", port=port)

    def run(self):
        if self.args.mode == "descriptions":
            self.process_images()
        elif self.args.mode == "embeddings":
            self.create_embeddings()
        elif self.args.mode == "query":
            self.query_interactive()
        elif self.args.mode == "server":
            self.launch_server()
        else:
            self.logger.error("Invalid mode!")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Image Search")
    parser.add_argument(
        "mode",
        choices=["descriptions", "embeddings", "query", "server"],
        help="Mode of operation:\n"
        "descriptions: process images and create descriptions\n"
        "embeddings: create embeddings\n"
        "query: query the database interactively\n"
        "server: launch backend server for querying at 127.0.0.1:8000",
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
    parser.add_argument(
        "-g",
        type=str,
        default=None,
        help='Glob pattern to filter folders to scan, relative to input directory. Example: "images/2024/**"',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    d = SemanticPipeline(args)
    d.run()


if __name__ == "__main__":
    main()
