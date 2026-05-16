# index.py — reads data/*.md, embeds with Ollama, stores in Chroma
import os
import glob
import chromadb
import ollama

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "about_me"
EMBED_MODEL = "nomic-embed-text"

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks of roughly chunk_size characters,
    breaking on paragraph boundaries when possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

def main():
    client = chromadb.PersistentClient(path=DB_DIR)

    # Wipe and recreate so re-running gives a clean index
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    if not files:
        print(f"No markdown files found in {DATA_DIR}/")
        return

    doc_id = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        print(f"{path}: {len(chunks)} chunks")

        for chunk in chunks:
            # Get embedding from Ollama
            response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
            embedding = response["embedding"]

            collection.add(
                ids=[str(doc_id)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": os.path.basename(path)}],
            )
            doc_id += 1

    print(f"\nIndexed {doc_id} chunks into '{COLLECTION}'.")

if __name__ == "__main__":
    main()