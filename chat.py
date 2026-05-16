# chat.py — retrieve relevant chunks, ask llama3.2 with context
import chromadb
import ollama

DB_DIR = "chroma_db"
COLLECTION = "about_me"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:latest"
TOP_K = 4

SYSTEM_PROMPT = """You are a helpful assistant on Sabbir's personal website.
You answer questions about Sabbir based ONLY on the context provided below.
If the answer isn't in the context, say you don't know — don't make things up.
Keep answers concise and conversational.

Context about Sabbir:
{context}
"""

def retrieve(question, collection):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=question)
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=TOP_K,
    )
    return results["documents"][0]  # list of chunk strings

def ask(question, collection):
    chunks = retrieve(question, collection)
    context = "\n\n---\n\n".join(chunks)
    system = SYSTEM_PROMPT.format(context=context)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"], chunks

def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)

    print("Chat with the bot. Ctrl+C to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            answer, chunks = ask(q, collection)
            print(f"\nBot: {answer}\n")
            # Uncomment to debug what was retrieved:
            # print("--- retrieved chunks ---")
            # for c in chunks:
            #     print(c[:120] + "...")
            # print()
        except KeyboardInterrupt:
            print("\nBye.")
            break

if __name__ == "__main__":
    main()