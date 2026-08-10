from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalAugmentedGeneration:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.store = []
        self.build_store(filepath)

    
    def load_documents(self, filepath):
        documents = []
        path = Path(filepath)



        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "source": str(path),
                "text": text
            })

        elif path.is_dir():
            for file_path in path.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                documents.append({
                    "source": str(file_path),
                    "text": text
                })

        else:
            print("path does not exist")


        return documents

    def chunk_text(self, text, chunk_size=500, overlap=20):
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        return chunks

            
    def build_store(self, filepath):
        documents = self.load_documents(filepath)
        store = []

        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for chunk_text in chunks:
                embedding = self.model.encode(chunk_text)
                store.append((embedding, chunk_text))

        self.store= store
        return store

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query)
        results = []
        for embedding, chunk_text in self.store:
            score = float(cosine_similarity(
                [query_embedding], # expects a list of vectors
                [embedding]
            )[0][0]) # cosine_similarity returns a 2d table, but you only want the number
            
            results.append((score, chunk_text))
            

        results.sort(key=lambda x:x[0], reverse=True)
        return results[:top_k]


rag = RetrievalAugmentedGeneration("news_knowledge")
rag.load_documents("news_knowledge")

