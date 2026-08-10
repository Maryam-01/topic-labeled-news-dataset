from rag import RetrievalAugmentedGeneration


def test_load_documents():
    rag = RetrievalAugmentedGeneration("news_knowledge/business_articles.txt")
    docs = rag.load_documents("news_knowledge/business_articles.txt")

    assert len(docs) == 1
    assert docs[0]["source"] == "news_knowledge/business_articles.txt"
    assert isinstance(docs[0]["text"], str)
    assert len(docs[0]["text"]) > 0


def test_chunks_are_created():
    rag = RetrievalAugmentedGeneration("news_knowledge")

    text = "one two three four five six seven"
    chunks = rag.chunk_text(text, chunk_size=4, overlap=1)

    assert len(chunks) == 3
    assert chunks[0] == "one two three four"
    assert chunks[1] == "four five six seven"
    assert chunks[2] == "seven"

def test_build_store():
    rag = RetrievalAugmentedGeneration("news_knowledge")
    assert len(rag.store) > 0
    embedding, chunk_text = rag.store[0]
    assert embedding is not None
    assert len(embedding) == 384
    assert isinstance(chunk_text,str)
    assert len(chunk_text.strip()) > 0


def test_retrieve():
    rag = RetrievalAugmentedGeneration("news_knowledge")

    results = rag.retrieve("CCTV pilot speeds up shoplifting investigations", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3

    score, chunk_text = results[0]

    assert isinstance(score, float)
    assert -1 <= score <= 1
    assert isinstance(chunk_text, str)
    assert len(chunk_text.strip()) > 0


def test_retrieve_empty_query():
    rag = RetrievalAugmentedGeneration("news_knowledge")

    results = rag.retrieve("", top_k=3)

    assert isinstance(results, list)

        



