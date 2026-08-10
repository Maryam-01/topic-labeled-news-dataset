# topic-labeled-news-dataset
100k+ topic labeled news articles published from thousands of news websites



### Context

We're [NewsCatcher](https://newscatcherapi.com/) team: we collect and index news articles. We provide News API to find relevant news data. 

We contribute a lot to the open-source community by sharing our work (find other links at the bottom of the description)



### Content

We collected over 100k articles for 8 different news topics
`BUSINESS`  |       15000
`ENTERTAINMENT`  |  15000
`HEALTH`      |     15000
`NATION`      |     15000
`SCIENCE`     |      3774
`SPORTS`       |    15000
`TECHNOLOGY`   |    15000
`WORLD`     |       15000

Those articles got published over the first half of August 2020. 

All `topics` have 15k articles except for `SCIENCE` which is 3774. Those articles are published by thousands of different news websites.


# News Topic Classifier with RAG Chatbot

This project is a Python-based news topic classification and response system. It classifies news headlines or short news text into topics such as `nation`, `world`, `business`, `technology`, `entertainment`, `sports`, `health`, and `science`, then uses a simple Retrieval-Augmented Generation system to provide a short explanation.

The project combines:

- a topic classifier
- a local text knowledge base
- sentence-transformer embeddings
- cosine similarity retrieval
- a Hugging Face text-generation model

---

## Project Structure

```text
project/
│
├── classifier.py
├── rag.py
├── chatbot.py
├── main.py
│
├── news_knowledge/
│   ├── example1.txt
│   ├── example2.txt
│   └── ...
│
├── test/
│   └── test_rag.py
│
├── requirements.txt
└── README.md
```

---

## Features

- Classifies news text into a predicted topic.
- Shows topic confidence scores.
- Loads local `.txt` files as a knowledge base.
- Splits documents into chunks.
- Creates embeddings using `sentence-transformers`.
- Retrieves the most relevant chunks using cosine similarity.
- Generates a short explanation using a Hugging Face language model.
- Includes assert-based tests for the RAG system.

---

## Topics

The classifier predicts one of the following topics:

```text
nation
world
business
technology
entertainment
sports
health
science
```

Example:

```text
You: CCTV pilot speeds up shoplifting investigations

Predicted topic: nation
nation 46%
world 12%
business 11%
technology 9%
entertainment 8%
sports 7%
health 4%
science 2%

Bot: This is classified as nation because it concerns public safety, policing, and crime investigation within the country.
```

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
transformers
sentence-transformers
scikit-learn
numpy
pytest
```

Depending on your classifier, you may also need:

```text
pandas
joblib
```

---

## Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

Or open the folder directly if you are working locally.

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add knowledge files

Create a folder called:

```text
news_knowledge
```

Add `.txt` files inside it.

Example:

```text
news_knowledge/
├── technology.txt
├── business.txt
├── sports.txt
├── health.txt
└── nation.txt
```

The RAG system currently loads only `.txt` files.

---

## RAG System

The RAG class is defined in `rag.py`.

It performs four main steps:

1. Loads documents from a file or folder.
2. Splits documents into word chunks.
3. Converts chunks into embeddings.
4. Retrieves the most similar chunks for a query.

Example:

```python
from rag import RetrievalAugmentedGeneration

rag = RetrievalAugmentedGeneration("news_knowledge")

results = rag.retrieve("CCTV pilot speeds up shoplifting investigations", top_k=3)

for score, chunk in results:
    print(score, chunk[:200])
```

---

## Important RAG Methods

### `load_documents(filepath)`

Loads a single `.txt` file or all `.txt` files in a folder.

Returns:

```python
[
    {
        "source": "news_knowledge/example.txt",
        "text": "document text..."
    }
]
```

---

### `chunk_text(text, chunk_size=500, overlap=20)`

Splits text into chunks by words.

Example:

```python
text = "one two three four five six seven"
chunks = rag.chunk_text(text, chunk_size=4, overlap=1)
```

Result:

```python
[
    "one two three four",
    "four five six seven",
    "seven"
]
```

---

### `build_store(filepath)`

Builds the vector store.

Each item in the store contains:

```python
(embedding, chunk_text)
```

---

### `retrieve(query, top_k=3)`

Finds the most relevant chunks for a query.

Returns:

```python
[
    (score, chunk_text),
    (score, chunk_text),
    (score, chunk_text)
]
```

The score is cosine similarity.

---

## Recommended `retrieve()` Implementation

To make testing easier, convert the cosine similarity score into a normal Python `float`:

```python
def retrieve(self, query, top_k=3):
    query_embedding = self.model.encode(query)
    results = []

    for embedding, chunk_text in self.store:
        score = float(cosine_similarity(
            [query_embedding],
            [embedding]
        )[0][0])

        results.append((score, chunk_text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
```

---

## Running the Chatbot

Run:

```bash
python main.py
```

Then enter a news headline:

```text
You: First OpenAI, now Meta - why does AI hacking keep happening?
```

Example output:

```text
Predicted topic: technology
technology 72%
business 10%
world 8%
science 5%
nation 3%
health 1%
entertainment 1%
sports 0%

Bot: This is classified as technology because it discusses AI companies, hacking, and digital security concerns.
```

---

## Hugging Face Generation Notes

This project may use a Hugging Face model with:

```python
model.generate(...)
```

You may see a warning like:

```text
Both `max_new_tokens` (=40) and `max_length` (=2048) seem to have been set.
`max_new_tokens` will take precedence.
```

This is not usually an error.

It means:

- `max_new_tokens` controls how many new tokens the model generates.
- `max_length` may exist in the model’s default generation configuration.
- `max_new_tokens` takes priority.

A typical generation call:

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
```

This uses greedy decoding because:

```python
do_sample=False
num_beams=1
```

---

## Testing

This project uses `pytest`.

Install pytest:

```bash
pip install pytest
```

Run all tests:

```bash
pytest
```

Run only the RAG tests:

```bash
pytest test/test_rag.py
```

Run with detailed output:

```bash
pytest -vv test/test_rag.py
```

---

### RAG returns no results

Check that:

- `news_knowledge` exists
- it contains `.txt` files
- the `.txt` files are not empty
- `rag.store` has chunks

---


### Bot talks about “classification” instead of the headline

Use a simpler prompt and avoid repeating the word `classified`.

Better:

```python
prompt = f"""
Headline:
{headline}

Topic:
{topic}

Write one short sentence explaining why the headline belongs to this topic.
Do not mention machine learning.
Do not mention algorithms.
"""
```

---

## Limitations

- The RAG system only reads `.txt` files.
- Retrieval quality depends on the content in `news_knowledge`.
- Small language models may not always follow instructions perfectly.
- If retrieved context is unrelated, the generated explanation may be poor.
- The classifier may produce uncertain predictions when topic scores are close.


---

## Author

Created as a news classification and RAG chatbot project.
