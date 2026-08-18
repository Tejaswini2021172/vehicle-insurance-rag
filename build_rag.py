import os
import pickle
import faiss
import numpy as np
import requests
import time

from pypdf import PdfReader
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "gemini-embedding-001"

# Use 768 dimensions for Gemini
EMBEDDING_DIMENSION = 768

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()


# ============================================================
# CREATE VECTORSTORE
# ============================================================

os.makedirs(
    VECTOR_FOLDER,
    exist_ok=True
)


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(filepath):

    filename = os.path.basename(filepath)

    print(f"\nReading: {filename}")

    pages = []

    try:

        reader = PdfReader(filepath)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text and text.strip():

                pages.append({

                    "page": page_number,

                    "text": text,

                    "method": "PyPDF"

                })

    except Exception as e:

        print(
            f"PDF extraction error: {e}"
        )

    print(
        f"  Pages extracted: {len(pages)}"
    )

    return pages


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []

print("\n==========================================")
print(" VEHICLE INSURANCE RAG BUILDER")
print(" GEMINI EMBEDDING VERSION")
print("==========================================\n")


for filename in sorted(
    os.listdir(DATA_FOLDER)
):

    if filename.lower().endswith(".pdf"):

        filepath = os.path.join(
            DATA_FOLDER,
            filename
        )

        pages = extract_pdf_text(
            filepath
        )

        documents.append({

            "filename": filename,

            "pages": pages

        })


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = []


for document in documents:

    filename = document["filename"]

    for page_data in document["pages"]:

        text = " ".join(
            page_data["text"].split()
        )

        page_number = page_data["page"]

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append({

                    "text": chunk_text,

                    "source": filename,

                    "page": page_number,

                    "method": page_data["method"]

                })

            start += (
                CHUNK_SIZE -
                CHUNK_OVERLAP
            )


print("\n==========================================")
print(" CHUNKING COMPLETE")
print("==========================================")

print(
    f"Documents: {len(documents)}"
)

print(
    f"Chunks: {len(chunks)}"
)


if not chunks:

    print(
        "ERROR: No text extracted."
    )

    exit()


# ============================================================
# GEMINI EMBEDDING FUNCTION
# ============================================================

def get_embedding(text):

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        "gemini-embedding-001:embedContent"
    )

    headers = {

        "x-goog-api-key": API_KEY,

        "Content-Type": "application/json"

    }

    data = {

        "model":
        "models/gemini-embedding-001",

        "content": {

            "parts": [

                {
                    "text": text
                }

            ]

        },

        "taskType":
        "RETRIEVAL_DOCUMENT",

        "outputDimensionality":
        EMBEDDING_DIMENSION

    }


    response = requests.post(

        url,

        headers=headers,

        json=data,

        timeout=60

    )


    if response.status_code != 200:

        raise Exception(

            f"Embedding API Error "
            f"{response.status_code}: "
            f"{response.text}"

        )


    result = response.json()

    vector = result[
        "embedding"
    ][
        "values"
    ]

    return vector


# ============================================================
# GENERATE GEMINI EMBEDDINGS
# ============================================================

print("\n==========================================")
print(" GENERATING GEMINI EMBEDDINGS")
print("==========================================\n")


embeddings = []


for i, chunk in enumerate(chunks):

    print(
        f"Embedding "
        f"{i + 1}/{len(chunks)}"
    )

    vector = get_embedding(
        chunk["text"]
    )

    embeddings.append(
        vector
    )

    # Small delay to avoid rate limits
    time.sleep(0.1)


embeddings = np.array(

    embeddings,

    dtype="float32"

)


print(
    "\nEmbedding shape:",
    embeddings.shape
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]


index = faiss.IndexFlatL2(
    dimension
)


index.add(
    embeddings
)


print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(

    index,

    os.path.join(

        VECTOR_FOLDER,

        "insurance.index"

    )

)


# ============================================================
# SAVE CHUNKS
# ============================================================

with open(

    os.path.join(

        VECTOR_FOLDER,

        "chunks.pkl"

    ),

    "wb"

) as file:

    pickle.dump(

        chunks,

        file

    )


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "embedding_model":
    EMBEDDING_MODEL,

    "embedding_dimension":
    dimension,

    "chunk_size":
    CHUNK_SIZE,

    "chunk_overlap":
    CHUNK_OVERLAP,

    "top_k":
    4

}


with open(

    os.path.join(

        VECTOR_FOLDER,

        "config.pkl"

    ),

    "wb"

) as file:

    pickle.dump(

        config,

        file

    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print(" RAG DATABASE CREATED")
print("==========================================")

print(
    f"Documents : {len(documents)}"
)

print(
    f"Chunks    : {len(chunks)}"
)

print(
    f"Vectors   : {index.ntotal}"
)

print(
    f"Dimension : {dimension}"
)

print(
    "\nGemini Embedding + FAISS ready!"
)