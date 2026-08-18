import os
import pickle
import faiss
import numpy as np
import requests
import streamlit as st

from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_FOLDER = os.path.join(
    BASE_DIR,
    "vectorstore"
)

TOP_K = 4

EMBEDDING_MODEL = (
    "gemini-embedding-001"
)

EMBEDDING_DIMENSION = 768

LLM_MODEL = (
    "gemini-3.5-flash"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
    "Vehicle Insurance Claim Assistant",

    page_icon="🚗",

    layout="wide"

)


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:

    st.error(
        "GEMINI_API_KEY not found "
        "in .env"
    )

    st.stop()


# ============================================================
# LOAD FAISS
# ============================================================

try:

    index = faiss.read_index(

        os.path.join(

            VECTOR_FOLDER,

            "insurance.index"

        )

    )

    with open(

        os.path.join(

            VECTOR_FOLDER,

            "chunks.pkl"

        ),

        "rb"

    ) as file:

        chunks = pickle.load(
            file
        )


except Exception as e:

    st.error(
        f"Vector database error: {e}"
    )

    st.stop()


# ============================================================
# VERIFY DIMENSION
# ============================================================

if index.d != EMBEDDING_DIMENSION:

    st.error(

        f"FAISS dimension mismatch. "

        f"Expected {EMBEDDING_DIMENSION}, "

        f"but found {index.d}."

    )

    st.stop()


# ============================================================
# GEMINI QUERY EMBEDDING
# ============================================================

def get_query_embedding(
    text
):

    url = (

        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        "gemini-embedding-001:embedContent"

    )


    headers = {

        "x-goog-api-key":
        API_KEY,

        "Content-Type":
        "application/json"

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
        "RETRIEVAL_QUERY",

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

            f"Gemini Embedding Error "
            f"{response.status_code}: "
            f"{response.text}"

        )


    result = response.json()


    vector = result[
        "embedding"
    ][
        "values"
    ]


    return np.array(

        vector,

        dtype="float32"

    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question
):

    query_vector = (
        get_query_embedding(
            question
        )
    )


    query_vector = (
        query_vector.reshape(
            1,
            -1
        )
    )


    distances, indices = (
        index.search(
            query_vector,
            TOP_K
        )
    )


    results = []


    for rank, (
        distance,
        idx
    ) in enumerate(

        zip(

            distances[0],

            indices[0]

        ),

        start=1

    ):


        if idx < 0:

            continue


        document = (
            chunks[idx].copy()
        )


        document[
            "rank"
        ] = rank


        document[
            "distance"
        ] = float(
            distance
        )


        results.append(
            document
        )


    return results


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(

    question,

    retrieved_documents

):


    context = ""


    for document in (
        retrieved_documents
    ):


        context += f"""

SOURCE {document['rank']}

Document:
{document['source']}

Page:
{document['page']}

Retrieved Content:
{document['text']}

------------------------------------
"""


    prompt = f"""

You are a Vehicle Insurance
Document Question Answering Assistant.

Your job is to answer questions ONLY
using the retrieved insurance documents.

DO NOT use outside knowledge.

DO NOT invent information.

DO NOT guess.

QUESTION:

{question}


RETRIEVED DOCUMENT CONTEXT:

{context}


RULES:

1. Answer only from the retrieved context.

2. If the information is not present,
say:

"I could not find this information in
the provided insurance documents."


3. For claim-process questions,
provide numbered steps.

4. For questions requiring information
from multiple chunks, combine the
relevant retrieved information.

5. For policy exclusion questions,
clearly identify the exclusion.

6. For exclusion questions, include
an exact short quotation from the
retrieved document.

7. NEVER create or invent a quotation.

8. After the answer, provide:

Source:
Document name
Page number

9. If the documents do not establish
whether a particular claim will be
approved, do not make a final claim
decision.

10. For unrelated questions such as
vehicle prices, weather, sports,
general news, etc., state that the
information is not available in the
provided insurance documents.


Now answer the question.
"""


    url = (

        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/"
        f"{LLM_MODEL}:generateContent"

    )


    headers = {

        "x-goog-api-key":
        API_KEY,

        "Content-Type":
        "application/json"

    }


    data = {

        "contents": [

            {

                "parts": [

                    {
                        "text": prompt
                    }

                ]

            }

        ]

    }


    response = requests.post(

        url,

        headers=headers,

        json=data,

        timeout=60

    )


    if response.status_code != 200:

        raise Exception(

            f"Gemini Generation Error "
            f"{response.status_code}: "
            f"{response.text}"

        )


    result = response.json()


    return (

        result[
            "candidates"
        ][0][
            "content"
        ][
            "parts"
        ][0][
            "text"
        ]

    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚗 Vehicle Insurance Claim Assistant"
)

st.subheader(
    "RAG-Based Motor Insurance Document QA"
)


st.markdown(
    """
Ask questions about:

📄 Insurance policies  
🚨 Accident claims  
📋 Required documents  
🏢 Cashless claims  
💰 Reimbursement claims  
⚠️ Policy exclusions
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ RAG Configuration"
    )


    st.write(
        "**Embedding:** "
        "Gemini Embedding 001"
    )


    st.write(
        "**Embedding Dimension:** "
        f"{EMBEDDING_DIMENSION}"
    )


    st.write(
        "**Vector Store:** FAISS"
    )


    st.write(
        f"**Top-K:** {TOP_K}"
    )


    st.write(
        f"**LLM:** {LLM_MODEL}"
    )


    st.divider()


    st.success(

        f"{len(chunks)} "
        "chunks loaded"

    )


    st.info(

        "Answers are generated "
        "from retrieved insurance "
        "documents."

    )


# ============================================================
# DEMO QUESTIONS
# ============================================================

st.subheader(
    "💡 Demo Questions"
)


sample_questions = [

    "What is the first step after a car accident?",

    "What documents are required to file a motor insurance claim?",

    "Explain the complete claim process from the accident until settlement.",

    "Will my motor insurance claim be covered if I was driving under the influence of alcohol?",

    "What is the premium for a Mercedes-Benz C-Class in 2027?"

]


selected_question = st.selectbox(

    "Select a question:",

    [

        "-- Select a question --"

    ]
    +
    sample_questions

)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(

    "Enter your question:",

    value=(

        ""

        if selected_question ==
        "-- Select a question --"

        else selected_question

    ),

    height=100

)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(

    "🔎 Ask Question",

    type="primary"

):


    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(

        "Searching insurance documents..."

    ):


        try:

            retrieved_documents = (
                retrieve_documents(
                    question
                )
            )


        except Exception as e:

            st.error(
                f"Retrieval error: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # RETRIEVAL METRICS
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(

            "Retrieved Chunks",

            len(
                retrieved_documents
            )

        )


    with col2:

        st.metric(

            "Top-K",

            TOP_K

        )


    with col3:

        st.metric(

            "FAISS Vectors",

            index.ntotal

        )


    # --------------------------------------------------------
    # EXCLUSION DETECTION
    # --------------------------------------------------------

    exclusion_words = [

        "drunk",

        "alcohol",

        "intoxicated",

        "drugs",

        "influence",

        "excluded",

        "exclusion",

        "not covered"

    ]


    is_exclusion = any(

        word in question.lower()

        for word in exclusion_words

    )


    if is_exclusion:

        st.warning(

            "⚠️ POLICY EXCLUSION QUERY"

        )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner(

        "Generating grounded answer..."

    ):


        try:

            answer = generate_answer(

                question,

                retrieved_documents

            )


        except Exception as e:

            st.error(

                f"Gemini error: {e}"

            )

            st.stop()


    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "🤖 Answer"
    )


    st.markdown(
        answer
    )


    # --------------------------------------------------------
    # RETRIEVED EVIDENCE
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "📚 Retrieved Evidence"
    )


    st.caption(

        "The following chunks were "
        "retrieved from the FAISS "
        "vector database and supplied "
        "to the language model."

    )


    for document in (
        retrieved_documents
    ):


        title = (

            f"Source "
            f"{document['rank']} — "
            f"{document['source']} — "
            f"Page "
            f"{document['page']}"

        )


        with st.expander(
            title
        ):


            st.write(
                document["text"]
            )


            st.caption(

                f"Retrieval distance: "
                f"{document['distance']:.4f}"

            )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()


    st.caption(

        "⚠️ Academic prototype. "
        "This system provides "
        "document-grounded answers "
        "and does not make official "
        "insurance claim decisions."

    )