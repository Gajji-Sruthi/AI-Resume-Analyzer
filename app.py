
import os
import re
import uuid
import time

import chromadb
import streamlit as st

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG-Based AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG-Based AI Resume Analyzer")

st.write(
    "Upload your resume and paste a job description. "
    "The application retrieves relevant resume sections "
    "and uses Gemini AI to generate an analysis."
)

st.caption(
    "Learning demo only. Results may be incomplete and "
    "must not be used as an automated hiring decision."
)


# --------------------------------------------------
# 3. EXTRACT TEXT FROM PDF
# --------------------------------------------------

def extract_resume_text(pdf_file):
    """Extract selectable text from each page of a PDF."""

    reader = PdfReader(pdf_file)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    return text.strip()


# --------------------------------------------------
# 4. CLEAN RESUME TEXT
# --------------------------------------------------

def clean_text(text):

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 5. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def split_into_chunks(text, chunk_size=180, overlap=35):
    """Split resume text into overlapping word chunks."""

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# 6. STORE AND RETRIEVE USING CHROMADB
# --------------------------------------------------

def retrieve_relevant_chunks(
    chunks,
    job_description,
    top_k=4
):
    """Retrieve resume chunks relevant to the job description."""

    if not chunks:
        return []

    chroma_client = chromadb.Client()

    collection_name = "resume_" + uuid.uuid4().hex[:12]

    collection = chroma_client.create_collection(
        name=collection_name
    )

    try:

        collection.add(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
        )

        result = collection.query(
            query_texts=[job_description],
            n_results=min(top_k, len(chunks)),
        )

        retrieved_documents = result.get(
            "documents", [[]]
        )[0]

        return retrieved_documents

    finally:

        chroma_client.delete_collection(
            name=collection_name
        )


# --------------------------------------------------
# 7. GENERATE ANALYSIS USING GEMINI
# --------------------------------------------------

def generate_analysis(
    api_key,
    retrieved_chunks,
    job_description
):
    """Generate an evidence-based resume analysis using Gemini."""

    client = genai.Client(api_key=api_key)

    evidence = "\n\n".join(
        f"[Retrieved resume excerpt {i}]\n{chunk}"
        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        )
    )

    prompt = f"""
You are an AI resume analysis assistant.

Analyze the candidate's resume against the given job description.

JOB DESCRIPTION:

{job_description}


RETRIEVED RESUME EVIDENCE:

{evidence}


Generate a detailed report using these headings:

1. Overall Match Summary

2. Relevant Skills Supported by Resume

3. Potential Skill Gaps

4. Relevant Projects and Experience

5. Five Suggestions for Resume Improvement

6. Five Likely Interview Questions


IMPORTANT RULES:

- Base candidate-related claims only on the supplied resume evidence.
- Never invent skills, qualifications, experience or achievements.
- If a skill is not found in the retrieved excerpts,
  mention that it was not found in the retrieved excerpts.
- Do not conclude that the candidate lacks a skill
  just because it was not retrieved.
- Explain that other resume information may not have been retrieved.
- Do not make an automated hiring decision.
- Use simple, clear language.
"""

    # Try available Gemini models in order.
    models = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash"
    ]

    last_error = None

    for model in models:

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            if response.text:
                return response.text

            raise ValueError(
                "Gemini returned an empty response."
            )

        except Exception as error:

            last_error = error

            error_message = str(error).upper()

            # Retry another model only for temporary
            # service overload or availability errors.
            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                time.sleep(2)

                continue

            raise

    raise last_error


# --------------------------------------------------
# 8. SIDEBAR - API CONFIGURATION
# --------------------------------------------------

with st.sidebar:

    st.header("API Configuration")

    if GEMINI_API_KEY:

        st.success(
            "Gemini API key loaded successfully!"
        )

    else:

        st.error(
            "Gemini API key not found."
        )

        st.write(
            "Create a .env file in the project folder "
            "and add GEMINI_API_KEY=your_actual_api_key"
        )


# --------------------------------------------------
# 9. USER INPUT
# --------------------------------------------------

resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the Job Description",
    height=220,
    placeholder="Paste the job responsibilities and required skills here..."
)


# --------------------------------------------------
# 10. ANALYZE RESUME BUTTON
# --------------------------------------------------

if st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
):

    if not GEMINI_API_KEY:

        st.error(
            "Please add your Gemini API key to the .env file."
        )

    elif resume_file is None:

        st.error(
            "Please upload your resume PDF."
        )

    elif not job_description.strip():

        st.error(
            "Please paste a job description."
        )

    else:

        try:

            # STEP 1: EXTRACT TEXT

            with st.spinner(
                "Extracting resume text..."
            ):

                resume_text = clean_text(
                    extract_resume_text(resume_file)
                )

            if len(resume_text) < 40:

                st.error(
                    "Very little text was extracted. "
                    "Your PDF may be scanned. "
                    "Try a PDF containing selectable text."
                )

            else:

                # STEP 2: CHUNKING

                chunks = split_into_chunks(
                    resume_text
                )

                st.success(
                    f"Resume text extracted! Created {len(chunks)} chunks."
                )

                # STEP 3: RETRIEVAL

                with st.spinner(
                    "Retrieving relevant resume sections..."
                ):

                    retrieved_chunks = retrieve_relevant_chunks(
                        chunks,
                        job_description.strip()
                    )

                if not retrieved_chunks:

                    st.error(
                        "No relevant resume sections could be retrieved."
                    )

                else:

                    st.success(
                        f"Retrieved {len(retrieved_chunks)} relevant sections."
                    )

                    # STEP 4: GEMINI GENERATION

                    with st.spinner(
                        "Generating AI report..."
                    ):

                        report = generate_analysis(
                            GEMINI_API_KEY,
                            retrieved_chunks,
                            job_description.strip()
                        )

                    # STEP 5: DISPLAY REPORT

                    st.success(
                        "Resume analysis completed successfully!"
                    )

                    st.subheader(
                        "📊 Resume Analysis Report"
                    )

                    st.markdown(report)

                    # STEP 6: DISPLAY RETRIEVED CHUNKS

                    with st.expander(
                        "View Retrieved Resume Sections"
                    ):

                        st.write(
                            "These resume excerpts were retrieved "
                            "from ChromaDB and supplied to Gemini."
                        )

                        for index, chunk in enumerate(
                            retrieved_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Resume Excerpt {index}**"
                            )

                            st.write(chunk)

        except Exception as error:

            st.error(
                "The application could not complete the analysis."
            )

            st.code(
                f"{type(error).__name__}: {error}"
            )

            st.info(
                "Please check your internet connection, "
                "API configuration, installed packages "
                "and PDF format. Never share your API key."
            )


# --------------------------------------------------
# 11. PROJECT WORKFLOW
# --------------------------------------------------

st.divider()

st.subheader(
    "🔍 How This RAG Project Works"
)

st.write(
    """
    **Step 1:** Upload Resume PDF

    **Step 2:** Extract Text from PDF

    **Step 3:** Split Text into Chunks

    **Step 4:** Store Chunks in ChromaDB

    **Step 5:** Retrieve Relevant Resume Sections

    **Step 6:** Send Retrieved Context and Job Description to Gemini

    **Step 7:** Generate Resume Analysis Report
    """
)

st.info(
    "Resume PDF → Text Extraction → Chunking → "
    "ChromaDB Retrieval → Gemini AI → Analysis Report"
)
