# RAG-Based AI Resume Analyzer

A beginner-friendly portfolio project that compares a resume with a job description using Retrieval-Augmented Generation (RAG).

## What it does

1. Accepts a text-based resume PDF and a job description.
2. Extracts text from the PDF using PyPDF.
3. Splits the resume into overlapping chunks.
4. Uses ChromaDB to create embeddings and retrieve relevant chunks.
5. Sends the retrieved excerpts and job description to Gemini.
6. Displays an analysis and the evidence excerpts used.

## Requirements

- Python 3.10 or newer recommended
- Internet connection
- Your own Gemini API key

## Setup (Windows / VS Code)

1. Extract this ZIP and open the `RAG_AI_Resume_Analyzer` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install the packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Copy `.env.example` and rename the copy to `.env`.
6. Open `.env` and replace the placeholder with your Gemini API key:

   ```text
   GEMINI_API_KEY=your_actual_api_key_here
   ```

   Do not add quotes unless needed, and do not share or upload this file.

7. Start the app:

   ```bash
   streamlit run app.py
   ```

8. Open the local URL shown in the terminal (usually `http://localhost:8501`).

The key is loaded automatically from `.env`; you do not need to enter it in the app each time. You do not need to generate a new key for every run. The key can stop working if revoked, disabled, or affected by account quotas or billing limits.

## RAG in this project

- **Load:** extract resume text from PDF.
- **Chunk:** split it into smaller overlapping text pieces.
- **Index:** add pieces to ChromaDB; its default embedding function converts text into vectors.
- **Retrieve:** search for chunks semantically relevant to the job description.
- **Generate:** Gemini writes a report using the retrieved excerpts as context.

## Security and privacy

- The ZIP includes `.env.example`, not a real API key.
- Copy `.env.example` to `.env`; keep `.env` private.
- `.gitignore` excludes `.env` to help prevent accidental Git uploads.
- The resume text and job description are sent to Gemini for generation. Use sample or consented data and review Google's applicable privacy terms.
- Never publish API keys or real resumes to a public repository.

## Limitations

- Scanned/image-only PDFs may need OCR before this app can read them.
- Retrieval can miss details elsewhere in a resume.
- The output is approximate and should not be treated as a hiring decision.
- The first ChromaDB run may take longer while its default embedding model is downloaded.
- Model names and package APIs can change; consult the current Google Gen AI SDK documentation if an API error appears.

## Interview explanation

"This is a RAG-based resume analyzer built with Python and Streamlit. It extracts text from a PDF, divides it into chunks, uses ChromaDB to retrieve relevant excerpts based on a job description, and passes those excerpts to Gemini to generate a grounded analysis. The retrieved excerpts can be inspected to see what context was used."
