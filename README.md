# Tennis RAG Coach

A beginner-friendly tennis learning assistant built with a local RAG pipeline and a Streamlit UI. It answers questions from uploaded tennis documents only, cites retrieved sources, and clearly labels reasonable inferences when the exact answer is not explicitly stated in the knowledge base.

Tennis RAG Coach is designed for new tennis players who want to learn from curated tennis material instead of general web results. Upload PDFs or text files, ask beginner-friendly questions about rules, scoring, technique, or strategy, and get grounded answers based on the uploaded knowledge base.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Project Structure

- `rag_agent.py`: document parsing, chunking, TF-IDF retrieval, chat history, and Gemini API calls
- `rag_app.py`: UI for tennis uploads, beginner Q&A, and source display
- `test_rag_agent.py`: lightweight unit tests for chunking and retrieval behavior
- `Makefile`: convenience commands for setup, running, and checks

## What It Does

- Accepts tennis PDFs and text-based files as the knowledge base
- Parses and chunks uploaded content locally
- Uses TF-IDF retrieval to find the most relevant context
- Answers beginner tennis questions using only the uploaded materials
- Labels inference clearly when an answer is derived rather than directly stated
- Displays retrieved source chunks for inspection

## Scope

- The app is intentionally scoped to tennis learning content.
- It is optimized for beginner education and grounded Q&A, not live sports data or web search.
- Retrieval is local and lightweight by design. There is no external vector database in this version.

## Demo

Example workflow:

1. Upload tennis learning material such as beginner guides, rule books, drills, or coaching notes.
2. Ask grounded questions like:
   - `How does tennis scoring work?`
   - `What terms in these documents would confuse a first-time player? Define them simply.`
   - `What are the main beginner mistakes described in these files?`
   - `Based only on these notes, what is the likely best strategy for a beginner on clay?`
3. Review the answer and open the retrieved source chunks to inspect what the assistant used.

Example prompt:

```text
List all tennis terms in these documents that could confuse a beginner, define each one simply, group them by topic, and include every supported term you can find from the uploaded files.
```

Expected behavior:

- Answers should stay grounded in the uploaded files
- Sources should be visible in the retrieval panel
- If the exact answer is not directly stated, the app should label the derived part as inference
- Broader questions may require a follow-up prompt when using the Gemini free tier

## Run

Recommended environment:

- Python `3.10+`
- A Gemini API key
- If you use the free Gemini tier, expect occasional rate limits or temporary availability issues

Environment variables:

```bash
cp .env.example .env
export GEMINI_API_KEY="your-key"
```

Quick start:

```bash
make install
source .venv/bin/activate
export GEMINI_API_KEY="your-key"
make run
```

Manual setup:

1. Create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key or enter it in the sidebar:

```bash
export GEMINI_API_KEY="your-key"
```

4. Start the app:

```bash
streamlit run rag_app.py
```

## Notes

- Text and PDF uploads are supported.
- Retrieval is local TF-IDF over chunked documents.
- Answers are generated with `gemini-2.5-flash`.
- The assistant is tuned for beginner-friendly tennis explanations and should be fed tennis-only content.
- The app retries temporary Gemini failures, but free-tier usage can still produce intermittent `429` or `503` responses.
- Broad questions may still need follow-up prompts, especially when the free-tier model is under load.

## Tests

Run the lightweight standard-library test suite with:

```bash
make test
```

Additional checks:

```bash
make check
```
