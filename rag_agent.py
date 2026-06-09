from __future__ import annotations

import json
import math
import mimetypes
import re
import time
from io import BytesIO
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


STOP_WORDS = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "but",
    "can",
    "will",
    "with",
    "that",
    "this",
    "from",
    "they",
    "their",
    "its",
    "also",
    "into",
    "than",
    "then",
    "when",
    "what",
    "which",
    "would",
    "could",
    "should",
    "been",
    "being",
    "more",
    "each",
    "most",
    "such",
    "like",
    "very",
    "just",
    "about",
    "some",
    "over",
    "after",
    "before",
    "there",
    "other",
    "many",
    "only",
    "even",
    "well",
    "may",
    "does",
    "did",
    "use",
    "used",
    "get",
    "got",
    "new",
    "first",
    "last",
    "long",
    "great",
    "little",
    "own",
    "all",
    "one",
    "two",
    "any",
    "few",
    "now",
    "him",
    "his",
    "her",
    "she",
    "our",
    "you",
    "your",
    "them",
    "these",
    "those",
    "who",
    "how",
    "why",
    "said",
    "up",
    "so",
    "do",
    "an",
    "at",
    "by",
    "be",
    "or",
    "as",
    "it",
    "in",
    "on",
    "to",
    "of",
    "is",
    "a",
}


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    page: int | None = None
    vector: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    id: str
    name: str
    content: str
    chunk_count: int
    mime_type: str
    is_pdf: bool = False
    page_count: int | None = None


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    score: float
    page: int | None = None


class GeminiAPIError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    return chunks


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [
        token
        for token in cleaned.split()
        if len(token) > 2 and token not in STOP_WORDS
    ]


def build_tfidf_vectors(texts: Iterable[str]) -> list[dict[str, float]]:
    tokenized = [tokenize(text) for text in texts]
    if not tokenized:
        return []

    document_count = len(tokenized)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue

        term_frequency = Counter(tokens)
        vector: dict[str, float] = {}
        token_total = len(tokens)
        for token, count in term_frequency.items():
            idf = math.log((document_count + 1) / (document_frequency[token] + 1)) + 1
            vector[token] = (count / token_total) * idf
        vectors.append(vector)

    return vectors


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    keys = set(left) | set(right)
    dot_product = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class RagAgent:
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.chat_history: list[dict[str, str]] = []
        self.assistant_name = "Tennis RAG Coach"

    def add_text_document(
        self,
        name: str,
        content: str,
        *,
        mime_type: str | None = None,
    ) -> Document:
        document_id = f"doc-{len(self.documents) + 1}"
        document_chunks = [
            Chunk(
                id=f"{document_id}-chunk-{index}",
                document_id=document_id,
                document_name=name,
                text=chunk,
            )
            for index, chunk in enumerate(chunk_text(content, size=400, overlap=80), start=1)
        ]

        document = Document(
            id=document_id,
            name=name,
            content=content,
            chunk_count=len(document_chunks),
            mime_type=mime_type or mimetypes.guess_type(name)[0] or "text/plain",
            is_pdf=False,
        )
        self.documents.append(document)
        self.chunks.extend(document_chunks)
        self._rebuild_index()
        return document

    def add_pdf_document(self, name: str, pdf_bytes: bytes) -> Document:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        document_id = f"doc-{len(self.documents) + 1}"
        page_texts: list[tuple[int, str]] = []
        document_chunks: list[Chunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_texts.append((page_number, text))
            for chunk_index, chunk in enumerate(
                chunk_text(text, size=350, overlap=60),
                start=1,
            ):
                document_chunks.append(
                    Chunk(
                        id=f"{document_id}-page-{page_number}-chunk-{chunk_index}",
                        document_id=document_id,
                        document_name=name,
                        text=chunk,
                        page=page_number,
                    )
                )

        if not page_texts:
            raise ValueError("The PDF did not contain extractable text.")

        document = Document(
            id=document_id,
            name=name,
            content="\n\n".join(text for _, text in page_texts),
            chunk_count=len(document_chunks),
            mime_type="application/pdf",
            is_pdf=True,
            page_count=len(reader.pages),
        )
        self.documents.append(document)
        self.chunks.extend(document_chunks)
        self._rebuild_index()
        return document

    def retrieve(self, question: str, limit: int = 10) -> list[RetrievalHit]:
        query_vector = build_tfidf_vectors([question])[0]
        hits = [
            RetrievalHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                text=chunk.text,
                score=cosine_similarity(query_vector, chunk.vector),
                page=chunk.page,
            )
            for chunk in self.chunks
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits[:limit] if hit.score > 0.01]

    def answer_question(self, question: str, api_key: str, limit: int = 10) -> tuple[str, list[RetrievalHit]]:
        hits = self.retrieve(question, limit=limit)
        context = self._build_context(hits)
        answer = self._generate_answer(api_key=api_key, question=question, context=context)
        self.chat_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        self.chat_history = self.chat_history[-20:]
        return answer, hits

    def clear_documents(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.chat_history.clear()

    def clear_chat(self) -> None:
        self.chat_history.clear()

    def stats(self) -> dict[str, int | str]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "questions": len(self.chat_history) // 2,
            "model": self.model,
        }

    def _rebuild_index(self) -> None:
        vectors = build_tfidf_vectors(chunk.text for chunk in self.chunks)
        for chunk, vector in zip(self.chunks, vectors):
            chunk.vector = vector

    def _build_context(self, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "No relevant documents found."
        return "\n\n---\n\n".join(
            (
                f"[Source {index}: {hit.document_name}"
                f"{f' (p.{hit.page})' if hit.page else ''}]\n{hit.text}"
            )
            for index, hit in enumerate(hits, start=1)
        )

    def _generate_answer(self, api_key: str, question: str, context: str) -> str:
        system_prompt = (
            "You are a tennis learning assistant for beginners. "
            "The user will provide tennis-only documents, and your job is to help new tennis players "
            "understand the game clearly. "
            "Use only the uploaded files and pasted text as the knowledge base. "
            "Do not rely on outside facts unless you are making a clearly labeled inference from the provided context. "
            "If the exact answer is explicitly supported by the context, answer directly and cite the supporting sources. "
            "If the exact answer is not documented but a tennis-related conclusion can be reasonably derived from the context, "
            "answer it and explicitly label that portion as 'Inference:' or 'Reasonable inference:'. "
            "Do not over-refuse just because the wording of the question is broader than the text, but do stay grounded in the provided material. "
            "If the context is too weak even for a reasonable inference, say that clearly. "
            "Use simple language for beginners, explain tennis terms when needed, and prefer detailed, structured explanations over short summaries. "
            "When the user asks for a list, glossary, confusing terms, comparisons, steps, drills, or examples, be exhaustive within the provided context rather than overly brief. "
            "If the context supports many items, include many items. Do not stop after one or two examples unless the user asked for a short answer. "
            "By default, go one level deeper than a typical answer: explain the concept, why it matters, and give a simple example when the context supports it. "
            "For teaching questions, optimize for real understanding rather than brevity. "
            "When possible, explain technical decisions, tradeoffs, metrics, tools, and deployment details if the uploaded tennis material discusses them. "
            "When useful, organize answers with headings such as Rules, Scoring, Technique, Strategy, Equipment, Coaching Notes, Examples, Why It Matters, and Common Beginner Mistakes. "
            "Always distinguish documented facts from inference. Cite source names and page numbers when available.\n\n"
            f"Context:\n{context}"
        )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        contents = self._build_contents(system_prompt, question)
        answer_parts: list[str] = []
        finish_reason = ""

        for continuation_index in range(3):
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": 3072,
                    "temperature": 0.25,
                },
            }
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            data = self._request_with_retry(request)

            candidate = data.get("candidates", [{}])[0]
            finish_reason = candidate.get("finishReason", "")
            text = self._extract_candidate_text(candidate)
            if text:
                answer_parts.append(text)

            if finish_reason != "MAX_TOKENS":
                break

            contents.extend(
                [
                    {"role": "model", "parts": [{"text": text or ""}]},
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Continue exactly where you left off. "
                                    "Do not restart or repeat earlier content."
                                )
                            }
                        ],
                    },
                ]
            )

            if continuation_index == 2:
                break

        answer = "\n".join(part for part in answer_parts if part).strip()
        if not answer:
            answer = "No response returned by the model."

        if finish_reason == "MAX_TOKENS":
            answer += (
                "\n\nNote: the model still hit the output limit after continuation. "
                "Ask a narrower follow-up if you want the remaining detail."
            )
        elif finish_reason and finish_reason != "STOP":
            answer += f"\n\nNote: the model stopped with finish reason `{finish_reason}`."

        return answer

    def _build_contents(self, system_prompt: str, question: str) -> list[dict]:
        return [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {
                "role": "model",
                "parts": [
                    {
                        "text": (
                            "Understood. I will act as a beginner-friendly tennis coach "
                            "using only the uploaded knowledge base, and I will clearly label "
                            "any reasonable inference that is not explicitly documented. "
                            "For list-style questions, I will be thorough and include as many supported items as the context allows. "
                            "I will favor deeper teaching answers over short summaries."
                        )
                    }
                ],
            },
            *[
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}],
                }
                for message in self.chat_history
            ],
            {"role": "user", "parts": [{"text": question}]},
        ]

    def _extract_candidate_text(self, candidate: dict) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()

    def _request_with_retry(self, request: urllib.request.Request) -> dict:
        delays = [1.0, 2.5, 5.0]
        last_error: Exception | None = None

        for attempt, delay in enumerate(delays, start=1):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = exc
                if exc.code in {429, 500, 503} and attempt < len(delays):
                    time.sleep(delay)
                    continue
                if exc.code == 503:
                    raise GeminiAPIError(
                        "Gemini is temporarily overloaded after multiple retries. "
                        "Wait a moment and try again.",
                        kind="temporary_overload",
                    ) from exc
                raise self._build_api_error(exc.code, error_body) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < len(delays):
                    time.sleep(delay)
                    continue
                raise GeminiAPIError(
                    f"Unable to reach Gemini API: {exc.reason}",
                    kind="network",
                ) from exc

        raise GeminiAPIError(
            "Gemini request failed after multiple retries.",
            kind="generic",
        ) from last_error

    def _build_api_error(self, status_code: int, error_body: str) -> GeminiAPIError:
        normalized = error_body.lower()
        if status_code == 429 or "quota" in normalized or "rate limit" in normalized:
            return GeminiAPIError(
                "Your Gemini API key has reached its current usage limit. "
                "Wait for the quota to reset, switch to a different key, or try again later.",
                kind="quota_limit",
            )
        if "api key not valid" in normalized or "invalid api key" in normalized:
            return GeminiAPIError(
                "The Gemini API key appears to be invalid. Update it in the sidebar and try again.",
                kind="invalid_key",
            )
        if status_code == 403:
            return GeminiAPIError(
                "Gemini rejected this request. Check that your API key has access to the selected model and try again.",
                kind="permission",
            )
        return GeminiAPIError(f"Gemini request failed: {error_body}", kind="generic")
