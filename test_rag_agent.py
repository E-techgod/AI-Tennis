import unittest

from rag_agent import RagAgent, chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_chunk_text_returns_multiple_chunks_for_long_input(self) -> None:
        text = " ".join(f"word{i}" for i in range(900))

        chunks = chunk_text(text, size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.strip()) > 20 for chunk in chunks))


class RagAgentTests(unittest.TestCase):
    def test_add_text_document_indexes_chunks(self) -> None:
        agent = RagAgent()

        document = agent.add_text_document(
            "tennis-basics.txt",
            " ".join(["serve", "rally", "forehand", "backhand"] * 150),
        )

        self.assertEqual(document.name, "tennis-basics.txt")
        self.assertGreater(document.chunk_count, 0)
        self.assertEqual(len(agent.documents), 1)
        self.assertEqual(len(agent.chunks), document.chunk_count)

    def test_retrieve_returns_relevant_chunk(self) -> None:
        agent = RagAgent()
        agent.add_text_document(
            "serving-guide.txt",
            (
                "A tennis serve starts the point. "
                "Players should learn toss timing, balance, and follow-through. "
            )
            * 50,
        )
        agent.add_text_document(
            "footwork-guide.txt",
            (
                "Good tennis footwork helps players recover to position, split step, "
                "and move efficiently during rallies. "
            )
            * 50,
        )

        hits = agent.retrieve("How do I improve my serve?", limit=5)

        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].document_name, "serving-guide.txt")
        self.assertGreater(hits[0].score, 0.01)


if __name__ == "__main__":
    unittest.main()
