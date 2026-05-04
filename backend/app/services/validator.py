from app.models.domain import RetrievedChunk


class AnswerValidator:
    def validate(self, answer: str, chunks: list[RetrievedChunk]) -> list[str]:
        notes: list[str] = []
        if not chunks:
            notes.append("No evidence chunks retrieved; answer confidence should be treated as low.")
        if "[" not in answer:
            notes.append("Inline citation markers were not detected in the answer.")
        if len(answer.split()) < 20:
            notes.append("Answer is brief; consider prompting for a deeper evidence-grounded explanation.")
        return notes

