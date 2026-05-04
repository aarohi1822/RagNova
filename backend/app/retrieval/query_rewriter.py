class QueryRewriter:
    def rewrite(self, question: str, history: list[str]) -> str:
        if not history:
            return question.strip()
        previous = " ".join(history[-2:])
        return f"{question.strip()} Context from prior conversation: {previous}"

