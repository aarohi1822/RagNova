from collections import defaultdict


class SessionMemory:
    def __init__(self) -> None:
        self.history: dict[str, list[str]] = defaultdict(list)

    def append(self, session_id: str, message: str) -> None:
        self.history[session_id].append(message)

    def get(self, session_id: str) -> list[str]:
        return self.history[session_id]

