from collections import defaultdict


class LightweightKnowledgeGraph:
    def __init__(self) -> None:
        self.edges: dict[str, set[str]] = defaultdict(set)

    def add_relations(self, source: str, entities: list[str]) -> None:
        for entity in entities:
            self.edges[source].add(entity)

    def neighbors(self, source: str) -> list[str]:
        return sorted(self.edges.get(source, set()))

