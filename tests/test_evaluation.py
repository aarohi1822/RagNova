from app.services.evaluation import Evaluator


def test_evaluation_metrics_are_bounded() -> None:
    evaluator = Evaluator()
    metrics = evaluator.evaluate([1, 2, 0], total_queries=3)
    assert 0 <= metrics.recall_at_k <= 1
    assert 0 <= metrics.mrr <= 1
    assert 0 <= metrics.answer_relevance <= 1
    assert 0 <= metrics.faithfulness <= 1

