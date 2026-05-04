from app.services.evaluation import Evaluator


if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.evaluate(relevant_hit_ranks=[1, 2, 0, 3, 1], total_queries=5)
    print(metrics)

