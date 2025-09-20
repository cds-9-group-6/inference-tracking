import mlflow
import pandas as pd
import os

# Example finance QA dataset
questions = ["What was the company’s revenue in Q2?"]
ground_truth = ["$25.3B"]
predicted_answers = ["$25B"]

mlflow.set_tracking_uri("http://0.0.0.0:5001/")
mlflow.set_experiment("QA_Evaluation")

with mlflow.start_run():
    data = pd.DataFrame(
        {
            "question": questions,
            "ground_truth": ground_truth,
            "predicted": predicted_answers,
        }
    )

    # Evaluate QA model (removed default evaluators to fix hang)
    results = mlflow.evaluate(
        data=data,
        targets="ground_truth",
        predictions="predicted",
        model_type="question-answering",
        evaluators=["default"],  # This was causing the hang - commented out
    )

print(results.metrics)
