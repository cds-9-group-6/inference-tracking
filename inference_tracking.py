import os
import mlflow
from openai import OpenAI
import pandas as pd
import httpx
from mlflow.tracking import MlflowClient
import logging
from mlflow.metrics.genai import answer_similarity  # , EvaluationExample, faithfulness,
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler('inference_tracking.log'),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Class to handle LLM evaluation using MLflow.
    This class is used to evaluate the performance of the model by using the judge model.
    The judge model is used to generate the expected responses.
    The expected responses are then used to evaluate the performance of the model.
    The performance of the model is evaluated by using the answer similarity metric.
    The answer similarity metric is used to evaluate the performance of the model by using the expected responses.

    """

    def __init__(
        self,
        openai_api_key: str,
        openai_model: str = "gpt-4",
        teacher_model_temperature: float = 0.7,
        teacher_model_max_tokens: int = 1500,
    ):
        """
        Initialize the evaluator with OpenAI credentials

        Args:
            openai_api_key: OpenAI API key for judge model
            openai_model: OpenAI model to use as judge (default: gpt-4)
        """
        self.openai_model = openai_model
        self.teacher_model_client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAPI_KEY"),
            max_retries=0,
            # timeout=custom_timeout,
        )
        # self.teacher_model_client = OpenAI(
        #     api_key=openai_api_key,
        #     timeout=httpx.Timeout(30.0, read=180.0),
        # )
        self.teacher_model_temperature = teacher_model_temperature
        self.teacher_model_max_tokens = teacher_model_max_tokens
        logger.info(f"Initialized LLMEvaluator with model: {openai_model}")

    @mlflow.trace
    def setup_mlflow(
        self, tracking_uri: str, experiment_name: str, active_model_name: str = None
    ):
        """
        Setup MLflow tracking

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
            active_model_name: Name of the active model (optional)
        """
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI must be provided")

        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)

        if exp is None:
            mlflow.set_experiment(experiment_name)
        elif exp.lifecycle_stage == "deleted":
            logger.info("The experiment is deleted. Restoring it...")
            client.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(experiment_name)
        else:
            mlflow.set_experiment(experiment_name)

        if active_model_name:
            mlflow.set_active_model(name=active_model_name)

        logger.info(f"tracking uri: {mlflow.get_tracking_uri()}")
        logger.info(f"experiment: {mlflow.get_experiment_by_name(experiment_name)}")
        logger.info(f"active model: {mlflow.get_active_model_id()}")

        # Enable MLflow automatic tracing for OpenAI
        # <TODO>: This is not working as expected. Need to fix this.
        mlflow.openai.autolog()

    @staticmethod
    def to_single_line(s: str) -> str:
        """Convert multi-line string to single line"""
        return " ".join(s.split())

    @mlflow.trace
    def generate_expected_response_with_judge_model(self, question: str) -> str:
        """
        Generate expected/reference response using GPT-4 for evaluation purposes.
        In this call GPT-4 is used as the teacher model also called as reference model or Judge model.

        Args:
            question: The question to generate expected response for

        Returns:
            Generated expected response string
        """
        system_prompt = """You are an expert agricultural advisor with deep knowledge of farming practices in India. 
        You provide comprehensive, accurate, and practical advice about plant diseases, treatments, and agricultural practices.
        Your responses should be detailed, scientifically accurate, and include specific recommendations for conditions in India.
        Include relevant information about:
        - Specific chemicals/treatments available in India
        - Any additional recommendations from Agricultural University from India
        - Local farming practices and conditions in India
        - Safety considerations and application methods
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            response = self.teacher_model_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=self.teacher_model_temperature,
                max_tokens=self.teacher_model_max_tokens,
            )
            logger.info(f"Generated expected response for question: {question[:50]}...")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(
                f"Error generating expected response with {self.openai_model}: {e}"
            )
            return ""

    @mlflow.trace
    def generate_expected_responses(self, questions: List[str]) -> List[str]:
        """
        Generate expected responses for a list of questions

        Args:
            questions: List of questions to generate expected responses for

        Returns:
            List of expected responses (single line format)
        """
        logger.info("Generating expected responses using GPT-4...")
        expected_responses_judge = []

        for question in questions:
            logger.info(f"Generating expected response for: {question}")
            expected_resp = self.generate_expected_response_with_judge_model(question)
            expected_responses_judge.append(expected_resp)
            logger.info(
                f"Generated expected response length: {len(expected_resp)} characters"
            )

        # Convert to single line format
        expected_responses_judge = [
            self.to_single_line(resp) for resp in expected_responses_judge
        ]
        logger.info(f"Generated {len(expected_responses_judge)} expected responses")

        return expected_responses_judge

    @mlflow.trace
    def evaluate_responses(
        self,
        questions: List[str],
        generated_responses: List[str],
        expected_responses: List[str] = None,
        use_judge_model: bool = True,
        session_id: str = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated responses using MLflow

        Args:
            questions: List of input questions
            generated_responses: List of responses generated by the model being evaluated
            expected_responses: List of expected responses (optional, will generate if not provided)
            use_judge_model: Whether to use judge model to generate expected responses
            session_id: Session ID for the trace
            user_id: User ID for the trace
        Returns:
            Dictionary containing evaluation metrics
        """
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id if session_id else "default",
                "mlflow.trace.user": user_id if user_id else "default",
            }
        )
        

        logger.info("Evaluating responses...")
        # Generate expected responses if not provided
        if expected_responses is None and use_judge_model:
            expected_responses = self.generate_expected_responses(questions)
        elif expected_responses is None:
            raise ValueError(
                "Either provide expected_responses or set use_judge_model=True"
            )

        # Validate input lengths
        if not (len(questions) == len(generated_responses) == len(expected_responses)):
            raise ValueError(
                "Questions, generated_responses, and expected_responses must have the same length"
            )

        logger.info(f"Evaluating {len(questions)} question-response pairs")

        # Create evaluation DataFrame
        data = pd.DataFrame(
            {
                "questions": questions,
                "expected_response_from_judge": expected_responses,
                "generated_response": generated_responses,
            }
        )

        # Setup evaluation metric
        answer_similarity_metric = answer_similarity(
            model=f"openai:/{self.openai_model}"
        )
        logger.info(f"Using metric: {answer_similarity_metric.name}")

        # Run evaluation
        with mlflow.start_run() as run:
            results = mlflow.evaluate(
                data=data,
                targets="expected_response_from_judge",
                predictions="generated_response",
                model_type="text",
                extra_metrics=[answer_similarity_metric],
                evaluators="default",
                evaluator_config={
                    "col_mapping": {
                        "inputs": "questions",
                    }
                },
            )

            # Log additional information
            mlflow.log_param("num_questions", len(questions))
            mlflow.log_param("judge_model", self.openai_model)
            mlflow.log_param(
                "teacher_model_temperature", self.teacher_model_temperature
            )
            mlflow.log_param("teacher_model_max_tokens", self.teacher_model_max_tokens)
            mlflow.log_metrics(results.metrics)

            # Build metadata dictionary
            trace_metadata = {}            
            # Add metrics to metadata
            for metric_key, metric_value in results.metrics.items():
                trace_metadata[f"metric.{metric_key}"] = str(metric_value)
            trace_metadata["mlflow.trace.runid"] = run.info.run_id
            mlflow.update_current_trace(metadata=trace_metadata)
            
            run_id = run.info.run_id

        mlflow.end_run()
        logger.info("Evaluation complete.")
        logger.info(f"Metrics:\n{results.metrics}")

        return {"metrics": results.metrics, "run_id": run_id, "results": results}


# <TODO>: This is not working as expected. Need to fix this.
def get_trace_for_active_model(n: int = 1):
    """Get traces for the active model"""
    active_model_id = mlflow.get_active_model_id()
    return mlflow.search_traces(model_id=active_model_id)
