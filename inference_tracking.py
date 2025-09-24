import os
import mlflow
from openai import OpenAI, AsyncOpenAI
import pandas as pd
import httpx
from mlflow.tracking import MlflowClient
import logging
from mlflow.metrics.genai import answer_similarity  # , EvaluationExample, faithfulness,
from typing import List, Dict, Any
import numpy as np
import asyncio
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
# from mlflow.pytorch import autolog
# from mlflow.tensorflow import autolog


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


class OTelMetricsExporter:
    """Class to handle sending metrics to OpenTelemetry Collector for Prometheus scraping."""
    
    def __init__(
        self, 
        otlp_endpoint: str = "http://localhost:4317",
        export_interval_millis: int = 10000,  # 10 seconds
        service_name: str = "llm-evaluation-service",
        service_version: str = "1.0.0"
    ):
        """
        Initialize the OTel metrics exporter.
        
        Args:
            otlp_endpoint: The OTLP gRPC endpoint for the OTel Collector
            export_interval_millis: How often to export metrics (in milliseconds)
            service_name: Name of the service for metric labeling
            service_version: Version of the service for metric labeling
        """
        self.otlp_endpoint = otlp_endpoint
        self.service_name = service_name
        self.service_version = service_version
        
        # Create OTLP exporter
        otlp_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use insecure connection for local development
        )
        
        # Create metric reader with periodic export
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=export_interval_millis
        )
        
        # Create meter provider
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        
        # Set the global meter provider
        metrics.set_meter_provider(meter_provider)
        
        # Create a meter
        self.meter = metrics.get_meter(
            name=service_name,
            version=service_version
        )
        
        # Create metric instruments
        self.toxicity_gauge = self.meter.create_gauge(
            name="llm_evaluation_toxicity",
            description="Toxicity metrics from LLM evaluation",
            unit="1"
        )
        
        self.readability_gauge = self.meter.create_gauge(
            name="llm_evaluation_readability",
            description="Readability metrics from LLM evaluation",
            unit="1"
        )
        
        self.similarity_gauge = self.meter.create_gauge(
            name="llm_evaluation_similarity",
            description="Answer similarity metrics from LLM evaluation",
            unit="1"
        )
        
        self.general_gauge = self.meter.create_gauge(
            name="llm_evaluation_metric",
            description="General metrics from LLM evaluation",
            unit="1"
        )
        
        logger.info(f"Initialized OTelMetricsExporter with endpoint: {otlp_endpoint}")
    
    def _convert_numpy_to_python(self, value):
        """Convert numpy types to Python native types."""
        if isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            return 0.0  # Replace NaN with 0 for metrics
        else:
            return value
    
    def _parse_metric_name(self, metric_name: str):
        """
        Parse metric name to extract components.
        
        Args:
            metric_name: Name like 'toxicity/v1/mean' or 'flesch_kincaid_grade_level/v1/variance'
            
        Returns:
            Tuple of (base_name, version, statistic)
        """
        parts = metric_name.split('/')
        if len(parts) >= 3:
            base_name = parts[0]
            version = parts[1]
            statistic = parts[2]
            return base_name, version, statistic
        else:
            return metric_name, "v1", "value"
    
    def send_metrics(
        self, 
        metrics_dict: Dict[str, Any], 
        run_id: str = None, 
        session_id: str = None,
        user_id: str = None,
        additional_labels: Dict[str, str] = None
    ):
        """
        Send metrics to OpenTelemetry Collector.
        
        Args:
            metrics_dict: Dictionary of metrics from MLflow evaluation
            run_id: MLflow run ID
            session_id: Session ID for tracing
            user_id: User ID for tracing
            additional_labels: Additional labels to attach to metrics
        """
        try:
            # Base labels for all metrics
            base_labels = {
                "service": self.service_name,
                "version": self.service_version,
            }
            
            # Add optional labels
            if run_id:
                base_labels["run_id"] = run_id
            if session_id:
                base_labels["session_id"] = session_id
            if user_id:
                base_labels["user_id"] = user_id
            if additional_labels:
                base_labels.update(additional_labels)
            
            # Process each metric
            for metric_name, metric_value in metrics_dict.items():
                # Convert numpy types to Python types
                converted_value = self._convert_numpy_to_python(metric_value)
                
                # Skip if still NaN after conversion
                if pd.isna(converted_value):
                    logger.warning(f"Skipping metric {metric_name} due to NaN value")
                    continue
                
                # Parse metric name
                base_name, version, statistic = self._parse_metric_name(metric_name)
                
                # Create labels specific to this metric
                metric_labels = base_labels.copy()
                metric_labels.update({
                    "metric_name": base_name,
                    "version": version,
                    "statistic": statistic
                })
                
                # Choose appropriate gauge based on metric type
                if "toxicity" in base_name:
                    gauge = self.toxicity_gauge
                elif any(readability_term in base_name for readability_term in ["flesch", "ari", "grade"]):
                    gauge = self.readability_gauge
                elif "similarity" in base_name:
                    gauge = self.similarity_gauge
                else:
                    gauge = self.general_gauge
                
                # Send the metric
                gauge.set(converted_value, attributes=metric_labels)
                logger.info(f"Sent metric: {metric_name} = {converted_value} with labels: {metric_labels}")
            
            logger.info(f"Successfully sent {len(metrics_dict)} metrics to OTel Collector")
            
        except Exception as e:
            logger.error(f"Error sending metrics to OTel Collector: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the metrics provider and flush any remaining metrics."""
        try:
            # Force a final export
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, '_metric_readers'):
                for reader in meter_provider._metric_readers:
                    if hasattr(reader, 'force_flush'):
                        reader.force_flush()
            logger.info("OTel metrics exporter shutdown complete")
        except Exception as e:
            logger.error(f"Error during OTel metrics shutdown: {e}")


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
        enable_otel_metrics: bool = True,
        otel_endpoint: str = "http://localhost:4317",
    ):
        """
        Initialize the evaluator with OpenAI credentials

        Args:
            openai_api_key: OpenAI API key for judge model
            openai_model: OpenAI model to use as judge (default: gpt-4)
            teacher_model_temperature: Temperature for teacher model responses
            teacher_model_max_tokens: Max tokens for teacher model responses
            enable_otel_metrics: Whether to enable OpenTelemetry metrics export
            otel_endpoint: OTLP endpoint for OpenTelemetry Collector
        """
        self.openai_model = "llama3.1:8b"
        
        # Synchronous client
        self.teacher_model_client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAPI_KEY"),
            max_retries=0,
            # timeout=custom_timeout,
        )
        
        # Async client for concurrent operations
        self.async_teacher_model_client = AsyncOpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAPI_KEY"),
            max_retries=0,
            timeout=httpx.Timeout(30.0, read=180.0),
        )
        # self.openai_model = openai_model
        # self.teacher_model_client = OpenAI(
        #     api_key=openai_api_key,
        #     timeout=httpx.Timeout(30.0, read=180.0),
        # )
        self.teacher_model_temperature = teacher_model_temperature
        self.teacher_model_max_tokens = teacher_model_max_tokens
        
        # Initialize OpenTelemetry metrics exporter if enabled
        self.otel_metrics_exporter = None
        if enable_otel_metrics:
            try:
                self.otel_metrics_exporter = OTelMetricsExporter(
                    otlp_endpoint=otel_endpoint,
                    service_name="llm-evaluation-service",
                    service_version="1.0.0"
                )
                logger.info("OpenTelemetry metrics exporter initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OTel metrics exporter: {e}")
                self.otel_metrics_exporter = None
        
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
        
        # mlflow.pytorch.autolog()
        # mlflow.tensorflow.autolog()
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
    async def generate_expected_response_with_judge_model_async(self, question: str) -> str:
        """
        Async version: Generate expected/reference response using GPT-4 for evaluation purposes.
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
            response = await self.async_teacher_model_client.chat.completions.create(
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
    async def generate_expected_responses_async(self, questions: List[str]) -> List[str]:
        """
        Async version: Generate expected responses for a list of questions concurrently.
        This can significantly speed up processing when multiple questions are involved.

        Args:
            questions: List of questions to generate expected responses for

        Returns:
            List of expected responses (single line format)
        """
        logger.info(f"Generating expected responses using {self.openai_model} concurrently...")
        
        # Create concurrent tasks for all questions
        tasks = [
            self.generate_expected_response_with_judge_model_async(question)
            for question in questions
        ]
        
        # Execute all tasks concurrently
        expected_responses_judge = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and convert to strings
        processed_responses = []
        for i, response in enumerate(expected_responses_judge):
            if isinstance(response, Exception):
                logger.error(f"Error generating response for question {i+1}: {response}")
                processed_responses.append("")
            else:
                processed_responses.append(response)
                logger.info(f"Generated expected response {i+1} length: {len(response)} characters")

        # Convert to single line format
        expected_responses_judge = [
            self.to_single_line(resp) for resp in processed_responses
        ]
        logger.info(f"Generated {len(expected_responses_judge)} expected responses concurrently")

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
            # trace_metadata = {}            
            # # Add metrics to metadata
            # for metric_key, metric_value in results.metrics.items():
            #     trace_metadata[f"metric.{metric_key}"] = str(metric_value)
            # trace_metadata["mlflow.trace.runid"] = run.info.run_id
            # mlflow.update_current_trace(metadata=trace_metadata)
            
            run_id = run.info.run_id

        mlflow.end_run()
        logger.info("Evaluation complete.")
        logger.info(f"Metrics:\n{results.metrics}")

        # Send metrics to OpenTelemetry Collector if exporter is available
        if self.otel_metrics_exporter:
            try:
                additional_labels = {
                    "model_type": "text",
                    "judge_model": self.openai_model,
                    "num_questions": str(len(questions))
                }
                
                self.otel_metrics_exporter.send_metrics(
                    metrics_dict=results.metrics,
                    run_id=run_id,
                    session_id=session_id,
                    user_id=user_id,
                    additional_labels=additional_labels
                )
                logger.info("Metrics successfully sent to OpenTelemetry Collector")
            except Exception as e:
                logger.error(f"Failed to send metrics to OTel Collector: {e}")

        return {"metrics": results.metrics, "run_id": run_id, "results": results}

    @mlflow.trace
    async def evaluate_responses_async(
        self,
        questions: List[str],
        generated_responses: List[str],
        expected_responses: List[str] = None,
        use_judge_model: bool = True,
        session_id: str = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Async version: Evaluate generated responses using MLflow with concurrent judge model calls.
        This can significantly reduce execution time when using judge model for multiple questions.

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
        
        logger.info("Evaluating responses asynchronously...")
        # Generate expected responses if not provided (this is the async part)
        if expected_responses is None and use_judge_model:
            expected_responses = await self.generate_expected_responses_async(questions)
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

        # Run evaluation (MLflow evaluate is synchronous, but that's fine since it's the final step)
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
            
            run_id = run.info.run_id

        mlflow.end_run()
        logger.info("Async evaluation complete.")
        logger.info(f"Metrics:\n{results.metrics}")

        # Send metrics to OpenTelemetry Collector if exporter is available
        if self.otel_metrics_exporter:
            try:
                additional_labels = {
                    "model_type": "text",
                    "judge_model": self.openai_model,
                    "num_questions": str(len(questions)),
                    "async_mode": "true"
                }
                
                self.otel_metrics_exporter.send_metrics(
                    metrics_dict=results.metrics,
                    run_id=run_id,
                    session_id=session_id,
                    user_id=user_id,
                    additional_labels=additional_labels
                )
                logger.info("Metrics successfully sent to OpenTelemetry Collector")
            except Exception as e:
                logger.error(f"Failed to send metrics to OTel Collector: {e}")

        return {"metrics": results.metrics, "run_id": run_id, "results": results}
    
    def shutdown(self):
        """Shutdown the evaluator and clean up resources."""
        if self.otel_metrics_exporter:
            try:
                self.otel_metrics_exporter.shutdown()
                logger.info("LLMEvaluator shutdown complete")
            except Exception as e:
                logger.error(f"Error during LLMEvaluator shutdown: {e}")
        
        # Close async client if it exists
        if hasattr(self, 'async_teacher_model_client') and self.async_teacher_model_client:
            try:
                # For AsyncOpenAI, we need to close it properly
                # Since we're in a sync context, we'll run the close in an event loop
                if hasattr(self.async_teacher_model_client, 'close'):
                    # Try to close it in the current event loop, or create a new one if needed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, schedule the close for later
                            loop.create_task(self.async_teacher_model_client.close())
                            logger.info("Async OpenAI client close scheduled")
                        else:
                            # If no loop or loop not running, run it directly
                            loop.run_until_complete(self.async_teacher_model_client.close())
                            logger.info("Async OpenAI client closed")
                    except RuntimeError:
                        # No event loop, create a new one
                        asyncio.run(self.async_teacher_model_client.close())
                        logger.info("Async OpenAI client closed with new event loop")
            except Exception as e:
                logger.error(f"Error closing async OpenAI client: {e}")
                logger.info("Consider using shutdown_async() for proper async cleanup")

    async def shutdown_async(self):
        """Async version of shutdown for proper async client cleanup."""
        if self.otel_metrics_exporter:
            try:
                self.otel_metrics_exporter.shutdown()
                logger.info("LLMEvaluator shutdown complete")
            except Exception as e:
                logger.error(f"Error during LLMEvaluator shutdown: {e}")
        
        # Properly close async client
        if hasattr(self, 'async_teacher_model_client') and self.async_teacher_model_client:
            try:
                await self.async_teacher_model_client.close()
                logger.info("Async OpenAI client closed")
            except Exception as e:
                logger.error(f"Error closing async OpenAI client: {e}")


# <TODO>: This is not working as expected. Need to fix this.
def get_trace_for_active_model(n: int = 1):
    """Get traces for the active model"""
    active_model_id = mlflow.get_active_model_id()
    return mlflow.search_traces(model_id=active_model_id)
