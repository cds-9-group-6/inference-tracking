import os
from openai import OpenAI
from dotenv import load_dotenv
import httpx
import logging
import mlflow
from inference_tracking import LLMEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('inference_tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


def check_env_vars():
    """Check if all required environment variables are set"""
    required_env_vars = [
        "OPENAI_API_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OPENAI_MODEL",
    ]

    for env_var in required_env_vars:
        assert env_var in os.environ, f"{env_var} environment variable must be set"
    logger.info("Environment variables are set")
    return True


@mlflow.trace
def chat_with_ollama(messages, client):
    """Chat with Ollama model"""
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL"),
        messages=messages,
    )
    return response

# <TODO>: This is not working as expected. Need to fix this.
@mlflow.trace
def generate_responses_with_ollama(questions):
    """Generate responses using Ollama for a list of questions"""
    # Setup Ollama client
    custom_timeout = httpx.Timeout(10.0, read=180.0)
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        api_key=os.getenv("OPENAPI_KEY"),
        max_retries=0,
        timeout=custom_timeout,
    )
    
    outputs = []
    
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a agriculture expert."},
            {"role": "user", "content": question},
        ]
        
        try:
            response = chat_with_ollama(messages, client)
            outputs.append(response.choices[0].message.content)
            logger.info(f"Generated response for question: {question[:50]}...")
        except Exception as e:
            logger.error(f"Error generating response for question '{question}': {e}")
            outputs.append("")  # Add empty response to maintain list alignment
    
    return outputs


def main():
    """Main execution function"""
    # Check environment variables
    check_env_vars()
    
    # Initialize the evaluator
    evaluator = LLMEvaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4")
    )
    
    # Setup MLflow
    evaluator.setup_mlflow(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "llm_tracing_ollama"),
        active_model_name=os.getenv("MLFLOW_ACTIVE_MODEL_NAME", "llama318b_model")
    )
    
    # Define questions
    questions = [
        "What is the insecticide and fungicide chemicals used for leaf disease Black Rot in Tamil Nadu, India?",
        "What are my organic options to cure the disease Black Rot in Tamil Nadu?",
        "will this impact my paddy plants as well?",
    ]
    
    # Generate responses with Ollama
    logger.info("Generating responses with Ollama...")
    generated_responses = generate_responses_with_ollama(questions)
    
    logger.info(f"Generated {len(generated_responses)} responses")
    for i, response in enumerate(generated_responses):
        logger.info(f"Response {i+1} length: {len(response)} characters")
    
    # Evaluate responses
    logger.info("Starting evaluation...")
    evaluation_results = evaluator.evaluate_responses(
        questions=questions,
        generated_responses=generated_responses,
        use_judge_model=True  # This will generate expected responses using GPT-4
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Run ID: {evaluation_results['run_id']}")
    print(f"Metrics: {evaluation_results['metrics']}")
    
    # Extract specific metrics
    metrics = evaluation_results['metrics']
    if 'answer_similarity/v1/score' in metrics:
        print(f"Answer Similarity Score: {metrics['answer_similarity/v1/score']}")
    
    return evaluation_results


if __name__ == "__main__":
    results = main()
    logger.info(f"Results: {results}")