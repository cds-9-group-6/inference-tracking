import os
import mlflow
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import httpx
from mlflow.tracking import MlflowClient
import logging
from mlflow.metrics.genai import EvaluationExample, faithfulness, answer_similarity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

def check_env_vars():
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

check_env_vars()

def setup_mlflow(tracking_uri, mlflow_experiment_name, active_model_name=None):
    """
    function to setup mlflow
    """

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable must be set")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = mlflow_experiment_name
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)

    if exp is None:
        mlflow.set_experiment(exp_name)
    elif exp.lifecycle_stage == "deleted":
        logger.info("The experiment is deleted. Restoring it...")
        client.restore_experiment(exp.experiment_id)
        mlflow.set_experiment(exp_name)
    else:
        mlflow.set_experiment(exp_name)

    if active_model_name:
        mlflow.set_active_model(name=active_model_name)

    logger.info(f"tracking uri: {mlflow.get_tracking_uri()}")
    logger.info(f"experiment: {mlflow.get_experiment_by_name(mlflow_experiment_name)}")
    logger.info(f"active model: {mlflow.get_active_model_id()}")

# setup_mlflow(
#     tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
#     mlflow_experiment_name="QA_Evaluation",
#     # active_model_name="default_anonymous",
# )

# # Example finance QA dataset
# questions = ["What was the company’s revenue in Q2?"]
# ground_truth = ["$25.3B"]
# predicted_answers = ["$25B"]

# # mlflow.set_tracking_uri("http://0.0.0.0:5001/")
# # mlflow.set_experiment("QA_Evaluation")

# with mlflow.start_run():
#     data = pd.DataFrame(
#         {
#             "question": questions,
#             "ground_truth": ground_truth,
#             "predicted": predicted_answers,
#         }
#     )

#     # Evaluate QA model (removed default evaluators to fix hang)
#     results = mlflow.evaluate(
#         data=data,
#         targets="ground_truth",
#         predictions="predicted",
#         model_type="question-answering",
#         # evaluators=["default"],  # This was causing the hang - commented out
#     )

# print(results.metrics)

def get_trace_for_active_model(n: int = 1):
    active_model_id = mlflow.get_active_model_id()
    return mlflow.search_traces(model_id=active_model_id)


def to_single_line(s: str) -> str:
    return " ".join(s.split())

setup_mlflow(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "llm_tracing_ollama"),
    active_model_name=os.getenv("MLFLOW_ACTIVE_MODEL_NAME", "llama318b_model"),
)

# Enable MLflow automatic tracing for OpenAI
mlflow.openai.autolog()

custom_timeout = httpx.Timeout(10.0, read=180.0)
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),  # The local Ollama REST endpoint
    api_key=os.getenv("OPENAPI_KEY"),
    max_retries=0,
    timeout=custom_timeout,
)

@mlflow.trace
def chat_with_ollama(messages):
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL"),
        messages=messages,  # timeout=180
    )
    return response



questions = [
    "What is the insecticide and fungicide chemicals used for leaf disease Black Rot in Tamil Nadu, India?",
    # "What are my organic options to cure the disease Black Rot in Tamil Nadu?",
    # "will this impact my paddy plants as well?",
]
outputs = []

messages = [
    {"role": "system", "content": "You are a agriculture expert."},
    {
        "role": "user",
        "content": f"{questions[0]}",
    },
]
try:
    response = chat_with_ollama(messages)
    outputs.append(response.choices[0].message.content)
    # other_stats(response)
except Exception as e:
    logger.error(f"Error: {e}")

logger.info(f"Outputs: {outputs}")

teacher_model_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Use actual OpenAI API key
    timeout=httpx.Timeout(30.0, read=120.0),
)


@mlflow.trace
def generate_expected_response_with_judge_model(question: str) -> str:
    """
    Generate expected/reference response using GPT-4 for evaluation purposes. 
    In this call GPT-4 is used as the teacher model also called as reference model or Judge model.
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
        {"role": "user", "content": question}
    ]
    
    try:
        response = teacher_model_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        logger.info("Generating expected responses using GPT-4...")
        logger.info(f"Response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating expected response with GPT-4: {e}")
        return ""

expected_response_raw = [
    """
            Hey — to give good advice, can you tell me which crop the “black rot” is on (for example, cabbage/crucifers, grape, citrus, etc.)? Different crops + pathogens need different chemicals. Meanwhile, here’s a general idea plus what Tamil Nadu Agricultural University (TNAU) recommends (for crucifers like cabbage) for black rot and similar leaf diseases.

⸻

What is “Black Rot” generally
	•	In crucifers (cabbage, cauliflower etc.), “black rot” is caused by Xanthomonas campestris pv. campestris, a bacterial pathogen. So antibiotics or copper-based bactericides are used, not just fungicides.
	•	There are also fungal diseases called “leaf blight,” “leaf spot,” etc., so need to be clear which you mean.

⸻

TNAU recommendations (Cabbage / Crucifers in Tamil Nadu) for Black Rot (bacterial)

From TNAU’s crop protection guide for cabbage:  ￼
	•	Seed treatment: Dip seeds in streptomycin 100 parts per million (ppm) for 30 minutes.  ￼
	•	Foliar sprays: Two sprays after planting and during head formation with a mixture of copper oxychloride + streptomycin (100 ppm) each time.  ￼

So in this case, the chemical control is:
	•	Streptomycin (an antibiotic)
	•	Copper oxychloride (a copper-based bactericide/fungicide)

⸻

Other chemicals used for leaf disease control (fungal/bacterial) in Tamil Nadu or similar regions

Even though these are for leaf spot, blight, etc., many overlap with what could help prevent secondary fungal infections:
	•	Mancozeb — commonly used for leaf spot, blight, etc.  ￼
	•	Carbendazim — a systemic fungicide used in various leaf spot/blight control.  ￼
	•	Copper-based compounds — e.g. copper oxychloride, copper hydroxide. These are useful for bacterial as well as fungal diseases.  ￼

⸻

What to consider / use depending on the pathogen
	•	If it’s truly bacterial black rot (Xanthomonas in crucifers): use copper compounds + antibiotics (if allowed/legal).
	•	If fungal (leaf spot, anthracnose, etc.): use fungicides like Mancozeb, Carbendazim, or such, depending on fungicide registration and safety.
	•	Always follow label rates, waiting periods, resistance management (rotate chemicals), avoid spray drift, etc.

⸻

If you tell me the crop and symptoms (what the spots look like, if there’s a smell, whether the veins turn black, etc.), I can give you a more specific list of insecticides/fungicides/antibiotics approved in Tamil Nadu (with dosages). Do you want me to pull that together?
            
            """,
    """
            Got it 🌱 Thanks for clarifying — you’re asking about organic ways to manage black rot (most often a bacterial disease on crucifers like cabbage/cauliflower, caused by Xanthomonas campestris). Since this is bacterial, chemical fungicides don’t really help, and organic control is more about prevention + cultural practices with a few allowed sprays.

Here are your organic options, especially suited for Tamil Nadu conditions:

⸻

🌿 Organic & Eco-Friendly Management of Black Rot

1. Seed & Seedling Care
	•	Hot water seed treatment: Soak seeds in water at 50 °C for 30 minutes → kills bacteria hiding on the seed coat.
	•	Trichoderma / Pseudomonas seed treatment: Coat seeds with Pseudomonas fluorescens powder (10 g/kg seed) – this is widely used in Tamil Nadu organic farming.

2. Soil & Nursery Management
	•	Always raise seedlings in disease-free soil or trays (avoid reusing infected soil).
	•	Apply neem cake or Pseudomonas fluorescens in the nursery bed to reduce pathogen survival.

3. Field Practices
	•	Follow 3-year crop rotation (avoid planting crucifers like cabbage, cauliflower, mustard, radish in the same plot repeatedly).
	•	Keep good drainage and spacing – wet, crowded plants spread bacteria faster.
	•	Remove and burn infected leaves/plants as soon as you notice symptoms.

4. Organic Sprays (allowed in organic farming)
	•	Neem oil (2–3%): Has mild antibacterial effect and reduces pest stress (since insect wounds spread bacteria).
	•	Panchagavya or Jeevamrutham foliar sprays: Boosts plant immunity.
	•	Garlic or ginger extract sprays (10%) – traditional practice in TN, with antibacterial properties.
	•	Copper-based sprays (Bordeaux mixture / copper hydroxide): These are permitted in organic farming (under restricted use). They suppress bacterial population on leaves.

5. Biological Control
	•	Regular foliar sprays of Pseudomonas fluorescens @ 0.2% (available from TNAU and organic input shops in Tamil Nadu).
→ It competes with Xanthomonas and prevents spread.

⸻

✅ Best Integrated Organic Strategy in Tamil Nadu
	1.	Hot water treat seeds → coat with Pseudomonas fluorescens.
	2.	Apply neem cake + Pseudomonas in nursery.
	3.	Use wide spacing + good drainage.
	4.	Remove diseased plants early.
	5.	Foliar sprays of neem oil or Pseudomonas regularly, and copper hydroxide (if certified organic standards allow).

⸻

Would you like me to make a simple spray schedule (month-wise) for Tamil Nadu cabbage/cauliflower under organic management, so you can follow step by step?
            
            """,
    """
            Good question 👍 — I can see why you’re worried!

The black rot disease in cabbage/cauliflower (crucifers) is caused by the bacterium Xanthomonas campestris pv. campestris. This pathogen is host-specific — meaning it infects crucifer family plants (like cabbage, cauliflower, mustard, radish, knolkhol, etc.), but not rice (paddy).

⸻

🌾 What about Paddy?
	•	Paddy has its own bacterial diseases, like bacterial leaf blight (BLB) caused by Xanthomonas oryzae pv. oryzae.
	•	Even though both are Xanthomonas, they are different strains (pathovars) and don’t jump from cabbage to rice.

⸻

✅ So the answer is:
	•	No direct impact — your paddy plants won’t get infected from cabbage black rot.
	•	But, if you’re spraying copper-based sprays or neem oil in the same area, those won’t harm paddy (in fact, copper is sometimes used for BLB in rice too).
	•	Just be careful about crop rotation — don’t rotate crucifers with crucifers, but paddy after cabbage is totally safe.

⸻

👉 Do you want me to also share some organic tips for bacterial leaf blight in paddy (since it’s a common issue in Tamil Nadu)? That way, you’ll have both cabbage and rice covered.
            
            """,
]

# expected_response = [to_single_line(s) for s in expected_response_raw]
expected_response = [to_single_line(expected_response_raw[0])]

logger.info(f"len of array expected_response: {len(expected_response)} x len of array outputs: {len(outputs)} x len of array questions: {len(questions)}")

logger.info(f"Expected response: {expected_response}")
expected_responses_judge = []
for question in questions:
    logger.info(f"Generating expected response for: {question}")
    expected_resp = generate_expected_response_with_judge_model(question)
    expected_responses_judge.append(expected_resp)
    logger.info(f"Generated expected response length: {len(expected_resp)} characters")

expected_responses_judge = [to_single_line(resp) for resp in expected_responses_judge]
logger.info(f"len of array expected_responses_judge: {len(expected_responses_judge)} x len of array outputs: {len(outputs)} x len of array questions: {len(questions)}")


for i, resp in enumerate(expected_response):
    logger.info(f"Expected response {i+1} preview: {resp[:200]}...")

data = pd.DataFrame(
    {
        "questions": questions,
        "expected_response_from_judge": expected_responses_judge,
        "generated_response": outputs,
    }
)

# faithfulness_metric = faithfulness(model="openai:/gpt-4")
# print(faithfulness_metric)
answer_similarity_metric = answer_similarity(model="openai:/gpt-4")
logger.info(f"Using metric: {answer_similarity_metric.name}")

with mlflow.start_run() as run:
    # Evaluate QA model (removed default evaluators to fix hang)
    results = mlflow.evaluate(
        data=data,
        targets="expected_response_from_judge",
        predictions="generated_response",
        # model_type="question-answering", # we were getting exact match score when this was used.
        model_type="text",
        extra_metrics=[answer_similarity_metric],
        evaluators="default",
        evaluator_config={
        "col_mapping": {
            "inputs": "questions",
           }}
        # evaluators=["default"],  # This was causing the hang - commented out
    )

mlflow.end_run()
logger.info("Evaluation complete.")
logger.info(f"Metrics:\n{results.metrics}")


# results.tables["eval_results_table"]