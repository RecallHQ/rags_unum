import os
import re
import csv

from dotenv import load_dotenv

from aimon import Client, AnalyzeEval, Application, Model


load_dotenv()


media_label = "Google I/O 2024"
media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
input_dataset_file = f"{media_label_path}_09-25-2024_eval.csv"
dataset_name = f"{media_label_path}_eval-09-25-2024"
dataset_collection_name = f"{media_label_path}_collection_eval-09-25-2024"

aimon_config = {"hallucination": {"detector_name": "default"},
                "completeness": {"detector_name": "default"},
                "toxicity": {"detector_name": "default"},
                "conciseness": {"detector_name": "default"},
                #"instruction_adherence": {"detector_name": "default"}
                }


# Instantiate the AIMon client
aimon_api_key = os.getenv("AIMON_API_KEY")
aimon_client = Client(auth_header=f"Bearer {aimon_api_key}")


analyze_eval = AnalyzeEval(
    Application(f"{media_label_path}_v0"),
    Model("gpt-4o_mini", "GPT-4o-mini"), 
    api_key=os.getenv("AIMON_API_KEY"),
    evaluation_name=f"{media_label_path}_eval_v2", 
    dataset_collection_name=dataset_collection_name,
    headers=["context_docs", "user_query", "output"],
    config=aimon_config
)

@analyze_eval
def run_evaluation(context_docs=None, user_query=None, output=None):
  # Open the CSV file
    return output

aimon_eval_res = run_evaluation()
print(aimon_eval_res)