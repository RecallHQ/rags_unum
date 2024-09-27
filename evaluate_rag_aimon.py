import os
import re
import csv

from dotenv import load_dotenv

from aimon import Client

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


def create_evaluation():
    dataset_collection = aimon_client.datasets.collection.retrieve(name=dataset_collection_name)
    gpt4_model = aimon_client.models.create(
      name="gpt-4o_mini", 
      type="GPT-4o-mini", 
      description="This model is a GPT4o-mini model"
    )
    #Using the AIMon client, create or get an existing application
    new_app = aimon_client.applications.create(
        name=f"{media_label_path}_v0", 
        model_name=gpt4_model.name, 
        stage="evaluation",
        type="question_answering"
    )
   # Using the AIMon client, create a new evaluation
    evaluation = aimon_client.evaluations.create(
        name=f"{media_label_path}_eval_v1", 
        application_id=new_app.id, 
        model_id=gpt4_model.id, 
        dataset_collection_id=dataset_collection.id
    )

    # Using the AIMon client, create a new evaluation run. 
    eval_run = aimon_client.evaluations.run.create(
        evaluation_id=evaluation.id
    )

    return eval_run, new_app

def run_evaluation(eval_run, new_app):
  # Open the CSV file
  with open(input_dataset_file, mode='r') as file:
      # Create a CSV reader object
      reader = csv.DictReader(file)
      # Iterate through the rows one by one
      for row in reader:
          payload = {
              "application_id": new_app.id,
              "version": new_app.version,
              "user_query": row["user_query"],
              "context_docs": row["context_docs"],
              "output": row["output"],
              "evaluation_id": eval_run.evaluation_id,
              "evaluation_run_id": eval_run.id,
              "config": aimon_config
          }
          res = aimon_client.analyze.create(
              body=[payload]
          )
          print("Aimon response: {}\n".format(res))

eval_run, new_app = create_evaluation()
run_evaluation(eval_run, new_app)