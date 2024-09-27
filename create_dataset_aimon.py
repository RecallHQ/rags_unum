# Import necessary libraries
from dotenv import load_dotenv

import csv
import json
import os
import re

from text_rag import search_knowledge_base
# Load environment variables
load_dotenv()

from openai import OpenAI
from aimon import Client

media_label = "Google I/O 2024"
media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
input_dataset_file = f"{media_label_path}_text_qa_dataset.json"
output_dataset_file = f"{media_label_path}_09-25-2024_eval.csv"
dataset_name = f"{media_label_path}_eval-09-25-2024"
dataset_collection_name = f"{media_label_path}_collection_eval-09-25-2024"

client = OpenAI()
# Instantiate the AIMon client
aimon_api_key = os.getenv("AIMON_API_KEY")
aimon_client = Client(auth_header=f"Bearer {aimon_api_key}")


def create_aimon_dataset(dataset):
    print(f"Creating aimon dataset: {media_label_path}_qa_pairs")

    dataset_data_description = json.dumps({
        "name": dataset_name,
        "description": f"This is the {media_label} dataset"
    })
    dataset_rows = []
    for item in dataset:
        relevant_docs, output = search_knowledge_base(item["question"], media_label)
        context_docs = "\n".join([relevant_docs[i].node.get_content() for i in range(len(relevant_docs))])
        dataset_rows.append({"user_query": item["question"], "context_docs": context_docs, "output": output})
       
    with open(output_dataset_file, 'w') as dataset_file:
        writer = csv.DictWriter(dataset_file, delimiter=',', fieldnames=['user_query', 'context_docs', 'output'])
        writer.writeheader()
        writer.writerows(dataset_rows)

    with open(output_dataset_file, 'rb') as dataset_file:
        dataset1 = aimon_client.datasets.create(
            file=dataset_file,
            json_data=dataset_data_description
        )
    aimon_dataset_obj = aimon_client.datasets.list(name=dataset_name)

    # Create a new dataset collection
    dataset_collection = aimon_client.datasets.collection.create(
        name=dataset_collection_name, 
        dataset_ids=[aimon_dataset_obj.sha], 
        description=f"This is a collection of a single {media_label} dataset."
    )

# Generate dataset
if os.path.exists(input_dataset_file):
    # Load dataset from local file if it exists
    with open(input_dataset_file, 'r') as f:
        dataset = json.load(f)
        create_aimon_dataset(dataset)
else:
    print(f"Dataset file {dataset_file} does not exist to generate AIMon dataset")
    

        


