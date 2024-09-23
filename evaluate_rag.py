import os
import re
from langfuse import Langfuse
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from text_rag import search_knowledge_base

load_dotenv()

media_label = "Google I/O 2024"
media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)

dataset_name = f"{media_label_path}_qa_pairs"
experiment_name = f"{media_label_path}_RAG_exp 1"

langfuse = Langfuse()

# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_evaluation(output, expected_output):
    client = openai.OpenAI()
    
    prompt = f"""
    Compare the following output with the expected output and evaluate its accuracy:
    
    Output: {output}
    Expected Output: {expected_output}
    
    Provide a score (0 for incorrect, 1 for correct) and a brief reason for your evaluation.
    Return your response in the following JSON format:
    {{
        "score": 0 or 1,
        "reason": "Your explanation here"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the accuracy of responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    if "```json" in evaluation:
        evaluation = evaluation.replace("```json", "").replace("```", "").strip()
    result = eval(evaluation)  # Convert the JSON string to a Python dictionary
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

from datetime import datetime
 
def rag_query(input):
  
  generationStartTime = datetime.now()

  output = search_knowledge_base(input, media_label)

  print(output)
 
  langfuse_generation = langfuse.generation(
    name=f"{media_label_path}_qa",
    input=input,
    output=output,
    model="gpt-3.5-turbo",
    start_time=generationStartTime,
    end_time=datetime.now()
  )
 
  return output, langfuse_generation

def run_experiment(experiment_name):
  dataset = langfuse.get_dataset(dataset_name)
 
  for item in dataset.items:
    completion, langfuse_generation = rag_query(item.input)
 
    item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id
 
    score, reason = llm_evaluation(completion, item.expected_output)
    langfuse_generation.score(
      name="accuracy",
      value=score,
      comment=reason
    )

run_experiment(experiment_name)