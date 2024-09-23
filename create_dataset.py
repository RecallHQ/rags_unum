# Import necessary libraries
from dotenv import load_dotenv

import os
import json
import re

from text_rag import load_knowledge_base, load_documents
# Load environment variables
load_dotenv()


from openai import OpenAI
import json

# Note: we're choosing to create the dataset in Langfuse below, but it's equally easy to create it in another platform.
from langfuse import Langfuse

media_label = "Google I/O 2024"
media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)

client = OpenAI()

# Function to generate questions and answers
def generate_qa(prompt, text, temperature=0.2):    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}],
        temperature=temperature,
    )
    
    print(response.choices[0].message.content)

    # Strip extraneous symbols from the response content
    content = response.choices[0].message.content.strip()
    
    # Remove potential JSON code block markers
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
    if content.endswith('```'):
        content = content.rsplit('\n', 1)[0]
    content = content.strip()
    
    # Attempt to parse the cleaned content as JSON
    try:
        parsed_content = json.loads(content.strip())
        return parsed_content
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON. Raw content:")
        #print(content)
        return []

def create_langfuse_dataset(dataset):
    print(f"Creating langfuse dataset: {media_label_path}_qa_pairs")
    langfuse = Langfuse()

    dataset_name = f"{media_label_path}_qa_pairs"

    langfuse.create_dataset(name=dataset_name);

    for item in dataset:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=item["expected_output"]
   )

factual_prompt = """
You are an expert tech industry content creator tasked with generating factual questions and answers of the Google I/O 2024 event based on the following document excerpt. These questions should focus on retrieving specific details, figures, definitions, and key facts from the text.

Instructions:

- Generate **5** factual questions, each with a corresponding **expected_output**.
- Ensure all questions are directly related to the document excerpt. The questions should be analytical and thought provoking.
- Present the output in the following structured JSON format:

[
  {
    "question": "What are the key highlights of the event?",
    "expected_output": "The key highlights of the event are Gemini Models: Introduction of Gemini 1.5 Pro and Gemini 1.5 Flash, emphasizing their speed and efficiency, particularly for tasks requiring low latency.
AI Integration: The keynote showcased how AI is being integrated across various Google products, enhancing capabilities in search, workspace, photos, and Android.
Developer Program: Launch of the Google Developer Program, offering new benefits at no cost, including access to Gemini for learning and documentation, expanded workstations for IDX users, and credits for interactive labs on Google Cloud Skills Boost.
Support for Startups: Highlighting the success of Google accelerators, which have supported over 1,300 startups globally, including notable successes like You Genie AI, which focuses on reducing carbon emissions using AI.
Community Engagement: Announcement of upcoming IO connect events in Berlin, Bangalore, and Beijing, along with community-led IO extended events.
Developer Tools: Updates to Android Studio to leverage Gemini 1.5 Pro for improved app development.
Live Demonstrations: The keynote included live demos showcasing new features and capabilities, particularly in accessibility and real-world applications of AI.
Exciting Atmosphere: The event was described as a developer festival with a vibrant atmosphere, featuring workshops, sessions, and hands-on experiences for attendees.
Overall, the keynote emphasized Google's commitment to empowering developers and enhancing user experiences through innovative AI technologies."
  },
  {
    "question": "Who were the presenters at the event?",
    "expected_output": "The presenters mentioned in the different talks at Google I/O 2024 include:
Marvin Chao - Marketing lead who discussed the significance of Google I/O.
Mark Rubier - Musician who hosted the I-O Pre-Show and used the new DJ mode in the music FX tool.
Sundar Pichai - Kicked off the Google keynote, sharing progress and advancements in AI.
Janine Banks - Led the Developer X and Core team, announcing updates and new tools for developers.
Seneca Meeks - Software engineer on the Quantum AI team, who presented a demonstration related to quantum computing.
Josh Woodward - VP of Google Labs, who discussed the Googlers-only demo slam."
  }
]
"""
# Generate dataset


dataset_file = f"{media_label_path}_text_qa_dataset.json"

if os.path.exists(dataset_file):
    # Load dataset from local file if it exists
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        create_langfuse_dataset(dataset)
else:
    
    input_data = load_knowledge_base(media_label)
    documents = load_documents(input_data)
    print(f"Number of documents: {len(documents)}")
    # Generate dataset if local file doesn't exist
    dataset = []
    for doc in documents:
        qa_pairs = generate_qa(factual_prompt, doc.text, temperature=0.2)
        dataset.extend(qa_pairs)
    
    # Write dataset to local file
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f)

        


