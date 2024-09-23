import argparse
import os
import re

import openai
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

from utils import load_state
from constants import KNOWLEDGE_BASE_PATH

# Load environment variables
load_dotenv()


def load_knowledge_base(media_label):
    # load knowledge base
    knowledge_base = load_state(KNOWLEDGE_BASE_PATH)[media_label]
    input_data = {
        os.path.join(os.path.dirname(os.getcwd()), "recallhq", text_path.replace("./", "").replace("//", "/")): media_label
        for text_path in knowledge_base["text_paths"]
    }
    print(f"Knowledge base loaded: {input_data}")
    return input_data

def load_documents(input_data):
    reader = SimpleDirectoryReader(input_files=input_data.keys())
    documents = []
    for doc in reader.load_data():
        # Assuming doc has a filename or similar attribute
        file_path = doc.metadata.get("file_path", None)  # or however the source is defined
        if file_path is not None:
            doc.metadata["media_label"] = input_data[file_path]
        documents.append(doc)
        
    return documents

def search_knowledge_base(query, media_label):
    print(f"Query: {query} Media label: {media_label}")
    
    media_label_path = re.sub(r'[^a-zA-Z0-9]', '_', media_label)
    storage_path=os.path.join(os.path.dirname(os.getcwd()), "recallhq", os.getenv("LOCALVS_PATH"), media_label_path)
    print(f"Loading index from storage: {storage_path}")
    
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    index = load_index_from_storage(storage_context, embed_model=embedding_model)
    
    print(f"Index count: {len(index.docstore.docs)}")
    retriever = index.as_retriever(retrieval_mode='hybrid', k=10)
    print(f"Query: {query} Media label: {media_label}")
    
    relevant_docs = retriever.retrieve(query)

    print(f"Number of relevant documents: {len(relevant_docs)}")
    print("\n" + "="*50 + "\n")

    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1}:")
        print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
        print(f"Metadata: {doc.node.metadata}")
        print(f"Score: {doc.score}")
        print("\n" + "="*50 + "\n")

    client = openai.OpenAI()
        
    prompt = f"""
        Given the following context, answer the question:
        {relevant_docs}
        Question: {query}
        """
        
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
    response_text = response.choices[0].message.content
    return response_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG on Knowledge Base.")

    parser.add_argument('-q', '--query', type=str, required=True, 
                        help="Query to search for in the knowledge base")
    
    parser.add_argument('-m', '--media_label', type=str, required=True, 
                        help="Media label to search in the knowledge base")

    args = parser.parse_args()

    print("Query:", args.query)
    print("Media label: ", args.media_label)
    answer = search_knowledge_base(args.query, args.media_label)
    print(f"Answer: {answer}")
