import json
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import openai
import os
import time


# Load environment variables from .env file
def load_env():
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Call this at the start of main()
load_env()

def load_dataset(json_file_path):
    """Load the JSON dataset"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=100):
    """
    Get embeddings from OpenAI API in batches
    """
    # Check for API key
    if not openai.api_key and not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable")
    
    all_embeddings = []
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        
        try:
            response = openai.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Retry individual items if batch fails
            for text in batch:
                try:
                    response = openai.embeddings.create(
                        model=model,
                        input=[text],
                        encoding_format="float"
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                    time.sleep(0.2)
                except Exception as e2:
                    print(f"Failed to embed text: {text[:50]}... Error: {e2}")
                    # Add zero vector as placeholder
                    embedding_dim = 1536  # Default for text-embedding-3-small
                    all_embeddings.append([0.0] * embedding_dim)
    
    return np.array(all_embeddings)

def embed_instructions(dataset_path, output_dir="embeddings", model_name="text-embedding-3-small"):
    """
    Embed all instructions from the dataset and save embeddings + metadata
    
    Args:
        dataset_path: Path to JSON file containing instruction-response pairs
        output_dir: Directory to save embeddings and metadata
        model_name: OpenAI embedding model to use
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    # Extract all instructions
    instructions = []
    responses = []
    
    for item in dataset:
        if 'instruction' in item and 'response' in item:
            instructions.append(item['instruction'])
            responses.append(item['response'])
        else:
            print(f"Warning: Skipping item missing 'instruction' or 'response': {item}")
    
    print(f"Found {len(instructions)} instruction-response pairs")
    
    # Embed all instructions using OpenAI
    print(f"Embedding instructions using OpenAI model: {model_name}")
    instruction_embeddings = get_openai_embeddings(instructions, model=model_name)
    
    # Save embeddings
    embeddings_file = output_path / "instruction_embeddings.pkl"
    with open(embeddings_file, 'wb') as f:
        pickle.dump(instruction_embeddings, f)
    
    # Save metadata (instructions and their corresponding responses)
    metadata_file = output_path / "metadata.pkl"
    metadata = {
        'instructions': instructions,
        'responses': responses,
        'model_name': model_name,
        'embedding_provider': 'openai',
        'num_items': len(instructions)
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Also save as JSON for human readability
    json_metadata_file = output_path / "metadata.json"
    with open(json_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Embeddings saved to: {embeddings_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Human-readable metadata saved to: {json_metadata_file}")
    print(f"Embedding shape: {instruction_embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description="Embed instructions from API dataset using OpenAI")
    parser.add_argument("dataset_path", help="Path to JSON dataset file")
    parser.add_argument("--output_dir", default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model to use")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    embed_instructions(args.dataset_path, args.output_dir, args.model)

if __name__ == "__main__":
    main()