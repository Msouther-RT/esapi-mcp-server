import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse
from tqdm import tqdm

def load_dataset(json_file_path):
    """Load the JSON dataset"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def embed_instructions(dataset_path, output_dir="embeddings", model_name="Qwen/Qwen3-Embedding-0.6B"):
    """
    Embed all instructions from the dataset and save embeddings + metadata
    
    Args:
        dataset_path: Path to JSON file containing instruction-response pairs
        output_dir: Directory to save embeddings and metadata
        model_name: Sentence transformer model to use
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
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
    
    # Embed all instructions
    print("Embedding instructions...")
    # Note: We don't use prompt_name="query" here because these are the documents we're searching through
    # The query prompt will be used when embedding the incoming LLM queries
    instruction_embeddings = model.encode(instructions, show_progress_bar=True)
    
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
    parser = argparse.ArgumentParser(description="Embed instructions from API dataset")
    parser.add_argument("dataset_path", help="Path to JSON dataset file")
    parser.add_argument("--output_dir", default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model to use")
    
    args = parser.parse_args()
    
    embed_instructions(args.dataset_path, args.output_dir, args.model)

if __name__ == "__main__":
    main()