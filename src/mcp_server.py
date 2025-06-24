import pickle
import numpy as np
import sys
import asyncio
import threading
from sentence_transformers import SentenceTransformer
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with a descriptive name
mcp = FastMCP("api_rag")

# Global variables to store our embeddings and metadata
model = None           # The sentence transformer model for embedding queries
embeddings = None      # Pre-computed embeddings of all API instructions
metadata = None        # The original instructions and their JSON responses
model_loading = True   # Flag to track if model is still loading
loading_error = None   # Store any loading errors

def load_model_async():
    """Load the model in a background thread to avoid blocking server startup"""
    global model, model_loading, loading_error
    
    try:
        print("Loading embedding model in background...", file=sys.stderr)
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("Embedding model loaded successfully!", file=sys.stderr)
        model_loading = False
    except Exception as e:
        loading_error = str(e)
        model_loading = False
        print(f"Error loading model: {e}", file=sys.stderr)

def load_embeddings_and_metadata(embeddings_dir="embeddings"):
    """
    Load the pre-computed embeddings and metadata from disk.
    This function runs once when the server starts up.
    
    Args:
        embeddings_dir: Directory containing the pickle files created by embed_dataset.py
    """
    global embeddings, metadata
    
    # Convert string path to Path object for easier file handling
    embeddings_path = Path(embeddings_dir)
    
    # Check if embeddings directory exists
    if not embeddings_path.exists():
        print(f"Error: Embeddings directory '{embeddings_dir}' not found", file=sys.stderr)
        print(f"Current working directory: {Path.cwd()}", file=sys.stderr)
        return False
    
    # Check if embeddings file exists
    embeddings_file = embeddings_path / "instruction_embeddings.pkl"
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}", file=sys.stderr)
        return False
    
    # Load the pre-computed embeddings (numpy array of vectors)
    print("Loading pre-computed embeddings...", file=sys.stderr)
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Check if metadata file exists
    metadata_file = embeddings_path / "metadata.pkl"
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}", file=sys.stderr)
        return False
    
    # Load the metadata (original instructions and their JSON responses)
    print("Loading metadata...", file=sys.stderr)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Successfully loaded {len(metadata['instructions'])} API examples", file=sys.stderr)
    return True

def find_similar_examples(query: str, top_k=3):
    """
    Find the most similar API examples to the user's query.
    
    Args:
        query: The question/query from the LLM (e.g., "How to call Equals method?")
        top_k: How many similar examples to return (default 3)
    
    Returns:
        List of dictionaries containing the most relevant API documentation
    """
    # Step 1: Embed the incoming query using the same model
    print(f"Embedding query: {query}", file=sys.stderr)
    query_embedding = model.encode([query], prompt_name="query")
    
    # Step 2: Calculate similarity between query and all pre-computed embeddings
    similarities = model.similarity(query_embedding, embeddings)
    
    # Convert to numpy array and flatten to handle any shape issues
    similarities = np.array(similarities).flatten()
    
    # Step 3: Get the indices of the most similar examples
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Step 4: Retrieve the actual API documentation for the most similar examples
    results = []
    for idx in top_indices:
        similarity_score = float(similarities[idx])
        instruction = metadata['instructions'][idx]
        response = metadata['responses'][idx]
        
        results.append({
            'instruction': instruction,
            'response': response,
            'similarity_score': similarity_score
        })
        
        print(f"Found similar example: '{instruction}' (similarity: {similarity_score:.3f})", file=sys.stderr)
    
    return results

@mcp.tool()
async def search_api_examples(query: str) -> str:
    """
    Search for relevant API examples based on a natural language query.
    
    This tool takes a question about API usage and returns the 3 most relevant
    examples from the pre-trained dataset, formatted as JSON.
    
    Examples of good queries:
    - "how to get DVH data"
    - "tell me about the Patient class" 
    - "how to access beam information"
    - "working with dose volume histogram"
    
    Args:
        query: Natural language question about API usage (keep it simple and focused)
    
    Returns:
        Formatted string containing the most relevant API examples with their JSON documentation
    """
    # Check if model is still loading
    if model_loading:
        return "🔄 The embedding model is still loading in the background. Please wait a moment and try again."
    
    # Check if there was a loading error
    if loading_error:
        return f"❌ Error loading the embedding model: {loading_error}. Please restart the server."
    
    # Make sure embeddings are loaded
    if embeddings is None or metadata is None or model is None:
        return "❌ Error: Embeddings not loaded. Please restart the server."
    
    try:
        # Find the most similar API examples to the user's query
        similar_examples = find_similar_examples(query, top_k=3)
        
        # Format the results for return to the LLM
        result_parts = []
        result_parts.append(f"Found {len(similar_examples)} relevant API examples for: '{query}'\n")
        
        # Add each similar example with its API documentation
        for i, example in enumerate(similar_examples, 1):
            result_parts.append(f"--- Example {i} (Similarity: {example['similarity_score']:.3f}) ---")
            result_parts.append(f"Question: {example['instruction']}")
            result_parts.append(f"API Documentation: {example['response']}")
            result_parts.append("")  # Empty line for readability
        
        # Join all parts into a single string response
        return "\n".join(result_parts)
        
    except Exception as e:
        print(f"Error in search_api_examples: {str(e)}", file=sys.stderr)
        return f"Error processing query: {str(e)}"

@mcp.tool()
async def check_model_status() -> str:
    """
    Check the current status of the embedding model loading.
    
    Returns:
        Status message indicating whether the model is ready, loading, or encountered an error
    """
    if model_loading:
        return "🔄 Embedding model is still loading..."
    elif loading_error:
        return f"❌ Model loading failed: {loading_error}"
    elif model is not None:
        return "✅ Embedding model is ready!"
    else:
        return "❓ Unknown model status"
    

# Add this to your existing script after the other global variables
templates = None  # Will store loaded templates

def load_templates(templates_dir="templates"):
    """
    Load ESAPI script templates from disk.
    
    Expected directory structure:
    templates/
    ├── single_file.cs
    ├── binary_plugin.cs
    ├── binary_plugin.csproj
    ├── executable_script.cs
    └── executable_script.csproj
    """
    global templates
    
    templates_path = Path(templates_dir)
    
    if not templates_path.exists():
        print(f"Warning: Templates directory '{templates_dir}' not found", file=sys.stderr)
        return False
    
    templates = {
        'single_file': {
            'description': 'Single C# file script for simple ESAPI operations',
            'files': {}
        },
        'binary_plugin': {
            'description': 'Binary plugin with .cs and .csproj files for compiled ESAPI plugins',
            'files': {}
        },
        'executable_script': {
            'description': 'Executable script with .cs and .csproj files for standalone ESAPI applications',
            'files': {}
        }
    }
    
    # Load templates
    for template_type in templates.keys():
        # Load .cs file
        cs_file = templates_path / f"{template_type}.cs"
        if cs_file.exists():
            with open(cs_file, 'r', encoding='utf-8') as f:
                templates[template_type]['files']['cs'] = f.read()
        
        # Load .csproj file (if it exists)
        csproj_file = templates_path / f"{template_type}.csproj"
        if csproj_file.exists():
            with open(csproj_file, 'r', encoding='utf-8') as f:
                templates[template_type]['files']['csproj'] = f.read()
    
    print(f"Loaded templates for: {list(templates.keys())}", file=sys.stderr)
    return True

@mcp.tool()
async def get_esapi_template(script_type: str = "single_file") -> str:
    """
    Get ESAPI script template(s) for the specified script type.
    
    This tool returns the appropriate C# template(s) and project files for creating
    ESAPI scripts of different types.
    
    Args:
        script_type: Type of script template to retrieve. Options:
                    - "single_file": Simple single C# file script
                    - "binary_plugin": Binary plugin with .cs and .csproj
                    - "executable_script": Executable script with .cs and .csproj
                    - "list": Show all available template types
    
    Returns:
        Formatted string containing the template files and descriptions
    """
    if templates is None:
        return "❌ Error: Templates not loaded. Please restart the server."
    
    # Handle list request
    if script_type.lower() == "list":
        result_parts = ["Available ESAPI Script Templates:\n"]
        for temp_type, temp_data in templates.items():
            file_types = list(temp_data['files'].keys())
            result_parts.append(f"• {temp_type}: {temp_data['description']}")
            result_parts.append(f"  Files: {', '.join(file_types)}")
            result_parts.append("")
        return "\n".join(result_parts)
    
    # Get specific template
    if script_type not in templates:
        available = ", ".join(templates.keys())
        return f"❌ Error: Template type '{script_type}' not found. Available types: {available}"
    
    template_data = templates[script_type]
    
    if not template_data['files']:
        return f"❌ Error: No template files found for '{script_type}'"
    
    # Format the response
    result_parts = []
    result_parts.append(f"ESAPI {script_type.replace('_', ' ').title()} Template")
    result_parts.append(f"Description: {template_data['description']}\n")
    
    # Add each file
    for file_type, content in template_data['files'].items():
        file_ext = file_type
        result_parts.append(f"--- {script_type}.{file_ext} ---")
        result_parts.append(content)
        result_parts.append("")  # Empty line between files
    
    return "\n".join(result_parts)



if __name__ == "__main__":
    # Load embeddings and metadata first (fast operation)
    print("Starting API RAG MCP server...", file=sys.stderr)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Go up one level to project root, then find embeddings
    project_root = script_dir.parent  # This goes from src/ to project root
    embeddings_dir = project_root / "embeddings"
    templates_dir = project_root / "templates"
    
    print(f"Script directory: {script_dir}", file=sys.stderr)
    print(f"Project root: {project_root}", file=sys.stderr)
    print(f"Looking for embeddings in: {embeddings_dir}", file=sys.stderr)
    print(f"Looking for templates in: {templates_dir}", file=sys.stderr)
    
    # Try to load embeddings, exit if failed
    if not load_embeddings_and_metadata(str(embeddings_dir)):
        print("Failed to load embeddings. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Load templates
    load_templates(str(templates_dir))
    
    # Start loading the model in a background thread
    print("Starting model loading in background thread...", file=sys.stderr)
    model_thread = threading.Thread(target=load_model_async, daemon=True)
    model_thread.start()
    
    # Start the MCP server immediately (model will load in background)
    print("MCP server ready. Model loading in background...", file=sys.stderr)
    print("You can check model status using the 'check_model_status' tool.", file=sys.stderr)
    mcp.run(transport='stdio')
