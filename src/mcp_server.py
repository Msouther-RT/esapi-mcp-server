import pickle
import numpy as np
import sys
import asyncio
import os
import openai
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastMCP server with a descriptive name
mcp = FastMCP("api_rag")

# Global variables to store our embeddings and metadata
embeddings = None      # Pre-computed embeddings of all API instructions
metadata = None        # The original instructions and their JSON responses
embeddings_semantic = None      # Pre-computed embeddings of all API instructions semantic
metadata_semantic = None        # The original instructions and their JSON responses semantic
embeddings_PropMeth = None      # Pre-computed embeddings of property/method instructions
metadata_PropMeth = None        # The original property/method instructions and their JSON responses

def load_env():
    """Load environment variables from .env file"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Go up one level to project root where .env should be
    project_root = script_dir.parent
    env_file = project_root / '.env'
    
    print(f"Script directory: {script_dir}", file=sys.stderr)
    print(f"Project root: {project_root}", file=sys.stderr)
    print(f"Looking for .env file at: {env_file.absolute()}", file=sys.stderr)
    
    if env_file.exists():
        print("Found .env file, loading...", file=sys.stderr)
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"Loaded env var: {key}=***", file=sys.stderr)
    else:
        print("No .env file found!", file=sys.stderr)
        print(f"Current working directory: {Path.cwd()}", file=sys.stderr)
        print(f"Files in current directory: {list(Path.cwd().iterdir())}", file=sys.stderr)

def load_embeddings_and_metadata(embeddings_dir="embeddings"):
    """
    Load the pre-computed embeddings and metadata from disk.
    This function runs once when the server starts up.
    
    Args:
        embeddings_dir: Directory containing the pickle files created by embed_dataset.py
    """
    global embeddings, metadata, embeddings_semantic, metadata_semantic, embeddings_PropMeth, metadata_PropMeth
    
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
    
    embeddings_file_semantic = embeddings_path / "instruction_embeddings_semantic.pkl"
    if not embeddings_file_semantic.exists():
        print(f"Error: Embeddings_semantic file not found: {embeddings_file}", file=sys.stderr)
        return False
    
    embeddings_file_PropMeth = embeddings_path / "instruction_embeddings_PropMeth.pkl"
    if not embeddings_file_PropMeth.exists():
        print(f"Error: Embeddings_PropMeth file not found: {embeddings_file_PropMeth}", file=sys.stderr)
        return False
    
    # Load the pre-computed embeddings (numpy array of vectors)
    print("Loading pre-computed embeddings...", file=sys.stderr)
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    # Load the pre-computed embeddings (numpy array of vectors)
    print("Loading pre-computed embeddings_semantic...", file=sys.stderr)
    with open(embeddings_file_semantic, 'rb') as f:
        embeddings_semantic = pickle.load(f)
    
    # Load the pre-computed embeddings (numpy array of vectors)
    print("Loading pre-computed embeddings_PropMeth...", file=sys.stderr)
    with open(embeddings_file_PropMeth, 'rb') as f:
        embeddings_PropMeth = pickle.load(f)
    
    # Check if metadata file exists
    metadata_file = embeddings_path / "metadata.pkl"
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}", file=sys.stderr)
        return False
    
    # Check if metadata file exists
    metadata_file_semantic = embeddings_path / "metadata_semantic.pkl"
    if not metadata_file_semantic.exists():
        print(f"Error: Metadata_semantic file not found: {metadata_file_semantic}", file=sys.stderr)
        return False
    
    # Check if metadata file exists
    metadata_file_PropMeth = embeddings_path / "metadata_PropMeth.pkl"
    if not metadata_file_PropMeth.exists():
        print(f"Error: Metadata_PropMeth file not found: {metadata_file_PropMeth}", file=sys.stderr)
        return False
    
    # Load the metadata (original instructions and their JSON responses)
    print("Loading metadata...", file=sys.stderr)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    # Load the metadata (original instructions and their JSON responses)
    print("Loading metadata_semantic...", file=sys.stderr)
    with open(metadata_file_semantic, 'rb') as f:
        metadata_semantic = pickle.load(f)
    
    # Load the metadata (original instructions and their JSON responses)
    print("Loading metadata_PropMeth...", file=sys.stderr)
    with open(metadata_file_PropMeth, 'rb') as f:
        metadata_PropMeth = pickle.load(f)
    
    print(f"Successfully loaded {len(metadata['instructions'])} API examples", file=sys.stderr)
    print(f"Successfully loaded {len(metadata_semantic['instructions'])} semantic API examples", file=sys.stderr)
    print(f"Successfully loaded {len(metadata_PropMeth['instructions'])} property/method examples", file=sys.stderr)
    return True

def get_query_embedding(query: str, model="text-embedding-3-small"):
    """
    Get embedding for a query using OpenAI API
    
    Args:
        query: The query text to embed
        model: OpenAI model to use
    
    Returns:
        numpy array containing the embedding vector
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=[query],
            encoding_format="float"
        )
        embedding = np.array(response.data[0].embedding)
        return embedding.reshape(1, -1)  # Reshape for sklearn compatibility
    except Exception as e:
        print(f"Error getting OpenAI embedding: {e}", file=sys.stderr)
        raise

def get_query_embedding_semantic(query: str, model="text-embedding-3-small"):
    """
    Get embedding for a query using OpenAI API
    
    Args:
        query: The query text to embed
        model: OpenAI model to use
    
    Returns:
        numpy array containing the embedding vector
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=[query],
            encoding_format="float"
        )
        embedding_semantic = np.array(response.data[0].embedding)
        return embedding_semantic.reshape(1, -1)  # Reshape for sklearn compatibility
    except Exception as e:
        print(f"Error getting OpenAI embedding semantic: {e}", file=sys.stderr)
        raise

def find_similar_examples(query: str, top_k=3):
    """
    Find the most similar API examples to the user's query.
    
    Args:
        query: The question/query from the LLM (e.g., "How to call Equals method?")
        top_k: How many similar examples to return (default 3)
    
    Returns:
        List of dictionaries containing the most relevant API documentation
    """
    # Step 1: Embed the incoming query using OpenAI
    print(f"Embedding query: {query}", file=sys.stderr)
    query_embedding = get_query_embedding(query)
    
    # Step 2: Calculate cosine similarity between query and all pre-computed embeddings
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    
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


def find_similar_examples_semantic(query: str, top_k=3):
    """
    Find the most similar API examples to the user's query.
    
    Args:
        query: The question/query from the LLM (e.g., "How to call Equals method?")
        top_k: How many similar examples to return (default 3)
    
    Returns:
        List of dictionaries containing the most relevant API documentation
    """
    # Step 1: Embed the incoming query using OpenAI
    print(f"Embedding semantic query: {query}", file=sys.stderr)
    query_embedding = get_query_embedding_semantic(query)
    
    # Step 2: Calculate cosine similarity between query and all pre-computed embeddings
    similarities = cosine_similarity(query_embedding, embeddings_semantic).flatten()
    
    # Step 3: Get the indices of the most similar examples
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Step 4: Retrieve the actual API documentation for the most similar examples
    results = []
    for idx in top_indices:
        similarity_score = float(similarities[idx])
        instruction = metadata_semantic['instructions'][idx]
        response = metadata_semantic['responses'][idx]
        
        results.append({
            'instruction': instruction,
            'response': response,
            'similarity_score': similarity_score
        })
        
        print(f"Found similar example: '{instruction}' (similarity: {similarity_score:.3f})", file=sys.stderr)
    
    return results

def find_similar_examples_PropMeth(query: str, top_k=3):
    """
    Find the most similar property/method examples to the user's query.
    
    Args:
        query: The question/query from the LLM (e.g., "Dose property information")
        top_k: How many similar examples to return (default 3)
    
    Returns:
        List of dictionaries containing the most relevant property/method documentation
    """
    # Step 1: Embed the incoming query using OpenAI
    print(f"Embedding PropMeth query: {query}", file=sys.stderr)
    query_embedding = get_query_embedding(query)
    
    # Step 2: Calculate cosine similarity between query and all pre-computed embeddings
    similarities = cosine_similarity(query_embedding, embeddings_PropMeth).flatten()
    
    # Step 3: Get the indices of the most similar examples
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Step 4: Retrieve the actual property/method documentation for the most similar examples
    results = []
    for idx in top_indices:
        similarity_score = float(similarities[idx])
        instruction = metadata_PropMeth['instructions'][idx]
        response = metadata_PropMeth['responses'][idx]
        
        results.append({
            'instruction': instruction,
            'response': response,
            'similarity_score': similarity_score
        })
        
        print(f"Found similar PropMeth example: '{instruction}' (similarity: {similarity_score:.3f})", file=sys.stderr)
    
    return results

@mcp.tool()
async def search_api_examples(query: str) -> str:
    """
    Reqest targeted API documentation queries using natural language.
    
    This tool takes a question about a specific API usage and returns the 3 most relevant
    examples from the pre-trained dataset, formatted as JSON.
    
    Examples of good queries:
    - "how to get DVH data"
    - "tell me about the Patient class" 
    - "What is the PlanSetup.Beams property?"
    - "What does EvaluationDose.DoseValueToVoxel do?"
    
    Args:
        query: Natural language question about API usage (keep it simple and focused) using entire API call
    
    Returns:
        Formatted string containing the most relevant API examples with their JSON documentation
    """
    # Debug: Check API key status
    api_key_env = os.getenv('OPENAI_API_KEY')
    print(f"API key from env: {'Found' if api_key_env else 'Not found'}", file=sys.stderr)
    print(f"OpenAI client API key: {'Set' if openai.api_key else 'Not set'}", file=sys.stderr)
    
    # Check for OpenAI API key
    if not openai.api_key and not os.getenv('OPENAI_API_KEY'):
        return "❌ Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Make sure embeddings are loaded
    if embeddings is None or metadata is None:
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
async def general_query_semantic(query: str) -> str:
    """
    Search for relevant API examples based on a natural language query if specific API call is not known.
    
    This tool takes a general question and returns the 3 most relevant API
    examples from the pre-trained dataset, formatted as JSON.
    
    Examples of good queries: 
    - "What are the control points for the proton beam in the current treatment?"
    - "How can I determine the number of fractions in a prescription?"
    - "What data can I retrieve about the patient support angle operating limit?"
    - "Is there a way to get the start date and time of a course?"
    
    Args:
        query: Natural language question about more general query (keep it simple and focused)
    
    Returns:
        Formatted string containing the most relevant API examples with their JSON documentation
    """
    # Debug: Check API key status
    api_key_env = os.getenv('OPENAI_API_KEY')
    print(f"API key from env: {'Found' if api_key_env else 'Not found'}", file=sys.stderr)
    print(f"OpenAI client API key: {'Set' if openai.api_key else 'Not set'}", file=sys.stderr)
    
    # Check for OpenAI API key
    if not openai.api_key and not os.getenv('OPENAI_API_KEY'):
        return "❌ Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Make sure embeddings are loaded
    if embeddings_semantic is None or metadata_semantic is None:
        return "❌ Error: Embeddings not loaded. Please restart the server."
    
    try:
        # Find the most similar API examples to the user's query
        similar_examples = find_similar_examples_semantic(query, top_k=3)
        
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
async def search_property_method_details(query: str) -> str:
    """
    Search for detailed information about specific properties or methods.
    
    This tool takes a query about a specific property or method and returns detailed
    information including type, parameters, usage examples, etc.
    
    Examples of good queries:
    - "Dose property information"
    - "BeamNumber.Number property details"
    - "GetDose method information"
    - "Patient.Name property"
    - "PlanSetup.GetOptimizationSetup method details"
    
    Args:
        query: Natural language question about a specific property or method
    
    Returns:
        Formatted string containing detailed property/method documentation
    """
    # Check for OpenAI API key
    if not openai.api_key and not os.getenv('OPENAI_API_KEY'):
        return "❌ Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Make sure embeddings are loaded
    if embeddings_PropMeth is None or metadata_PropMeth is None:
        return "❌ Error: Property/Method embeddings not loaded. Please restart the server."
    
    try:
        # Find the most similar property/method examples to the user's query
        similar_examples = find_similar_examples_PropMeth(query, top_k=3)
        
        # Format the results for return to the LLM
        result_parts = []
        result_parts.append(f"Found {len(similar_examples)} relevant property/method examples for: '{query}'\n")
        
        # Add each similar example with its detailed documentation
        for i, example in enumerate(similar_examples, 1):
            result_parts.append(f"--- Example {i} (Similarity: {example['similarity_score']:.3f}) ---")
            result_parts.append(f"Query: {example['instruction']}")
            result_parts.append(f"Details: {example['response']}")
            result_parts.append("")  # Empty line for readability
        
        # Join all parts into a single string response
        return "\n".join(result_parts)
        
    except Exception as e:
        print(f"Error in search_property_method_details: {str(e)}", file=sys.stderr)
        return f"Error processing query: {str(e)}"

@mcp.tool()
async def check_model_status() -> str:
    """
    Check the current status of the embedding system.
    
    Returns:
        Status message indicating whether the system is ready
    """
    status_parts = []
    
    if embeddings is not None and metadata is not None:
        status_parts.append("✅ Embeddings loaded successfully!")
        status_parts.append(f"📊 Dataset: {len(metadata['instructions'])} examples")
        if 'model_name' in metadata:
            status_parts.append(f"🤖 Model: {metadata['model_name']}")
        if 'embedding_provider' in metadata:
            status_parts.append(f"🔗 Provider: {metadata['embedding_provider']}")
    else:
        status_parts.append("❌ Embeddings not loaded")
    
    if embeddings_semantic is not None and metadata_semantic is not None:
        status_parts.append("✅ Semantic embeddings loaded successfully!")
        status_parts.append(f"📊 Semantic dataset: {len(metadata_semantic['instructions'])} examples")
    else:
        status_parts.append("❌ Semantic embeddings not loaded")
    
    if embeddings_PropMeth is not None and metadata_PropMeth is not None:
        status_parts.append("✅ Property/Method embeddings loaded successfully!")
        status_parts.append(f"📊 Property/Method dataset: {len(metadata_PropMeth['instructions'])} examples")
    else:
        status_parts.append("❌ Property/Method embeddings not loaded")
    
    # Debug API key
    api_key_env = os.getenv('OPENAI_API_KEY')
    if api_key_env:
        status_parts.append(f"🔑 OpenAI API key configured (ends with: ...{api_key_env[-4:]})")
    else:
        status_parts.append("❌ OpenAI API key not configured")
    
    status_parts.append(f"📁 Working directory: {Path.cwd()}")
    env_file = Path('.env')
    status_parts.append(f"📄 .env file exists: {env_file.exists()}")
    
    return "\n".join(status_parts)

# Template loading functions (unchanged from your original)
templates = None

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
    # Load environment variables FIRST
    print("Loading environment variables...", file=sys.stderr)
    load_env()
    
    # Load embeddings and metadata first (fast operation)
    print("Starting API RAG MCP server with OpenAI embeddings...", file=sys.stderr)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("The server will start but search functionality will not work.", file=sys.stderr)
    else:
        print(f"OpenAI API key loaded (ends with: ...{api_key[-4:]})", file=sys.stderr)
    
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
    
    # Start the MCP server
    print("MCP server ready with OpenAI embeddings!", file=sys.stderr)
    mcp.run(transport='stdio')