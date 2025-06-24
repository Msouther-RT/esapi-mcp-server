# ESAPI MCP Server

A Model Context Protocol (MCP) server that provides AI language models with accurate ESAPI (Eclipse Scripting API) documentation and code templates, eliminating API hallucination issues.

## 🚀 Features

- 🔍 **Semantic API Search**: Find relevant ESAPI examples using natural language queries
- 📝 **Code Templates**: Get correct boilerplate for different ESAPI script types  
- ⚡ **Fast Lookup**: Pre-computed embeddings for instant search results (8,154+ examples)
- 🎯 **Accurate Results**: Based on real ESAPI documentation, not AI guesses
- 🧠 **Smart Matching**: Uses Qwen3 embedding model for semantic understanding

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/esapi-mcp-server
   cd esapi-mcp-server

2. Setup environment with UV
bashuv sync
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

3. Run the server
bashpython src\mcp_server.py

4. Connect to your AI client

Add as MCP server in Claude Desktop or compatible AI client
Server runs on stdio transport



## 🛠️ Available Tools

### `search_api_examples(query: str)`

Search through 8,154+ ESAPI examples using natural language:

```python
# Example queries:
search_api_examples("how to get DVH data")
search_api_examples("calculate dose statistics")
search_api_examples("access beam parameters")
search_api_examples("patient plan information")
```

### `get_esapi_template(script_type: str)`

Get code templates for different ESAPI script types:

```python
get_esapi_template("single_file")      # Simple .cs file
get_esapi_template("binary_plugin")    # Plugin with .cs + .csproj
get_esapi_template("executable_script") # Standalone executable
get_esapi_template("list")             # Show all available types
```

### `check_model_status()`

Check if the embedding model has finished loading.

## 📁 Project Structure

```
esapi-mcp-server/
├── src/
│   ├── mcp_server.py           # Main MCP server
│   └── embed_instructions.py   # Embedding generation script
├── templates/                  # ESAPI code templates
│   ├── single_file.cs
│   ├── binary_plugin.cs
│   ├── binary_plugin.csproj
│   ├── executable_script.cs
│   └── executable_script.csproj
├── embeddings/                 # Pre-computed embeddings
│   ├── metadata.pkl           # Questions & answers
│   └── instruction_embeddings.pkl  # Vector embeddings
├── data/                      # Training data
└── pyproject.toml             # Project configuration
```


## 🔧 Development

This project uses [UV](https://docs.astral.sh/uv/) for fast dependency management.

```bash
# Install development dependencies
uv sync --dev

# Run the server
uv run python src/mcp_server.py

# Generate new embeddings (if needed)
uv run python src/embed_instructions.py
```

## 📊 Dataset

- **8,154+ ESAPI examples** from comprehensive documentation (16.1 Libraries)
- **Semantic embeddings** using Qwen/Qwen3-Embedding-0.6B model
- **Real API calls** with accurate parameters and return types
- **Multiple script patterns** covering common ESAPI use cases

## 🎯 Use Cases

Perfect for medical physicists and developers working with:

- Treatment planning automation
- Dose analysis scripts
- Quality assurance tools
- Research applications
- Educational projects

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional ESAPI examples
- New script templates
- Documentation improvements
- Performance optimizations

## 📄 License

MIT License - feel free to use in your medical physics workflows.

## 🙏 Acknowledgments

Built for the medical physics community to improve ESAPI development experience and reduce API documentation friction.

---

**Note**: This tool provides accurate ESAPI documentation but always validate scripts in a safe testing environment before clinical use.
