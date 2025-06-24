# ESAPI MCP Server

A Model Context Protocol (MCP) server that provides AI language models with accurate ESAPI (Eclipse Scripting API) documentation and code templates, eliminating API hallucination issues.


## 🖥️ Claude Desktop Setup

This server is designed to work with [Claude Desktop](https://claude.ai/download) using the Model Context Protocol (MCP). Follow these steps to connect:

### Prerequisites
- [Claude Desktop](https://claude.ai/download) installed
- This ESAPI MCP Server running locally

### Configuration Steps

1. **Download and Install Claude Desktop**
   - Visit [claude.ai/download](https://claude.ai/download)
   - Install Claude Desktop for your operating system

2. **Locate Claude Desktop Config File**
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

3. **Add ESAPI MCP Server Configuration**
   
   Edit the config file and add the following (adjust paths to your setup):

   **Windows Example:**
   ```json
   {
     "mcpServers": {
       "esapi_mcp": {
         "command": "C:\\path\\to\\your\\project\\esapi-mcp-server\\.venv\\Scripts\\python.exe",
         "args": ["C:\\path\\to\\your\\project\\esapi-mcp-server\\src\\mcp_server.py"],
         "cwd": "C:\\path\\to\\your\\project\\esapi-mcp-server",
         "env": {},
         "timeout": 120000
       }
     }
   }
   ```

   **macOS/Linux Example:**
   ```json
   {
     "mcpServers": {
       "esapi_mcp": {
         "command": "/path/to/your/project/esapi-mcp-server/.venv/bin/python",
         "args": ["/path/to/your/project/esapi-mcp-server/src/mcp_server.py"],
         "cwd": "/path/to/your/project/esapi-mcp-server",
         "env": {},
         "timeout": 120000
       }
     }
   }
   ```

4. **Restart Claude Desktop**
   - Close Claude Desktop completely
   - Reopen Claude Desktop
   - The ESAPI MCP tools should now be available


### Troubleshooting

- **Tools not appearing**: Check that file paths in config are correct and use forward slashes or escaped backslashes
- **Server not starting**: Ensure the virtual environment is properly set up with `uv sync`
- **Permission errors**: Make sure Claude Desktop has permission to execute Python in your project directory
- **Timeout errors**: Increase the timeout value if the embedding model takes longer to load

### Learn More

For detailed MCP setup instructions, see: [Model Context Protocol Documentation](https://modelcontextprotocol.io/docs/getting-started)


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

This Repo is new, as a result it will have areas for vast improvement.

I plan to continually improve and enhance this repo when I have time to dedicate to it.

That being said, contributions are welcome, and encouraged! Areas for improvement:

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
