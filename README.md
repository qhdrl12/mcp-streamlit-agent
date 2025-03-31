# MCP Config & Chat Interface

A Streamlit-based web interface for managing MCP (Multi-Channel Platform) configurations and interacting with Large Language Models (LLMs).

## Features

- **MCP Configuration Management**: Edit, validate, and apply JSON configurations for MCP systems
- **Multi-Config Support**: Save, load, and manage multiple named configurations
- **Schema Validation**: JSON Schema validation for MCP configurations
- **LLM-powered Chat Interface**: Chat with multiple LLM providers (Gemini, ChatGPT, Claude)
- **MCP System Integration**: Execute MCP operations based on applied configurations

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mcp-streamddeck
   ```

2. Set up the environment (using Make):
   ```
   make setup
   ```
   
   Or manually:
   ```
   pip install -r requirements.txt
   cp .env.example .env
   ```

3. Edit the `.env` file to add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Usage

1. Start the Streamlit app (with Make):
   ```
   make run
   ```
   
   Or manually:
   ```
   streamlit run app.py
   ```
   
   Or using the run script:
   ```
   ./run.sh
   ```

2. Access the web interface at `http://localhost:8501`

3. In the interface:
   - Select/create MCP configurations in the left panel
   - Edit JSON configurations with schema validation
   - Save and apply configurations to the MCP adapter
   - Chat with your selected LLM in the right panel
   - The applied MCP configuration will be available to the LLM during chat

## Project Structure

- `app.py`: Main Streamlit application
- `models/`: LLM integration modules
  - `llm_handler.py`: Handler for different LLM providers
- `adapters/`: MCP system integration adapters
  - `mcp_adapter.py`: Basic MCP adapter implementation
  - `entry.py`: Entry point for MCP command execution
- `utils/`: Utility functions
  - `config_manager.py`: Configuration management utilities
  - `mcp_schema.py`: JSON schema for MCP configurations
- `configs/`: Directory for stored JSON configurations
  - `default.json`: Default configuration template

## Development

- `make setup`: Install dependencies and create .env file
- `make run`: Run the Streamlit application
- `make clean`: Clean up Python cache files
- `make init`: Initialize a new configuration based on the default template

## Dependencies

- Streamlit
- Langchain
- Google Generative AI (Gemini)
- OpenAI (ChatGPT)
- Anthropic (Claude)
- Python dotenv
- JSON Schema

## License

[MIT License](LICENSE) 