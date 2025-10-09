# Structured Data Extraction with JSON Schema

Demonstrates structured data extraction from documents using JSON Schema validation across multiple LLM models via the Groq API.

## Setup

### Installation

This project uses `uv` for dependency management. To install `uv`, follow the instructions at:
https://docs.astral.sh/uv/getting-started/installation/#github-releases

### Environment Setup

Create a virtual environment and install dependencies:

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```bash
GROQ_API_KEY=your_api_key_here
```

## Running

Execute the main script:

```bash
uv run main.py
```

The script will:
- Test multiple LLM models for structured extraction
- Apply a JSON schema to enforce output structure
- Save comparison results to `comparison_results.json`

## Requirements

- Python 3.13
- Dependencies listed in `requirements.txt`
- Groq API key
