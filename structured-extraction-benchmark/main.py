import argparse
import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import Groq

_ = load_dotenv(find_dotenv())
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# Available models
MODELS = {
    "gpt-oss-20b": ("openai/gpt-oss-20b", "GPT-OSS 20B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "GPT-OSS 120B"),
    "kimi-k2": ("moonshotai/kimi-k2-instruct-0905", "Kimi K2 Instruct"),
    "llama-maverick": ("meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick"),
    "llama-scout": ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout"),
}

# System prompt with extraction rules
SYSTEM_PROMPT = """Task: Extract structured information about the single main computer model described in the text.
Return only a valid JSON object matching the schema below.

Rules:
- If unsure, return null rather than inventing facts.
- Exclude unrelated content (e.g., rival models, anecdotes, general history).
- Keep field names literal — do not rename or paraphrase them.
- Keep text concise and factual.
- Enforce field length limits where defined.
- """


def load_schema():
    schema_path = Path(__file__).parent / "schema.json"
    return json.loads(schema_path.read_text())


def load_data():
    data_path = Path(__file__).parent / "data.json"
    return json.loads(data_path.read_text())


def find_computer_by_title(data, title):
    for item in data:
        if item["title"] == title:
            return item
    return None


def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip().replace(" ", "_")


def extract_computer_info(content, model_key, schema):
    model_id, model_name = MODELS[model_key]
    print(f"Using model: {model_name} ({model_id})")
    print("Extracting computer information...")

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "computer_extraction", "schema": schema}},
    )

    content_response = response.choices[0].message.content
    if content_response is None:
        raise RuntimeError("API returned empty content")

    return json.loads(content_response)


def main():
    parser = argparse.ArgumentParser(description="Extract structured computer information from data.json")
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODELS.keys()),
        default="gpt-oss-120b",
        help="Model to use for extraction (default: gpt-oss-120b)",
    )
    parser.add_argument("--title", "-t", required=True, help="Title of the computer to extract from data.json")

    args = parser.parse_args()

    # Load data and find computer
    data = load_data()
    computer_data = find_computer_by_title(data, args.title)

    if computer_data is None:
        print(f"Error: Computer with title '{args.title}' not found in data.json")
        return 1

    print(f"Found: {computer_data['title']}")
    content = computer_data["content"]

    # Load schema
    schema = load_schema()

    # Extract information
    try:
        extraction = extract_computer_info(content, args.model, schema)

        # Create output directory
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Generate output filename
        computer_name = sanitize_filename(extraction.get("name", "unknown"))
        model_short = args.model.replace("-", "_")
        output_filename = f"{computer_name}_{model_short}.json"
        output_path = output_dir / output_filename

        # Save result
        output_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False))
        print(f"\nExtraction saved to: {output_path}")
        print("\nExtracted data:")
        print(json.dumps(extraction, indent=2, ensure_ascii=False))

        return 0

    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
