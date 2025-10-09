import json
import os
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import Groq

_ = load_dotenv(find_dotenv())
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

content = Path("example_document.md").read_text()

# Schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "year_start": {"type": "integer"},
        "year_end": {"type": ["integer", "null"]},
        "category": {"type": "string"},
        "summary": {"type": "string", "minLength": 100, "maxLength": 200},
        "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 7},
        "identification_number": {"type": ["string", "null"]},
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "data": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": {"type": ["string", "number", "null"]}},
                    },
                },
                "required": ["data"],
            },
        },
    },
    "required": ["name", "summary", "keywords"],
}

# Models to compare
models = [
    ("openai/gpt-oss-20b", "GPT-OSS 20B"),
    ("openai/gpt-oss-120b", "GPT-OSS 120B"),
    ("moonshotai/kimi-k2-instruct-0905", "Kimi K2 Instruct"),
    ("meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout"),
]

results = []

for model_id, model_name in models:
    print(f"Testing {model_name}...")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Extract structured data from the document according to the schema."},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "extraction", "schema": schema}},
        )

        content_response = response.choices[0].message.content
        if content_response is None:
            raise RuntimeError("API returned empty content")

        extraction = json.loads(content_response)

        results.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "success": True,
                "extraction": extraction,
            }
        )

    except Exception as e:
        results.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "success": False,
                "error": str(e),
            }
        )
        print(f"Error: {e}")

    time.sleep(1)

# Save results
output_path = Path("comparison_results.json")
output_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {output_path}")
