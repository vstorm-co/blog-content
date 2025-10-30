"""Generate retrieval evaluation questions for structured extraction benchmark records.

This script creates three questions per record to evaluate document retrieval systems.
Questions test whether the correct document is retrieved and at what position/rank.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

PROMPT_TEMPLATE = """Task: Generate retrieval evaluation questions for the record below. These questions will test whether a retrieval system correctly fetches THIS specific document when queried. Return ONLY a JSON array with EXACTLY 3 items (no prose).

Question Design Philosophy:
- Each question should be crafted so that THIS document is the most relevant result
- Questions can reference facts, specs, or characteristics unique to this computer
- Adversarial questions mentioning rivals/similar models are encouraged (e.g., "Amiga 500 vs 600") - the correct document should still be THIS one
- Version-specific questions are valuable for testing retrieval precision

Composition (exactly 3 total):
- Mix question types: identifying_query, spec_query, comparison_query, temporal_query, feature_query
- At least 1 question should be potentially confusing (mentions a rival, similar model, or requires disambiguation)
- Questions should vary in difficulty and specificity

Rules:
- Use ONLY facts from the record fields (name, aliases, release_year, manufacturer, keywords, tech_specs, description, hallmark, summary, evidence)
- Questions concise (≤ 20 words). Natural search query style.
- expected_doc_name must exactly match record.name (this is the document that should be retrieved)
- For confusing questions, set confuser_entity to the rival/alternative mentioned (if any)
- Provide evidence_quote (≤ 120 chars) showing why THIS doc is the right answer
- ASCII only: normal space (U+0020), straight quotes ("'), hyphen (-). No special Unicode characters.
- Output ONLY the JSON array; no Markdown or explanations.

Example questions with relation to source data (style guide; vary your actual questions):


## IBM 5150 Personal Computer:

- Which machine set a standard for a PC computer architecture?
- Which computer first offered both 5×ISA expansion slots?
- When was 5150 PC released?

Answers to those questions to be found in json below:
```json
{{{{
    "name": "IBM 5150 Personal Computer",
    "aliases": [
      "IBM PC",
      "5150",
      "IBM 5150 PC"
    ],
    "release_year": 1981,
    "manufacturer": "IBM",
    "keywords": [
      "IBM PC",
      "Intel 8088",
      "PC-DOS",
      "ISA expansion",
      "BIOS",
      "clones",
      "Wintel"
    ],
    "tech_specs": [
      {{
        "key": "CPU",
        "value": "Intel 8088 @ 4.77 MHz"
      }},
      {{
        "key": "RAM",
        "value": [
          "16 KB base",
          "640 KB max"
        ]
      }},
      {{
        "key": "Display",
        "value": "80×24 text"
      }},
      {{
        "key": "Storage",
        "value": "dual 160 KB 5.25-inch floppies (optional)"
      }},
      {{
        "key": "Expansion",
        "value": "5×ISA slots"
      }},
      {{
        "key": "OS",
        "value": "PC-DOS v1.0"
      }},
      {{
        "key": "Ports",
        "value": "cassette, keyboard"
      }},
      {{
        "key": "Price",
        "value": "US $1,565–$3,000"
      }}
    ],
    "description": "Released in September 1981, the IBM 5150 Personal Computer established the architectural template for the dominant PC platform. Built around Intel’s 8-bit 8088 running at 4.77 MHz, the base unit shipped with as little as 16 KB RAM and relied on an audio cassette for storage; one or two 160 KB 5.25-inch floppy drives were optional. A keyboard port, cassette interface and five internal ISA expansion slots provided the only connectivity—video, serial, parallel and memory cards occupied the slots in typical configurations. PC-DOS 1.0, Microsoft’s licensed MS-DOS, supplied the operating environment. Housed in a heavy sheet-metal chassis with the same buckling-spring keyboard used on IBM’s earlier Datamaster, the machine projected a professional image that appealed to businesses. Off-the-shelf hardware and a proprietary, copyrighted BIOS let IBM bring the system to market quickly while attempting to limit direct copying. Memory could be expanded from the motherboard’s 64 KB (later 256 KB with higher-density chips) to 640 KB via cards, and the open ISA bus encouraged third-party peripherals. Despite modest technical specifications compared with contemporary 16-bit machines, the 5150’s IBM badge, expandability and rapidly growing software library made it the de-facto business standard and the progenitor of the global PC clone ecosystem.",
    "hallmark": "First IBM PC whose open ISA bus and off-the-shelf parts created the standard that spawned the worldwide PC-clone industry.",
    "summary": "The IBM 5150 Personal Computer, launched in 1981, used an Intel 8088 CPU, optional floppy drives, five ISA slots and PC-DOS to establish the hardware standard that evolved into today’s Wintel platform.",
    "target_entity": "IBM 5150 Personal Computer",
  }}
```

## Macintosh

- First big success for individ. consumers that used mouse and had graphical interface?
- Famous computer set released in 1984?
- Which Mac had MHz 68000 CPU, 128 KB RAM, and a 9-inch monochrome display?
-
Answers to those questions to be found in json below:
```json
{{
  "name": "Macintosh",
  "aliases": [
    "M0001",
    "Original Macintosh",
    "128K Macintosh"
  ],
  "release_year": 1984,
  "manufacturer": "Apple",
  "keywords": [
    "GUI",
    "mouse",
    "68000",
    "monochrome",
    "all-in-one",
    "1984"
  ],
  "tech_specs": [
    {{
      "key": "CPU",
      "value": "Motorola 68000 @ 7.83 MHz"
    }},
    {{
      "key": "RAM",
      "value": [
        "128 KB",
        "512 KB (Fat Mac)"
      ]
    }},
    {{
      "key": "Display",
      "value": "9-inch monochrome 512×342"
    }},
    {{
      "key": "Storage",
      "value": "400 KB SSDD floppy"
    }},
    {{
      "key": "Ports",
      "value": [
        "2× DB9 serial",
        "printer",
        "external floppy"
      ]
    }},
    {{
      "key": "Intro price",
      "value": "$2 495"
    }}
  ],
  "description": "Released in January 1984, the Apple Macintosh (model M0001) was a compact all-in-one personal computer built around the 8 MHz Motorola 68000 processor and initially equipped with 128 KB RAM. Housed in a beige plastic case with an integral 9-inch 512 × 342-pixel monochrome CRT, it relied on a single 400 KB single-sided 3.5-inch floppy drive for storage; an external second drive was optional. The machine’s defining characteristic was its graphical user interface, accessed via a desktop of icons and windows manipulated with the bundled mouse—an approachable alternative to the text-only command lines of contemporary PCs. With no internal expansion slots and a sealed case, the base model was soon eclipsed by the 512 KB “Fat Mac,” but the Macintosh’s GUI, paired with bundled applications MacWrite and MacPaint, popularized desktop publishing and set the interaction paradigm for virtually all subsequent personal computers.",
  "hallmark": "First commercially successful computer controlled entirely by a mouse-driven graphical user interface.",
  "summary": "The original Apple Macintosh, launched in January 1984, combined an 8 MHz 68000 CPU, 128 KB RAM, and a 9-inch monochrome display in a single enclosure. Its mouse-operated GUI and built-in 400 KB floppy made graphical personal computing accessible and established the interface model still dominant today.",
  "target_entity": "Macintosh",
  "warnings": [
    "rival models (Lisa, Amiga 1000) mentioned; trimmed"
  ]
}}
```

### Commodore Amiga 600

- Which Amiga could be bought under at $500 price tag: 500 or 600?
- Which Amiga was the smallest ever released?
- Was Amiga 600 a Commodore or another manufacturer product?

Answers to those questions to be found in json below:
```json
{{
  "name": "Commodore Amiga 600",
  "aliases": [
    "Amiga 600",
    "A600"
  ],
  "release_year": 1992,
  "manufacturer": "Commodore",
  "keywords": [
    "Amiga",
    "Commodore",
    "PCMCIA",
    "ECS chipset",
    "Surface-mount technology"
  ],
  "tech_specs": [
    {{
      "key": "CPU",
      "value": "Motorola 68000 @ 7.16 MHz"
    }},
    {{
      "key": "RAM",
      "value": "1 MB (stock)"
    }},
    {{
      "key": "Display",
      "value": "32 colors @ 320×200, 4,096 colors HAM mode, max 640×400"
    }},
    {{
      "key": "Ports",
      "value": "parallel, serial, floppy, audio, RGB video, mouse/joystick"
    }},
    {{
      "key": "Expansion",
      "value": "trapdoor, PCMCIA slot"
    }},
    {{
      "key": "Storage",
      "value": "880 KB floppy; optional internal 20 MB or 40 MB hard drive"
    }},
    {{
      "key": "Operating System",
      "value": "AmigaDOS 2.05 \"Workbench\" GUI"
    }},
    {{
      "key": "Price",
      "value": "US $500"
    }}
  ],
  "description": "The Commodore Amiga 600, released in March 1992, was the smallest member of the Amiga line and the final 16‑bit model. It employed surface‑mount technology for the motherboard, lowering cost and increasing reliability, and introduced a PCMCIA slot for expansion. Powered by a Motorola 68000 CPU running at 7.16 MHz and equipped with 1 MB of RAM, it used the Enhanced Chip Set (ECS) to support 32 colours at 320×200 or up to 4,096 colours in HAM mode, with a maximum resolution of 640×400. Storage options included an 880 KB floppy and optional internal 20 MB or 40 MB hard drives. The system ran AmigaDOS 2.05 with the Workbench GUI and offered parallel, serial, audio, RGB video, and mouse/joystick ports, as well as trapdoor and PCMCIA expansion.",
  "hallmark": "First Amiga to use surface‑mount technology and include a PCMCIA expansion slot, distinguishing it in the 16‑bit line.",
  "summary": "The Amiga 600, launched by Commodore in March 1992, is a compact 16‑bit computer featuring surface‑mount construction, a PCMCIA slot, a 7.16 MHz 68000 CPU, 1 MB RAM, and the ECS graphics chipset.",
  "target_entity": "Commodore Amiga 600",
  "warnings": [
    "Rival models and broader history omitted for brevity"
  ]
}}
```


Allowed values:
- question_type ∈ {{"identifying_query", "spec_query", "comparison_query", "temporal_query", "feature_query", "disambiguation_query"}}
- difficulty ∈ {{"easy", "medium", "hard"}}
- tags ⊆ {{"adversarial", "version_specific", "ambiguous"}} (0-2 tags)

Record:
<<<BEGIN_RECORD
{record_json}
END_RECORD>>>"""


RETRY_HINT = "Fix schema issues but keep the questions relevant for retrieval testing."


# JSON Schema for future reference (not supported by GPT models)
# GPT models should rely on prompt instructions for JSON formatting
JSON_SCHEMA = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "expected_doc_name": {"type": "string", "minLength": 1, "maxLength": 200},
            "question": {"type": "string", "minLength": 5, "maxLength": 250},
            "question_type": {
                "type": "string",
                "enum": [
                    "identifying_query",
                    "spec_query",
                    "comparison_query",
                    "temporal_query",
                    "feature_query",
                    "disambiguation_query",
                ],
            },
            "evidence_quote": {
                "anyOf": [
                    {"type": "string", "maxLength": 120},
                    {"type": "null"},
                ]
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "adversarial",
                        "version_specific",
                        "ambiguous",
                    ],
                },
                "maxItems": 2,
            },
            "confuser_entity": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
        },
        "required": [
            "expected_doc_name",
            "question",
            "question_type",
            "evidence_quote",
            "difficulty",
            "tags",
            "confuser_entity",
        ],
    },
}


ASCII_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u2007": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00d7": "x",  # multiplication sign
}


ALLOWED_TAGS = {"adversarial", "version_specific", "ambiguous"}


class GenerationError(RuntimeError):
    """Custom error to indicate validation problems with model output."""


@dataclass
class GenerationResult:
    record_name: str
    questions: Sequence[dict[str, Any]]


def sanitize_ascii(text: str) -> str:
    """Normalize a JSON string to ASCII-safe characters."""
    if not text:
        return text

    for original, replacement in ASCII_REPLACEMENTS.items():
        text = text.replace(original, replacement)
    return text


def ensure_ascii(label: str, value: str) -> None:
    if any(ord(ch) > 127 for ch in value):
        raise GenerationError(f"{label} contains non-ASCII characters: {value!r}")


def count_words(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def prune_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy without evidence or warnings fields."""
    excluded_keys = {"evidence", "warnings"}
    return {key: value for key, value in record.items() if key not in excluded_keys}


def validate_questions(items: Sequence[dict[str, Any]], record_name: str) -> None:
    if len(items) != 3:
        raise GenerationError(f"Expected 3 items, received {len(items)}")

    for idx, item in enumerate(items):
        if item.get("expected_doc_name") != record_name:
            raise GenerationError(f"Item {idx + 1} expected_doc_name mismatch: {item.get('expected_doc_name')} != {record_name}")

        question = item.get("question", "")
        ensure_ascii("question", question)

        if count_words(question) > 20:
            raise GenerationError(f"Item {idx + 1} question exceeds 20 words: {question!r}")

        tags = item.get("tags", [])
        if not isinstance(tags, Iterable):
            raise GenerationError(f"Item {idx + 1} tags must be an array")

        invalid_tags = [tag for tag in tags if tag not in ALLOWED_TAGS]
        if invalid_tags:
            raise GenerationError(f"Item {idx + 1} has invalid tags: {invalid_tags}")


def build_messages(record: dict[str, Any], attempt: int) -> list[dict[str, str]]:
    filtered_record = prune_record(record)
    record_json = json.dumps(filtered_record, indent=2, ensure_ascii=False)
    user_prompt = PROMPT_TEMPLATE.format(record_json=record_json)
    messages = [
        {
            "role": "system",
            "content": "You are an expert at generating retrieval evaluation datasets for vintage computer records.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if attempt > 1:
        messages.append({"role": "user", "content": RETRY_HINT})
    return messages


def generate_for_record(
    client: OpenAI,
    model: str,
    record: dict[str, Any],
    request_delay: float = 0.0,
    seed: int | None = 42,
    temperature: float | None = None,
    use_schema: bool = False,
    max_attempts: int = 2,
) -> GenerationResult:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            messages = build_messages(record, attempt)
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if seed is not None:
                request_kwargs["seed"] = seed
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            # Note: use_schema parameter kept for future models that support json_schema
            # GPT models should use use_schema=False and rely on prompt instructions
            if use_schema:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "retrieval_eval", "schema": JSON_SCHEMA},
                }

            response = client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content
            if not content:
                raise GenerationError("API returned empty content")

            sanitized = sanitize_ascii(content)
            items = json.loads(sanitized)
            validate_questions(items, record["name"])
            return GenerationResult(record_name=record["name"], questions=items)
        except (json.JSONDecodeError, GenerationError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            if request_delay > 0:
                time.sleep(request_delay)
    assert last_error is not None
    raise GenerationError(f"Failed after {max_attempts} attempts: {last_error}")


def load_records(data_path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    records = json.loads(data_path.read_text())
    if limit <= 0:
        raise ValueError("Limit must be positive")
    if offset < 0:
        raise ValueError("Offset must be non-negative")
    if offset >= len(records):
        raise ValueError(f"Offset {offset} exceeds total records {len(records)}")

    end_idx = min(offset + limit, len(records))
    return records[offset:end_idx]


def append_generation(output_path: Path, result: GenerationResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for item in result.questions:
            payload = {
                "expected_doc_name": item.get("expected_doc_name"),
                "question": item.get("question"),
                "question_type": item.get("question_type"),
                "evidence_quote": item.get("evidence_quote"),
                "difficulty": item.get("difficulty"),
                "tags": item.get("tags"),
                "confuser_entity": item.get("confuser_entity"),
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrieval evaluation question dataset using GPT-4.1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of records to process from data.json (default: 3)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start processing from this record index (0-based, default: 0)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="OpenAI model name to use (default: gpt-4.1)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "retrieval_eval_dataset.jsonl"),
        help="Path to write the generated dataset",
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent / "data.json"),
        help="Path to the source data.json file",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between API requests (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic generation (default: 42)",
    )
    return parser.parse_args()


def load_api_key() -> str:
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please populate it in the environment or .env file.")
    return api_key


def main() -> int:
    args = parse_arguments()
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    data_path = Path(args.data)
    records = load_records(data_path, args.limit, args.offset)
    total_records = len(records)

    if args.offset > 0:
        print(f"Starting from record {args.offset + 1} (offset {args.offset})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't overwrite existing file - append mode will be used

    total_questions = 0

    for index, record in enumerate(records, start=1):
        record_name = record.get("name", f"record_{index}")
        print(f"[Record {index}] Generating questions for: {record_name}")
        result = generate_for_record(
            client,
            args.model,
            record,
            request_delay=args.request_delay,
            seed=args.seed,
        )
        print(f"  ✓ Generated {len(result.questions)} questions")
        append_generation(output_path, result)
        total_questions += len(result.questions)
        print(f"  ↳ Appended to {output_path}")
        if args.request_delay > 0 and index < total_records:
            time.sleep(args.request_delay)

    print(f"\nSaved dataset to: {output_path}")
    print(f"Total questions generated: {total_questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
