# Hybrid Retrieval Evaluation

Evaluate hybrid search performance by combining semantic (dense) and keyword (sparse) retrieval methods.

## What This Does

Tests retrieval quality across 507 computer-related queries using:
- **Dense search**: JinaAI embeddings for semantic matching
- **Sparse search**: BM25 for keyword matching
- **Hybrid**: Configurable blend of both (tune the weight ratio)

## Prerequisites

- Python 3.13
- JinaAI API key ([get 10M free tokens without signup](https://jina.ai/embeddings))
- `data.json` - 169 computer documents
- `retrieval_eval_dataset.jsonl` - 507 evaluation queries

## Quick Start

### 1. Install uv

This project uses `uv` for dependency management. To install `uv`, follow the instructions at:
https://docs.astral.sh/uv/getting-started/installation/#github-releases

### 2. Environment Setup

Create a virtual environment and install dependencies:

```bash
cd hybrid-retrieval-evaluation
uv venv --python 3.13
uv pip install -r requirements.txt
```

### 3. Set API Key

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your JinaAI API key
```

Your `.env` file should contain:
```
JINAAI_KEY=your_actual_jina_api_key_here
```

### 4. Create Indexes (First Time Only)

This creates embeddings and indexes for all documents and queries (~10-20 minutes):

```bash
uv run main.py --setup-only
```

### 5. Run Evaluation

**Batch Evaluation (default)** - Tests all weights from 0.0 to 1.0:

```bash
# Test weights: 0.0, 0.1, 0.2, ..., 1.0 (default behavior)
uv run main.py
# Outputs: evaluation_results.json

# Test specific weights
uv run main.py --weights 0.0 0.3 0.5 0.7 1.0

# Custom output file
uv run main.py --output my_results.json
```

**Single Evaluation** - Test one specific weight:

```bash
# Pure semantic search (best for conceptual queries)
uv run main.py --dense-weight 1.0

# Pure keyword search (best for exact terms)
uv run main.py --dense-weight 0.0

# Balanced hybrid (usually best overall)
uv run main.py --dense-weight 0.5

# Custom ratio (70% semantic, 30% keyword)
uv run main.py --dense-weight 0.7
```

## Understanding Results

### Example Output (Batch Mode)

```
================================================================================
BATCH EVALUATION SUMMARY
================================================================================
 dense_weight  sparse_weight  accuracy@1  accuracy@5  accuracy@10  mrr@10  confuser_at_2  confuser_mrr@10
          0.0            1.0      0.7142      0.8932       0.9350  0.7854         0.6570           0.4123
          0.5            0.5      0.7832      0.9230       0.9625  0.8534         0.7442           0.4856
          1.0            0.0      0.7834      0.9201       0.9601  0.8456         0.6977           0.4567
================================================================================

Best MRR@10: 0.8534 at dense_weight=0.50
Best Accuracy@1: 0.7834 at dense_weight=1.00

Results saved to evaluation_results.json
```

### Example Output (Single Mode)

When using `--dense-weight`:

```
============================================================
EVALUATION RESULTS
============================================================
Dense weight: 0.50 (Sparse: 0.50)
Total queries: 507

Accuracy@1:     0.7832   78.32%
Accuracy@5:     0.9230   92.30%
Accuracy@10:    0.9625   96.25%
MRR@10:         0.8534   85.34%

--- Comparison Query Metrics (n=172) ---
Confuser@2:           0.7442   74.42%
Confuser in top5:     0.9186   91.86%
Confuser in top10:    0.9593   95.93%
Confuser MRR@10:      0.4856   48.56%
============================================================
```

### Key Metrics

**Standard Metrics (all 507 queries):**
- **Accuracy@1**: Correct document ranked #1 (most important)
- **Accuracy@5**: Correct document in top 5
- **Accuracy@10**: Correct document in top 10
- **MRR@10**: Mean Reciprocal Rank - average position of correct answer

**Confuser Metrics (172 comparison queries only):**

For questions like "Does the Altos 586 or DEC Rainbow offer more serial ports?":
- **Confuser@2**: Comparison entity (DEC Rainbow) ranked #2 after correct answer
- **Confuser in top5/10**: Both entities appear in results
- **Confuser MRR@10**: Average rank of comparison entity

Good confuser metrics show the system understands both entities in comparison questions.

## Tuning Tips

**Dense weight = 1.0** (Pure semantic)
- Best for: Conceptual queries, paraphrasing
- Weak for: Exact model names, specifications

**Dense weight = 0.0** (Pure BM25)
- Best for: Exact terms, model numbers
- Weak for: Synonyms, conceptual similarity

**Dense weight = 0.5-0.7** (Hybrid, recommended)
- Best balance for mixed query types
- Usually achieves highest overall accuracy

## Expected Performance

| Configuration | Accuracy@1 | MRR@10 | Confuser@2 |
|--------------|------------|---------|------------|
| Pure semantic (1.0) | 75-85% | 0.80-0.90 | 60-75% |
| Pure BM25 (0.0) | 65-75% | 0.75-0.85 | 50-65% |
| Hybrid (0.5-0.7) | 78-88% | 0.85-0.92 | 70-80% |

## Performance Notes

- **First run**: 10-20 minutes (embedding creation)
- **Subsequent runs**: Seconds (cached embeddings reused)
- **Changing weights**: Instant (no re-embedding needed)

Embeddings are cached in `chroma_db/` directory and persist across runs.

## Files

### Core Files
- **`main.py`** - Main evaluation script. Runs hybrid retrieval evaluation with configurable weights. Supports both single-weight and batch mode (default: tests all weights 0.0–1.0). Creates embeddings once, then efficiently tests multiple weight combinations.
- **`data.json`** - Computer documents corpus (169 vintage computer descriptions)
- **`retrieval_eval_dataset.jsonl`** - Evaluation dataset (507 questions with expected answers)
- **`requirements.txt`** - Python dependencies (chromadb, rank-bm25, etc.)

### Utility Scripts
- **`test_query.py`** - Interactive query testing tool. Run custom queries on the fly to debug retrieval behavior and tune weights manually.
- **`create_eval_questions.py`** - Dataset generation script. Generates the 507-question evaluation dataset from source documents. Requires OpenAI API key. Only needed if regenerating the dataset (questions are already provided).

### Generated Files
- **`chroma_db/`** - Cached embeddings directory (created on first run, ~10-20 min). Contains pre-computed JinaAI embeddings for all documents and queries. Can be safely deleted to force re-indexing.
- **`evaluation_results.json`** - Evaluation results from batch mode runs (baseline performance metrics)

### Configuration
- **`.env`** - Your JinaAI API key (copy from `.env.example` and add your key)
- **`.env.example`** - Template for API key configuration

## Troubleshooting

**Import errors:**
```bash
uv pip install -r requirements.txt
```

**ChromaDB errors:**
```bash
rm -rf chroma_db/
uv run main.py --setup-only
```

**API key issues:**
```bash
# Make sure .env file exists with your key
cat .env
# Should show: JINAAI_KEY=your_actual_key
```

## Advanced Options

```bash
# Custom data files
uv run main.py --data /path/to/docs.json --queries /path/to/queries.jsonl

# Custom ChromaDB location
uv run main.py --chroma-db /path/to/db

# Different top-k for retrieval
uv run main.py --dense-weight 0.5 --top-k 20

# Batch with custom output location
uv run main.py --weights 0.3 0.5 0.7 --output results/experiment1.json

# Full example: custom everything
uv run main.py \
  --data /path/to/docs.json \
  --queries /path/to/queries.jsonl \
  --weights 0.0 0.5 1.0 \
  --output my_experiment.json \
  --top-k 20
```
