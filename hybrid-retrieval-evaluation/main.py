"""Hybrid retrieval evaluation suite with dense (JinaAI) and sparse (BM25) search."""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv()


# Configuration
class Config:
    """Configuration for hybrid retrieval evaluation."""

    def __init__(self):
        self.jinaai_api_key = os.environ.get("JINAAI_API_KEY")
        self.jina_embedding_url = "https://api.jina.ai/v1/embeddings"
        self.jina_model = "jina-embeddings-v3"
        self.embedding_dim = 1024
        self.batch_size = 32
        self.default_top_k = 10

    def validate(self):
        """Validate configuration."""
        if not self.jinaai_api_key:
            logger.error("JINAAI_API_KEY not found in environment variables")
            raise ValueError(
                "JINAAI_API_KEY is required. Please set it in your .env file.\nGet a free key at: https://jina.ai/embeddings"
            )


config = Config()


@dataclass
class EvaluationMetrics:
    """Evaluation metrics from retrieval evaluation."""

    accuracy_at_1: float
    accuracy_at_3: float
    accuracy_at_5: float
    accuracy_at_10: float
    mrr_at_1: float
    mrr_at_3: float
    mrr_at_5: float
    mrr_at_10: float
    dense_weight: float
    sparse_weight: float
    total_queries: int
    comparison_queries: int
    regular_failure_count: int
    regular_failures_mrr5: list[dict[str, Any]] = field(default_factory=list)
    confuser_at_2: float | None = None
    confuser_in_top3: float | None = None
    confuser_in_top5: float | None = None
    confuser_in_top10: float | None = None
    confuser_mrr_at_3: float | None = None
    confuser_mrr_at_5: float | None = None
    confuser_mrr_at_10: float | None = None


def doc_to_markdown(doc: dict[str, Any], *, include_description: bool = True) -> str:
    """Convert document JSON to markdown format.

    Args:
        doc: Document dictionary
        include_description: Whether to include the description field (default: True).
                           Set to False for BM25 indexing to avoid long text.
    """
    lines = []

    lines.append(f"# {doc['name']}\n")

    if doc.get("aliases"):
        lines.append(f"**Aliases:** {', '.join(doc['aliases'])}\n")

    if doc.get("manufacturer"):
        lines.append(f"**Manufacturer:** {doc['manufacturer']}\n")

    if doc.get("release_year"):
        lines.append(f"**Release Year:** {doc['release_year']}\n")

    if doc.get("hallmark"):
        lines.append(f"**Hallmark:** {doc['hallmark']}\n")

    if doc.get("keywords"):
        lines.append(f"**Keywords:** {', '.join(doc['keywords'])}\n")

    if doc.get("tech_specs"):
        lines.append("\n## Technical Specifications\n")
        for spec in doc["tech_specs"]:
            key = spec["key"]
            value = spec["value"]
            if isinstance(value, list):
                value = ", ".join(value)
            lines.append(f"- **{key}:** {value}\n")

    if doc.get("summary"):
        lines.append(f"\n## Summary\n{doc['summary']}\n")

    if include_description and doc.get("description"):
        lines.append(f"\n## Description\n{doc['description']}\n")

    return "".join(lines)


def embed_with_jina(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Create embeddings using JinaAI API.

    Args:
        texts: List of text strings to embed
        task: Embedding task type ("retrieval.passage" or "retrieval.query")

    Returns:
        List of embedding vectors

    Raises:
        requests.HTTPError: If API request fails
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.jinaai_api_key}"}
    data = {"model": config.jina_model, "task": task, "input": texts}

    try:
        response = requests.post(config.jina_embedding_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("JinaAI API request timed out")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"JinaAI API request failed: {e}")
        raise

    result = response.json()
    embeddings = [item["embedding"] for item in result["data"]]
    return embeddings


class HybridRetriever:
    """Hybrid retrieval system combining dense (ChromaDB) and sparse (BM25) search."""

    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize the hybrid retriever."""
        self.chroma_db_path = chroma_db_path
        self.chroma_client = None
        self.doc_collection = None
        self.query_collection = None
        self.bm25 = None
        self.doc_names = []
        self.doc_texts = []
        self.tokenized_docs = []

    def setup_collections(self):
        """Set up ChromaDB collections."""
        logger.info(f"Setting up ChromaDB at {self.chroma_db_path}")

        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise

        # Create or get collections
        try:
            self.doc_collection = self.chroma_client.get_collection(name="documents")
            logger.info(f"Loaded existing document collection with {self.doc_collection.count()} documents")
        except ValueError:
            self.doc_collection = self.chroma_client.create_collection(name="documents", metadata={"hnsw:space": "cosine"})
            logger.info("Created new document collection")

        try:
            self.query_collection = self.chroma_client.get_collection(name="queries")
            logger.info(f"Loaded existing query collection with {self.query_collection.count()} queries")
        except ValueError:
            self.query_collection = self.chroma_client.create_collection(name="queries", metadata={"hnsw:space": "cosine"})
            logger.info("Created new query collection")

    def process_and_index_documents(self, data_path: str):
        """Process documents from JSON, create embeddings, and index in ChromaDB and BM25."""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Processing documents from {data_path}...")

        with open(data_path) as f:
            docs = json.load(f)

        logger.info(f"Found {len(docs)} documents")

        # Check if already indexed
        if self.doc_collection.count() >= len(docs):
            logger.info("Documents already indexed in ChromaDB, creating BM25 index...")
            # Create BM25 index from original docs (without description)
            bm25_docs = [doc_to_markdown(doc, include_description=False) for doc in docs]
            self.doc_names = [doc["name"] for doc in docs]
            self.doc_texts = bm25_docs
            self.tokenized_docs = [text.lower().split() for text in bm25_docs]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info("BM25 index created (without description field)")
            return

        # Convert to markdown
        markdown_docs = []
        doc_names = []
        metadatas = []

        for doc in docs:
            # Dense embeddings: use full markdown (includes description)
            md_content = doc_to_markdown(doc, include_description=True)
            markdown_docs.append(md_content)
            doc_names.append(doc["name"])
            metadatas.append(
                {
                    "name": doc["name"],
                    "manufacturer": doc.get("manufacturer") or "",
                    "release_year": str(doc.get("release_year") or ""),
                    "type": "document",
                }
            )

        # Create embeddings in batches (using full markdown with description)
        logger.info("Creating embeddings for documents...")
        all_embeddings = []

        for i in tqdm(range(0, len(markdown_docs), config.batch_size), desc="Embedding documents", unit="batch"):
            batch = markdown_docs[i : i + config.batch_size]
            embeddings = embed_with_jina(batch, task="retrieval.passage")
            all_embeddings.extend(embeddings)

        # Index in ChromaDB
        logger.info("Indexing in ChromaDB...")
        ids = [f"doc_{i}" for i in range(len(docs))]
        self.doc_collection.add(ids=ids, embeddings=all_embeddings, documents=markdown_docs, metadatas=metadatas)
        logger.info(f"Indexed {len(docs)} documents in ChromaDB")

        # Create BM25 index (using BM25-optimized markdown without description)
        logger.info("Creating BM25 index (without description field)...")
        bm25_docs = [doc_to_markdown(doc, include_description=False) for doc in docs]
        self.doc_names = doc_names
        self.doc_texts = bm25_docs
        self.tokenized_docs = [text.lower().split() for text in bm25_docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25 index created")

    def process_and_index_queries(self, queries_path: str):
        """Process queries, create embeddings, and store in ChromaDB."""
        if not Path(queries_path).exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")

        logger.info(f"Processing queries from {queries_path}...")

        queries_data = []
        with open(queries_path) as f:
            for line in f:
                queries_data.append(json.loads(line))

        logger.info(f"Found {len(queries_data)} queries")

        # Check if already indexed
        if self.query_collection.count() >= len(queries_data):
            logger.info("Queries already indexed, skipping...")
            return

        # Extract query texts and metadata
        query_texts = []
        metadatas = []

        for qdata in queries_data:
            query_texts.append(qdata["question"])
            metadatas.append(
                {
                    "expected_doc_name": qdata["expected_doc_name"],
                    "question_type": qdata.get("question_type", ""),
                    "difficulty": qdata.get("difficulty", ""),
                    "tags": json.dumps(qdata.get("tags", [])),
                    "confuser_entity": qdata.get("confuser_entity", "") or "",
                    "type": "query",
                }
            )

        # Create embeddings in batches
        logger.info("Creating embeddings for queries...")
        all_embeddings = []

        for i in tqdm(range(0, len(query_texts), config.batch_size), desc="Embedding queries", unit="batch"):
            batch = query_texts[i : i + config.batch_size]
            embeddings = embed_with_jina(batch, task="retrieval.query")
            all_embeddings.extend(embeddings)

        # Index in ChromaDB
        logger.info("Indexing queries in ChromaDB...")
        ids = [f"query_{i}" for i in range(len(queries_data))]
        self.query_collection.add(ids=ids, embeddings=all_embeddings, documents=query_texts, metadatas=metadatas)
        logger.info(f"Indexed {len(queries_data)} queries in ChromaDB")

    def retrieve_dense(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve documents using dense vector search."""
        results = self.doc_collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved = []
        for meta, distance in zip(results["metadatas"][0], results["distances"][0], strict=False):
            # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
            similarity = 1 - distance
            retrieved.append((meta["name"], similarity))

        return retrieved

    def retrieve_sparse(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve documents using BM25 sparse search."""
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        retrieved = []
        for idx in top_indices:
            doc_name = self.doc_names[idx]
            score = scores[idx]
            retrieved.append((doc_name, score))

        return retrieved

    def hybrid_retrieve(
        self, query_text: str, query_embedding: list[float], dense_weight: float = 0.5, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Hybrid retrieval combining dense and sparse scores."""
        sparse_weight = 1.0 - dense_weight

        # Get results from both methods
        dense_results = self.retrieve_dense(query_embedding, top_k=top_k * 2)
        sparse_results = self.retrieve_sparse(query_text, top_k=top_k * 2)

        # Normalize scores to [0, 1] range
        def normalize_scores(results):
            if not results:
                return {}
            scores = [s for _, s in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {name: 1.0 for name, _ in results}
            return {name: (score - min_score) / (max_score - min_score) for name, score in results}

        dense_normalized = normalize_scores(dense_results)
        sparse_normalized = normalize_scores(sparse_results)

        # Combine scores
        all_docs = set(dense_normalized.keys()) | set(sparse_normalized.keys())
        combined_scores = {}

        for doc_name in all_docs:
            dense_score = dense_normalized.get(doc_name, 0.0)
            sparse_score = sparse_normalized.get(doc_name, 0.0)
            combined_scores[doc_name] = dense_weight * dense_score + sparse_weight * sparse_score

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]

    def evaluate(self, dense_weight: float = 0.5, top_k: int = 10) -> EvaluationMetrics:
        """Evaluate retrieval performance."""
        logger.info(f"Evaluating with dense_weight={dense_weight:.2f}, top_k={top_k}")

        # Get all queries
        all_queries = self.query_collection.get(include=["embeddings", "documents", "metadatas"])

        total = len(all_queries["ids"])
        correct_at_1 = 0
        correct_at_3 = 0
        correct_at_5 = 0
        correct_at_10 = 0
        mrr_sum_1 = 0.0
        mrr_sum_3 = 0.0
        mrr_sum_5 = 0.0
        mrr_sum_10 = 0.0

        # Confuser entity tracking
        comparison_queries = 0
        confuser_at_2 = 0
        confuser_in_top3 = 0
        confuser_in_top5 = 0
        confuser_in_top10 = 0
        confuser_mrr_sum_3 = 0.0
        confuser_mrr_sum_5 = 0.0
        confuser_mrr_sum_10 = 0.0
        regular_failures_mrr5: list[dict[str, Any]] = []

        for query_id, query_embedding, query_text, metadata in tqdm(
            zip(all_queries["ids"], all_queries["embeddings"], all_queries["documents"], all_queries["metadatas"], strict=False),
            total=total,
            desc="Evaluating queries",
            unit="query",
        ):
            expected_doc = metadata["expected_doc_name"]
            confuser_entity = metadata.get("confuser_entity", "")

            # Retrieve
            results = self.hybrid_retrieve(query_text, query_embedding, dense_weight, top_k=10)
            retrieved_docs = [name for name, _ in results]

            # Calculate accuracy metrics
            if len(retrieved_docs) > 0 and retrieved_docs[0] == expected_doc:
                correct_at_1 += 1

            if expected_doc in retrieved_docs[:3]:
                correct_at_3 += 1

            if expected_doc in retrieved_docs[:5]:
                correct_at_5 += 1

            if expected_doc in retrieved_docs[:10]:
                correct_at_10 += 1

            # Calculate MRR metrics at different cutoffs
            expected_rank: int | None = None
            try:
                expected_rank = retrieved_docs.index(expected_doc) + 1

                # MRR@1: only counts if at rank 1
                if expected_rank == 1:
                    mrr_sum_1 += 1.0

                # MRR@3: counts if in top 3
                if expected_rank <= 3:
                    mrr_sum_3 += 1.0 / expected_rank

                # MRR@5: counts if in top 5
                if expected_rank <= 5:
                    mrr_sum_5 += 1.0 / expected_rank

                # MRR@10: counts if in top 10
                if expected_rank <= 10:
                    mrr_sum_10 += 1.0 / expected_rank

            except ValueError:
                expected_rank = None  # Not found in top-10

            # Track confuser entity for comparison queries
            if confuser_entity and confuser_entity.strip():
                comparison_queries += 1

                # Check if confuser is at rank 2
                if len(retrieved_docs) > 1 and retrieved_docs[1] == confuser_entity:
                    confuser_at_2 += 1

                # Check if confuser is in top 3
                if confuser_entity in retrieved_docs[:3]:
                    confuser_in_top3 += 1

                # Check if confuser is in top 5
                if confuser_entity in retrieved_docs[:5]:
                    confuser_in_top5 += 1

                # Check if confuser is in top 10
                if confuser_entity in retrieved_docs[:10]:
                    confuser_in_top10 += 1

                # MRR for confuser at different cutoffs
                try:
                    confuser_rank = retrieved_docs.index(confuser_entity) + 1

                    # MRR@3: counts if in top 3
                    if confuser_rank <= 3:
                        confuser_mrr_sum_3 += 1.0 / confuser_rank

                    # MRR@5: counts if in top 5
                    if confuser_rank <= 5:
                        confuser_mrr_sum_5 += 1.0 / confuser_rank

                    # MRR@10: counts if in top 10
                    if confuser_rank <= 10:
                        confuser_mrr_sum_10 += 1.0 / confuser_rank

                except ValueError:
                    pass  # Not found in top-10
            else:
                mrr5_value = 1.0 / expected_rank if expected_rank is not None and expected_rank <= 5 else 0.0
                if mrr5_value == 0.0:
                    regular_failures_mrr5.append(
                        {
                            "query_id": query_id,
                            "question": query_text,
                            "expected_doc": expected_doc,
                            "expected_rank": expected_rank,
                            "top_results": [{"doc": doc_name, "score": score} for doc_name, score in results[:5]],
                        }
                    )

        regular_failures_mrr5.sort(key=lambda item: (item["expected_rank"] is not None, item["expected_rank"] or 0))

        metrics = EvaluationMetrics(
            accuracy_at_1=correct_at_1 / total,
            accuracy_at_3=correct_at_3 / total,
            accuracy_at_5=correct_at_5 / total,
            accuracy_at_10=correct_at_10 / total,
            mrr_at_1=mrr_sum_1 / total,
            mrr_at_3=mrr_sum_3 / total,
            mrr_at_5=mrr_sum_5 / total,
            mrr_at_10=mrr_sum_10 / total,
            dense_weight=dense_weight,
            sparse_weight=1.0 - dense_weight,
            total_queries=total,
            comparison_queries=comparison_queries,
            regular_failure_count=len(regular_failures_mrr5),
            regular_failures_mrr5=regular_failures_mrr5,
        )

        # Add confuser metrics if there are comparison queries
        if comparison_queries > 0:
            metrics.confuser_at_2 = confuser_at_2 / comparison_queries
            metrics.confuser_in_top3 = confuser_in_top3 / comparison_queries
            metrics.confuser_in_top5 = confuser_in_top5 / comparison_queries
            metrics.confuser_in_top10 = confuser_in_top10 / comparison_queries
            metrics.confuser_mrr_at_3 = confuser_mrr_sum_3 / comparison_queries
            metrics.confuser_mrr_at_5 = confuser_mrr_sum_5 / comparison_queries
            metrics.confuser_mrr_at_10 = confuser_mrr_sum_10 / comparison_queries

        return metrics


def print_single_result(metrics: EvaluationMetrics, console: Console):
    """Print results for a single evaluation using rich."""
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]EVALUATION RESULTS[/bold cyan]")
    console.print("=" * 70)
    console.print(f"Dense weight: {metrics.dense_weight:.2f} (Sparse: {metrics.sparse_weight:.2f})")
    console.print(f"Total queries: {metrics.total_queries}")

    # Create metrics table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Accuracy@1", f"{metrics.accuracy_at_1:.4f}", f"{metrics.accuracy_at_1 * 100:.2f}%")
    table.add_row("Accuracy@3", f"{metrics.accuracy_at_3:.4f}", f"{metrics.accuracy_at_3 * 100:.2f}%")
    table.add_row("Accuracy@5", f"{metrics.accuracy_at_5:.4f}", f"{metrics.accuracy_at_5 * 100:.2f}%")
    table.add_row("Accuracy@10", f"{metrics.accuracy_at_10:.4f}", f"{metrics.accuracy_at_10 * 100:.2f}%")
    table.add_row("", "", "")  # Empty row
    table.add_row("MRR@1", f"{metrics.mrr_at_1:.4f}", f"{metrics.mrr_at_1 * 100:.2f}%")
    table.add_row("MRR@3", f"{metrics.mrr_at_3:.4f}", f"{metrics.mrr_at_3 * 100:.2f}%")
    table.add_row("MRR@5", f"{metrics.mrr_at_5:.4f}", f"{metrics.mrr_at_5 * 100:.2f}%")
    table.add_row("MRR@10", f"{metrics.mrr_at_10:.4f}", f"{metrics.mrr_at_10 * 100:.2f}%")

    console.print(table)

    if metrics.regular_failure_count > 0:
        console.print(f"\n[red]Regular failures (MRR@5=0): {metrics.regular_failure_count}[/red]")
        if metrics.regular_failures_mrr5:
            console.print("[yellow]Worst regular queries (MRR@5=0):[/yellow]")
            for entry in metrics.regular_failures_mrr5[:5]:
                rank_str = entry["expected_rank"] if entry["expected_rank"] is not None else "not in top10"
                console.print(f"  • {entry['question']}")
                console.print(f"    Expected: [cyan]{entry['expected_doc']}[/cyan] | Rank: [red]{rank_str}[/red]")
            if metrics.regular_failure_count > 5:
                console.print(f"  ... {metrics.regular_failure_count - 5} more")

    # Print confuser metrics if available
    if metrics.comparison_queries > 0:
        console.print(f"\n[bold]Comparison Query Metrics (n={metrics.comparison_queries})[/bold]")

        confuser_table = Table(show_header=True, header_style="bold magenta")
        confuser_table.add_column("Metric", style="cyan", width=20)
        confuser_table.add_column("Value", justify="right", style="green")
        confuser_table.add_column("Percentage", justify="right", style="yellow")

        confuser_table.add_row("Confuser@2", f"{metrics.confuser_at_2:.4f}", f"{metrics.confuser_at_2 * 100:.2f}%")
        confuser_table.add_row("Confuser in top3", f"{metrics.confuser_in_top3:.4f}", f"{metrics.confuser_in_top3 * 100:.2f}%")
        confuser_table.add_row("Confuser in top5", f"{metrics.confuser_in_top5:.4f}", f"{metrics.confuser_in_top5 * 100:.2f}%")
        confuser_table.add_row("Confuser in top10", f"{metrics.confuser_in_top10:.4f}", f"{metrics.confuser_in_top10 * 100:.2f}%")
        confuser_table.add_row("", "", "")  # Empty row
        confuser_table.add_row("Confuser MRR@3", f"{metrics.confuser_mrr_at_3:.4f}", f"{metrics.confuser_mrr_at_3 * 100:.2f}%")
        confuser_table.add_row("Confuser MRR@5", f"{metrics.confuser_mrr_at_5:.4f}", f"{metrics.confuser_mrr_at_5 * 100:.2f}%")
        confuser_table.add_row("Confuser MRR@10", f"{metrics.confuser_mrr_at_10:.4f}", f"{metrics.confuser_mrr_at_10 * 100:.2f}%")

        console.print(confuser_table)

    console.print("=" * 70)


def print_batch_summary(results: list[EvaluationMetrics], console: Console):
    """Print batch evaluation summary using rich."""
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]BATCH EVALUATION SUMMARY[/bold cyan]")
    console.print("=" * 80)

    # Convert to dicts for DataFrame
    results_dicts = [asdict(m) for m in results]
    df = pd.DataFrame(results_dicts)

    # Create rich table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dense", justify="right", style="cyan")
    table.add_column("Sparse", justify="right", style="cyan")
    table.add_column("Acc@1", justify="right", style="green")
    table.add_column("Acc@5", justify="right", style="green")
    table.add_column("Acc@10", justify="right", style="green")
    table.add_column("MRR@10", justify="right", style="yellow")

    if "confuser_at_2" in df.columns:
        table.add_column("Conf@2", justify="right", style="blue")
        table.add_column("Conf MRR@10", justify="right", style="blue")

    for _, row in df.iterrows():
        row_data = [
            f"{row['dense_weight']:.2f}",
            f"{row['sparse_weight']:.2f}",
            f"{row['accuracy_at_1']:.4f}",
            f"{row['accuracy_at_5']:.4f}",
            f"{row['accuracy_at_10']:.4f}",
            f"{row['mrr_at_10']:.4f}",
        ]
        if "confuser_at_2" in df.columns:
            row_data.append(f"{row['confuser_at_2']:.4f}" if pd.notna(row["confuser_at_2"]) else "N/A")
            row_data.append(f"{row['confuser_mrr_at_10']:.4f}" if pd.notna(row["confuser_mrr_at_10"]) else "N/A")

        table.add_row(*row_data)

    console.print(table)
    console.print("=" * 80)

    # Find best configurations
    best_mrr = df.loc[df["mrr_at_10"].idxmax()]
    best_acc1 = df.loc[df["accuracy_at_1"].idxmax()]

    console.print(f"\n[bold green]Best MRR@10:[/bold green] {best_mrr['mrr_at_10']:.4f} at dense_weight={best_mrr['dense_weight']:.2f}")
    console.print(
        f"[bold green]Best Accuracy@1:[/bold green] {best_acc1['accuracy_at_1']:.4f} at dense_weight={best_acc1['dense_weight']:.2f}"
    )


def main():
    """Main entry point."""
    import argparse

    # Get script directory for resolving relative paths
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Hybrid retrieval evaluation")
    parser.add_argument("--data", default=None, help="Path to documents JSON")
    parser.add_argument("--queries", default=None, help="Path to queries JSONL")
    parser.add_argument("--chroma-db", default=None, help="Path to ChromaDB storage")
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help="Weight for dense retrieval - single evaluation (0=pure BM25, 1=pure semantic)",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None, help="Multiple weights for batch evaluation (e.g., --weights 0.0 0.5 1.0)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--setup-only", action="store_true", help="Only setup indexes without evaluation")
    parser.add_argument("--output", default=None, help="Output file for batch results (JSON)")

    args = parser.parse_args()

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Set default paths relative to script directory
    if args.data is None:
        args.data = str(script_dir / "data.json")
    if args.queries is None:
        args.queries = str(script_dir / "retrieval_eval_dataset.jsonl")
    if args.chroma_db is None:
        args.chroma_db = str(script_dir / "chroma_db")

    # Initialize retriever
    try:
        retriever = HybridRetriever(chroma_db_path=args.chroma_db)
        retriever.setup_collections()

        # Process and index documents
        retriever.process_and_index_documents(args.data)

        # Process and index queries
        retriever.process_and_index_queries(args.queries)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)

    if args.setup_only:
        logger.info("Setup complete!")
        return None

    # Determine which mode: batch or single
    if args.weights is not None:
        # Batch evaluation mode
        weights = args.weights
    elif args.dense_weight is not None:
        # Single evaluation mode
        weights = [args.dense_weight]
    else:
        # Default: batch evaluation with standard weights
        weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run evaluations
    console = Console()
    results = []

    for dense_weight in weights:
        console.print(f"\n{'=' * 70}")
        console.print(f"[bold]Evaluating dense_weight={dense_weight:.2f}...[/bold]")
        console.print(f"{'=' * 70}")

        try:
            metrics = retriever.evaluate(dense_weight=dense_weight, top_k=args.top_k)
            results.append(metrics)

            # Print result for this weight
            print_single_result(metrics, console)
        except Exception as e:
            logger.error(f"Evaluation failed for dense_weight={dense_weight}: {e}", exc_info=True)
            continue

    if not results:
        logger.error("All evaluations failed")
        sys.exit(1)

    # If batch evaluation, show comparison table
    if len(results) > 1:
        print_batch_summary(results, console)

        # Convert to dicts for JSON serialization
        results_dicts = [asdict(m) for m in results]
        df = pd.DataFrame(results_dicts)

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results_dicts, f, indent=2)
            logger.info(f"Results saved to {output_path}")

            # Save CSV
            csv_path = output_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV summary saved to {csv_path}")
        else:
            # Default output for batch mode
            default_output = script_dir / "evaluation_results.json"
            with open(default_output, "w") as f:
                json.dump(results_dicts, f, indent=2)
            csv_path = default_output.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {default_output}")
            logger.info(f"CSV summary saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()
