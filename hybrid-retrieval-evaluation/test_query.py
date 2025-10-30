"""Interactive query tool to test retrieval with custom queries."""

import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from main import HybridRetriever, embed_with_jina


def format_result(rank: int, doc_name: str, score: float, show_details: bool = False) -> str:
    """Format a single result for display."""
    bar_length = int(score * 40)  # 40 char max bar
    bar = "█" * bar_length
    return f"  {rank:2d}. [{score:6.4f}] {bar:<40s} {doc_name}"


def run_query(
    query: str, dense_weight: float = 0.5, top_k: int = 10, chroma_db_path: str = "./chroma_db", data_path: str = "./data.json"
) -> None:
    """Run a single query and display results."""

    print(f"\n{'=' * 80}")
    print(f"Query: {query}")
    print(f"Dense weight: {dense_weight:.2f} (Semantic: {dense_weight * 100:.0f}%, BM25: {(1 - dense_weight) * 100:.0f}%)")
    print(f"{'=' * 80}")

    # Initialize retriever
    retriever = HybridRetriever(chroma_db_path=chroma_db_path)
    retriever.setup_collections()
    retriever.process_and_index_documents(data_path)

    # Create query embedding
    print("\nCreating query embedding...")
    query_embedding = embed_with_jina([query], task="retrieval.query")[0]

    # Retrieve results
    print(f"Retrieving top {top_k} results...\n")
    results = retriever.hybrid_retrieve(query, query_embedding, dense_weight, top_k=top_k)

    # Display results
    print(f"Results (showing top {len(results)}):")
    print("-" * 80)
    for i, (doc_name, score) in enumerate(results, 1):
        print(format_result(i, doc_name, score))
    print("-" * 80)

    # Get dense and sparse components for comparison
    dense_results = retriever.retrieve_dense(query_embedding, top_k=top_k)
    sparse_results = retriever.retrieve_sparse(query, top_k=top_k)

    print("\nDense (Semantic) Top 5:")
    for i, (doc_name, score) in enumerate(dense_results[:5], 1):
        print(f"  {i}. {doc_name} ({score:.4f})")

    print("\nSparse (BM25) Top 5:")
    for i, (doc_name, score) in enumerate(sparse_results[:5], 1):
        print(f"  {i}. {doc_name} ({score:.4f})")

    print("\n" + "=" * 80)


def interactive_mode(chroma_db_path: str = "./chroma_db", data_path: str = "./data.json"):
    """Run in interactive mode, accepting queries from user."""

    print("=" * 80)
    print("Interactive Query Tool")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type a query to search")
    print("  - 'weight <0.0-1.0>' to change dense weight (e.g., 'weight 0.7')")
    print("  - 'topk <n>' to change number of results (e.g., 'topk 20')")
    print("  - 'quit' or 'exit' to quit")
    print()

    # Initialize retriever once
    retriever = HybridRetriever(chroma_db_path=chroma_db_path)
    retriever.setup_collections()
    retriever.process_and_index_documents(data_path)

    dense_weight = 0.5
    top_k = 10

    print(f"Current settings: dense_weight={dense_weight:.2f}, top_k={top_k}")
    print()

    while True:
        try:
            query = input("Query> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Handle weight command
            if query.lower().startswith("weight "):
                try:
                    new_weight = float(query.split()[1])
                    if 0.0 <= new_weight <= 1.0:
                        dense_weight = new_weight
                        print(
                            f"✓ Dense weight set to {dense_weight:.2f} (Semantic: {dense_weight * 100:.0f}%, BM25: {(1 - dense_weight) * 100:.0f}%)"
                        )
                    else:
                        print("✗ Weight must be between 0.0 and 1.0")
                except (IndexError, ValueError):
                    print("✗ Usage: weight <0.0-1.0>")
                continue

            # Handle topk command
            if query.lower().startswith("topk "):
                try:
                    new_topk = int(query.split()[1])
                    if new_topk > 0:
                        top_k = new_topk
                        print(f"✓ Top-K set to {top_k}")
                    else:
                        print("✗ Top-K must be positive")
                except (IndexError, ValueError):
                    print("✗ Usage: topk <number>")
                continue

            # Process actual query
            print(f"\n{'=' * 80}")
            print(f"Dense weight: {dense_weight:.2f} (Semantic: {dense_weight * 100:.0f}%, BM25: {(1 - dense_weight) * 100:.0f}%)")
            print(f"{'=' * 80}")

            # Create query embedding
            query_embedding = embed_with_jina([query], task="retrieval.query")[0]

            # Retrieve results
            results = retriever.hybrid_retrieve(query, query_embedding, dense_weight, top_k=top_k)

            # Display results
            print(f"\nTop {len(results)} results:")
            print("-" * 80)
            for i, (doc_name, score) in enumerate(results, 1):
                print(format_result(i, doc_name, score))
            print("-" * 80)

            # Show component results
            dense_results = retriever.retrieve_dense(query_embedding, top_k=5)
            sparse_results = retriever.retrieve_sparse(query, top_k=5)

            print("\nDense (Semantic) Top 5:")
            for i, (doc_name, score) in enumerate(dense_results, 1):
                marker = "★" if doc_name == results[0][0] else " "
                print(f"  {marker} {i}. {doc_name} ({score:.4f})")

            print("\nSparse (BM25) Top 5:")
            for i, (doc_name, score) in enumerate(sparse_results, 1):
                marker = "★" if doc_name == results[0][0] else " "
                print(f"  {marker} {i}. {doc_name} ({score:.4f})")

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Interactive query tool for hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python3 query_interactive.py "Which computer had a Z80 CPU?"

  # With custom weight (pure semantic)
  python3 query_interactive.py "Which computer had a Z80 CPU?" --dense-weight 1.0

  # Show more results
  python3 query_interactive.py "Which computer had a Z80 CPU?" --top-k 20

  # Interactive mode
  python3 query_interactive.py --interactive
        """,
    )

    parser.add_argument("query", nargs="?", help="Query string (omit for interactive mode)")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Weight for dense retrieval (0=pure BM25, 1=pure semantic)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--data", default=None, help="Path to documents JSON")
    parser.add_argument("--chroma-db", default=None, help="Path to ChromaDB storage")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Set default paths
    if args.data is None:
        args.data = str(script_dir / "data.json")
    if args.chroma_db is None:
        args.chroma_db = str(script_dir / "chroma_db")

    # Interactive mode or single query mode
    if args.interactive or args.query is None:
        interactive_mode(chroma_db_path=args.chroma_db, data_path=args.data)
    else:
        run_query(
            query=args.query, dense_weight=args.dense_weight, top_k=args.top_k, chroma_db_path=args.chroma_db, data_path=args.data
        )


if __name__ == "__main__":
    main()
