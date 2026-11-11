"""Run one advanced retrieval arm for a selected generator model."""
import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--mode", choices=["semantic", "reranked", "hybrid"], default="hybrid")
    parser.add_argument("--conditions", default="zero_shot,prompted,rag")
    args = parser.parse_args()
    os.environ["FARMER_MODEL_ID"] = args.model_id

    from src.models import generate, load_model
    from src.rag import retrieve
    from src.run_conditions import run_conditions
    from src.hybrid import HybridRetriever

    if args.mode == "semantic":
        retriever = retrieve
    elif args.mode == "reranked":
        def retriever(question, k=4):
            return retrieve(question, k=k, rerank=True, candidate_k=20)
    else:
        retriever = HybridRetriever(candidate_k=20, top_k=4, min_rerank_score=0.5)

    model, tokenizer = load_model(adapter_path=args.adapter)
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    run_conditions(conditions, out=args.out, adapter=args.adapter, model=model,
                   tokenizer=tokenizer, retrieve_fn=retriever, generate_fn=generate)
    print(f"Completed {conditions} with {args.model_id} ({args.mode}) -> {args.out}")


if __name__ == "__main__":
    main()
