"""Five-condition interactive demo. Run on a Colab T4 with an adapter path."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_ID, TOP_K
from src.prompts import CONDITIONS, build_messages

logger = logging.getLogger("farmer_demo")
_MODEL = _TOKENIZER = None


def launch(adapter: str, share: bool = True):
    import gradio as gr
    from src.models import load_model, generate
    from src.rag import build_index, retrieve

    global _MODEL, _TOKENIZER
    if not adapter:
        raise ValueError("A trained LoRA adapter is required for the five-condition demo")
    build_index()
    _MODEL, _TOKENIZER = load_model(adapter_path=adapter)

    def answer_all(question: str):
        if not question.strip():
            raise gr.Error("Enter a question about PM-KISAN, PMFBY, or KCC.")
        found = retrieve(question, k=TOP_K)
        output = []
        for condition in CONDITIONS:
            retrieved = found if condition in {"rag", "finetuned_rag"} else None
            messages = build_messages(condition, question, chunks=retrieved)
            result = generate(_MODEL, _TOKENIZER, messages, use_adapter=condition in {"finetuned", "finetuned_rag"})
            output.extend([result["text"], f"{result['latency_s']:.2f} s"])
            if condition in {"rag", "finetuned_rag"}:
                output.append("\n\n".join(
                    f"**{item.get('source_file')} — Page {item.get('page')}** (score {item.get('score', 0):.3f})\n\n{item.get('text', '')}"
                    for item in found
                ) or "No chunks retrieved.")
        return output

    with gr.Blocks(title="Indian Farmer Schemes QA Comparison") as demo:
        gr.Markdown(f"# Farmer scheme answer comparison\nModel: `{MODEL_ID}`. Source documents are dated 2018–2026.")
        question = gr.Textbox(label="Ask about PM-KISAN, PMFBY, or KCC", lines=2)
        run = gr.Button("Compare answers", variant="primary")
        components = []
        with gr.Row():
            for condition in CONDITIONS:
                with gr.Column():
                    gr.Markdown(f"### {condition}")
                    answer = gr.Textbox(label="Answer", lines=5, interactive=False)
                    latency = gr.Textbox(label="Latency", interactive=False)
                    components.extend([answer, latency])
                    if condition in {"rag", "finetuned_rag"}:
                        with gr.Accordion("Retrieved sources", open=False):
                            sources = gr.Markdown()
                        components.append(sources)
        run.click(answer_all, inputs=question, outputs=components)
        question.submit(answer_all, inputs=question, outputs=components)
    demo.queue().launch(share=share)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--share", action="store_true", default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    launch(args.adapter, share=args.share)


if __name__ == "__main__":
    main()
