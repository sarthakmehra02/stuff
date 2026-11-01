"""Create a separate, explicitly synthetic three-fact update experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LABEL = "SYNTHETIC EXPERIMENT — NOT A REAL AMENDMENT"

EDITS = [
    {"chunk_id": "pmkisan_operational_guid_p003_c003",
     "old": "cut-off date for determination of eligibility of beneficiaries under the scheme shall be 01.02.2019",
     "new": "cut-off date for determination of eligibility of beneficiaries under the scheme shall be 01.02.2020",
     "scheme": "PM-KISAN", "source_file": "pmkisan_operational_guidelines_2020.pdf", "page": 3,
     "old_answer": "The original PM-KISAN eligibility cut-off date in the source text is 01.02.2019.",
     "qas": [
         ("In the synthetic PM-KISAN update, what cut-off date determines beneficiary eligibility?", "The synthetic cut-off date is 01.02.2020."),
         ("According to the clearly synthetic PM-KISAN experiment, which date is used for eligibility?", "The synthetic cut-off date is 01.02.2020."),
         ("For this synthetic update only, what is the PM-KISAN eligibility date?", "The synthetic cut-off date is 01.02.2020."),
     ]},
    {"chunk_id": "pmfby_operational_guidel_p047_c020",
     "old": "2.0% of SI or Actuarial rate, whichever is less",
     "new": "2.5% of SI or Actuarial rate, whichever is less",
     "scheme": "PMFBY", "source_file": "pmfby_operational_guidelines_2023.pdf", "page": 47,
     "old_answer": "The original maximum farmer premium is 2.0% of the sum insured or actuarial rate, whichever is less.",
     "qas": [
         ("In the synthetic PMFBY update, what maximum premium rate applies to Kharif food-grain and oilseed crops?", "The synthetic maximum is 2.5% of the sum insured or actuarial rate, whichever is less."),
         ("What Kharif premium cap is used in the synthetic PMFBY experiment?", "The synthetic maximum is 2.5% of the sum insured or actuarial rate, whichever is less."),
         ("For this synthetic change, what is the PMFBY Kharif farmer premium ceiling?", "The synthetic maximum is 2.5% of the sum insured or actuarial rate, whichever is less."),
     ]},
    {"chunk_id": "kcc_interest_subvention__p002_c004",
     "old": "2025-26 7% 1.50%",
     "new": "2025-26 8% 1.50%",
     "scheme": "KCC", "source_file": "kcc_interest_subvention_2025_26.pdf", "page": 2,
     "old_answer": "The original farmer lending rate for 2025-26 is 7% per year.",
     "qas": [
         ("In the synthetic 2025-26 KCC update, what is the lending rate to farmers?", "The synthetic lending rate is 8% per year."),
         ("What lending rate does the synthetic KCC experiment use for 2025-26?", "The synthetic lending rate is 8% per year."),
         ("For the synthetic 2025-26 circular only, state the KCC farmer lending rate.", "The synthetic lending rate is 8% per year."),
         ("Has the synthetic KCC example changed the 2025-26 lending rate to 8%?", "Yes. The synthetic lending rate is 8% per year."),
     ]},
]


def create_update(source: Path = ROOT / "data" / "processed" / "chunks.json",
                  out_dir: Path = ROOT / "data" / "update_test"):
    chunks = json.loads(source.read_text(encoding="utf-8"))
    by_id = {chunk["id"]: chunk for chunk in chunks}
    manifest_edits = []
    for edit in EDITS:
        if edit["chunk_id"] not in by_id:
            raise KeyError(f"Source chunk missing: {edit['chunk_id']}")
        text = by_id[edit["chunk_id"]]["text"]
        if text.count(edit["old"]) != 1:
            raise ValueError(f"Expected exactly one source occurrence for synthetic edit {edit['chunk_id']}")
        by_id[edit["chunk_id"]]["text"] = text.replace(edit["old"], edit["new"], 1)
        by_id[edit["chunk_id"]]["synthetic_update_label"] = LABEL
        manifest_edits.append({"chunk_id": edit["chunk_id"], "old_text": edit["old"], "new_text": edit["new"]})
    questions = []
    for edit in EDITS:
        for question, answer in edit["qas"]:
            questions.append({"id": f"update_{len(questions)+1:03d}", "question": question,
                              "gold_answer": answer, "type": "synthetic_update", "scheme": edit["scheme"],
                              "old_answer": edit["old_answer"],
                              "answerable": True, "source_file": edit["source_file"], "source_pages": [edit["page"]],
                              "synthetic": True, "synthetic_update_label": LABEL})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks_modified.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "update_qa.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "update_manifest.json").write_text(json.dumps({"label": LABEL, "edits": manifest_edits,
        "notice": "These edits are synthetic and do not reflect real amendments."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "SYNTHETIC_NOTICE.txt").write_text(LABEL + ". These facts are invented for the experiment and are not real amendments.\n", encoding="utf-8")
    from src.rag import build_index
    build_index(chunks_path=out_dir / "chunks_modified.json", index_path=out_dir / "faiss_index.bin",
                metadata_path=out_dir / "faiss_metadata.json", manifest_path=out_dir / "faiss_manifest.json")
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "processed" / "chunks.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "update_test")
    args = parser.parse_args()
    questions = create_update(args.source, args.out_dir)
    print(f"Created {len(questions)} synthetic update questions in {args.out_dir}")


if __name__ == "__main__":
    main()
