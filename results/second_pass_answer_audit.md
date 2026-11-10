# Second-Pass Test Answer Review

Date: 2026-11-02

This review first checked all 60 gold answers against the source PDFs in
`data/raw_pdfs/`, then compared the five sets of 60 generated answers against
that source-checked key. The nine unanswerable questions are correctly marked
as not covered by the supplied corpus. No substantive factual correction to a
gold answer was required. Some `source_file`/`source_pages` metadata is
incomplete; see `docs/GOLD_KEY_AUDIT.md`.

## Adjudication rule

- **Correct:** answers the question with the material requested facts and no
  material contradiction.
- **Partial:** contains a correct core fact but misses a material condition or
  includes a material error.
- **Wrong:** misses or contradicts the requested answer.
- For unanswerable questions, a refusal is correct; an unsupported specific
  claim is a hallucination.

The completed NIM judge labels were reviewed against the source-checked key.
The small manual adjustments below were applied to the summary counts. Other
labels were retained when the answer matched the rule above.

## Corrections to automated labels

- q002 prompted: partial -> wrong. It invents a 2-acre threshold and does not
  give PM-KISAN's family definition.
- q010 fine-tuned: partial -> wrong. It substitutes outstanding balance for
  the highest sanctioned loan amount.
- q031 zero-shot: partial -> wrong. It gives only generic insurance language,
  not the two required wild-animal indemnity thresholds.
- q057 RAG and fine-tuned+RAG: wrong -> partial. Both give the overall Rs 3 lakh
  figure but misstate its composition as crop-loan-only; the source illustration
  splits it into Rs 1.75 lakh crop loan plus Rs 1.25 lakh allied activities.
- q058 zero-shot: refused -> hallucinated. It declines to identify the bank but
  adds a specific, unsupported claim about PM-KISAN being administered through
  the National Food Security Mission.

## Second-pass scores

Exact overall correctness counts a correct answer to an answerable question or
a refusal to an unanswerable question. Partial answers are not counted as exact.

| Method | Answerable correct | Unanswerable refused | Exact overall | Exact overall rate |
|---|---:|---:|---:|---:|
| Zero-shot | 1/51 | 6/9 | 7/60 | 11.7% |
| Prompted | 4/51 | 9/9 | 13/60 | 21.7% |
| RAG | 21/51 | 9/9 | 30/60 | 50.0% |
| Fine-tuned | 6/51 | 0/9 | 6/60 | 10.0% |
| Fine-tuned + RAG | 30/51 | 0/9 | 30/60 | 50.0% |

Partial grades are inherently more subjective than exact correctness. These
counts are an assistant's second-pass review of the model judge outputs, not a
blind independent human annotation. Preserve the raw answers and original
judge files; do not tune model prompts or training data against these test
answers. A separate human audit is still appropriate before presenting the
scores as final project findings.
