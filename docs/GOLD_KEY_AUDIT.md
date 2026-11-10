# Golden Test Key Audit

Audit date: 2026-11-02

## Scope and method

Reviewed all 60 entries in `data/test_qa.json` against the four source PDFs in
`data/raw_pdfs/`. Answerable entries were checked against the cited passages or
the matching passage elsewhere in the named source. The nine entries marked
unanswerable were checked for whether the requested fact appears in the supplied
corpus. This checks consistency with the project's source corpus; it does not
claim that the corpus itself is current or complete beyond those documents.

## Findings

- The substantive facts in the 51 answerable gold answers are supported by the
  supplied source documents. No numeric answer-key error was found.
- The nine `Not covered in the provided documents` answers are appropriate for
  this corpus. A year or scheme may be mentioned in a document without the
  specific requested statistic being present; for example, the PMFBY guidelines
  mention 2023-24 in an unrelated table, but do not give the KCC 2023-24
  interest-subvention rate asked in q014.
- q045's conclusion is supported for the person in the question: the PM-KISAN
  guidelines exclude persons who paid income tax in the last assessment year.
  The gold wording about an entire farmer family is broader than the clause's
  literal wording and should be read narrowly for this question.
- Some provenance metadata is incomplete for cross-document answers. In
  particular, q044's gold answer combines PM-KISAN Aadhaar rules with the
  2025-26 KCC circular, but its `source_file`/`source_pages` fields point only
  to PM-KISAN. q043 also discusses the older RBI circular alongside the newer
  PMFBY rule.
- Some page lists do not capture all supporting passages. q035 needs both the
  page describing the Tehsil/Taluk officer's 7-day action and the page describing
  the Appellate Committee's 15-day disposal. q057's answer is Illustration III
  in the 2025-26 KCC circular.

## Conclusion

Keep the answer texts and question set frozen for the answer audit: the source
check did not identify a substantive correction that needs to be made first.
Treat the provenance gaps above as metadata issues, not evidence that the facts
are wrong. Preserve this review alongside the original fixed test set.
