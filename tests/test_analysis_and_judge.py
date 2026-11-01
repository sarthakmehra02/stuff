from src.analyze_results import bootstrap_ci, mcnemar_exact
from src.judge import ANSWER_PROMPT, extract_json_object, format_prompt, is_refusal


def test_bootstrap_and_mcnemar_known_values():
    assert bootstrap_ci([True] * 10) == (1.0, 1.0)
    b, c, p = mcnemar_exact([True, True, False, False], [False, False, True, True])
    assert (b, c) == (2, 2)
    assert p == 1.0


def test_json_extraction_and_refusal_matching():
    assert extract_json_object('answer:\n```json\n{"verdict":"correct"}\n```') == {"verdict": "correct"}
    assert extract_json_object('preface {"x": 1} trailing') == {"x": 1}
    assert is_refusal(" Not found in the documents. ")
    assert not is_refusal("Not found in the document")


def test_answer_judge_prompt_formats_json_example():
    prompt = format_prompt(ANSWER_PROMPT, question="Q", gold="G", answer="A", answerable="true")
    assert '{"verdict":"...","reason":"one sentence"}' in prompt
