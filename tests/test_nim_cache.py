from src.nim_client import NIMResponseCache


def test_request_options_are_part_of_cache_key(tmp_path):
    cache = NIMResponseCache(str(tmp_path / "nim.sqlite"))
    messages = [{"role": "user", "content": "Return JSON"}]
    plain = cache.compute_cache_key("teacher", messages, 0.3)
    explicit_plain = cache.compute_cache_key("teacher", messages, 0.3, None)
    no_thinking = cache.compute_cache_key("teacher", messages, 0.3, {"reasoning_effort": "none"})
    assert plain == explicit_plain
    assert plain != no_thinking
    cache.set("teacher", messages, 0.3, "[]", request_options={"reasoning_effort": "none"})
    assert cache.get("teacher", messages, 0.3, {"reasoning_effort": "none"}) == "[]"
    assert cache.get("teacher", messages, 0.3) is None
