import json

import pytest

from alignment.generate import build_pairs
from alignment.judge import judge_pairs, judge_prompt, parse_decisions, summary_for
from alignment.utils import atomic_json, read_json, read_jsonl, stable_hash, write_jsonl


def pairs(count=3, track="hh"):
    return [{"id": str(i), "track": track, "instruction": "question", "response_a": "answer1", "response_b": "answer2",
             "category": "benign", "hidden_metadata": {"model_a": "secret-sft", "model_b": "secret-dpo",
                                                        "baseline": "secret-sft", "candidate": "secret-dpo"}}
            for i in range(count)]


def test_judge_identity_is_not_in_prompt():
    prompt = judge_prompt(pairs(), "hh")
    assert "secret" not in prompt and "hidden_metadata" not in prompt
    assert "Response A" in prompt and "Response B" in prompt


def test_parser_handles_fences_and_rejects_duplicate_unknown_ids():
    assert parse_decisions('```json\n[{"id":"1","winner":"A","reason":"good"}]\n```', ["1"])["1"]["winner"] == "A"
    assert parse_decisions('[{"id":"1","winner":"A","reason":"x"},{"id":"1","winner":"B","reason":"x"}]', ["1"]) == {}
    with pytest.raises(ValueError, match="unknown IDs"):
        parse_decisions('[{"id":"2","winner":"A","reason":"x"}]', ["1"])


def test_partial_batches_retry_only_missing_and_resume_skips_completed(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, pairs())
    calls, waits = [], []
    def request(prompt):
        visible = json.loads(prompt.split("Quoted evaluation data:\n")[1])
        calls.append([r["id"] for r in visible])
        if len(calls) == 1:
            return '[{"id":"0","winner":"B","reason":"better"}]'
        return json.dumps([{"id": r["id"], "winner": "TIE", "reason": "equal"} for r in visible])
    summary = judge_pairs(path, tmp_path / "judged", request=request, sleeper=waits.append)
    assert calls == [["0", "1", "2"], ["1", "2"]]
    assert waits == [5., 5.]
    assert summary["quality"]["wins"] == 1 and summary["quality"]["ties"] == 2
    def forbidden(_):
        pytest.fail("Completed pairs must not be rejudged")
    resumed = judge_pairs(path, tmp_path / "new", resume_from=tmp_path / "judged", request=forbidden, sleeper=waits.append)
    assert resumed == summary
    assert len(read_jsonl(tmp_path / "new/judgments.jsonl")) == 3


def test_failures_are_persisted_and_retried_on_rerun(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, pairs(1))
    def failure(_):
        raise RuntimeError("proxy denied")
    result = judge_pairs(path, tmp_path / "out", request=failure, sleeper=lambda _: None)
    assert result["quality"]["failures"] == 1 and result["request_attempts"] == 3
    success = lambda _: '[{"id":"0","winner":"UNKNOWN","reason":"cannot assess"}]'
    result = judge_pairs(path, tmp_path / "out", request=success, sleeper=lambda _: None)
    assert result["quality"]["failures"] == 0 and result["quality"]["unknown"] == 1
    assert result["quality"]["win_rate_without_ties"] is None
    assert read_jsonl(tmp_path / "out/failures.jsonl") == []
    with pytest.raises(ValueError, match="differs"):
        judge_pairs(path, tmp_path / "out", judge_model="different", request=success, sleeper=lambda _: None)


def test_denominators_unknown_failure_and_persona_style():
    records = pairs(5, "persona")
    judgments = [{"id": str(i), "winner": winner, "style_winner": "B"}
                 for i, winner in enumerate(("B", "A", "TIE", "UNKNOWN"))]
    summary = summary_for(records, judgments, [{"id": "4"}])
    assert summary["win_rate_without_ties"] == .5 and summary["tie_adjusted_win_rate"] == .5
    assert summary["unknown"] == 1 and summary["failures"] == 1
    assert summary_for(records, judgments, [{"id": "4"}], "style_winner")["wins"] == 4
    assert parse_decisions('[{"id":"0","winner":"A","reason":"x"}]', ["0"], persona=True) == {}


def test_pairs_validate_generation_settings_and_randomize_reproducibly(tmp_path):
    prompt_rows = [{"id": str(i), "instruction": "Question", "dataset": "test"} for i in range(12)]
    first, second = tmp_path / "sft.json", tmp_path / "dpo.json"
    for path, name in ((first, "sft"), (second, "dpo")):
        atomic_json(path, [{**row, "output": name + " answer", "generator": name} for row in prompt_rows])
        atomic_json(path.with_suffix(".manifest.json"), {"generator": name, "prompt_hash": stable_hash(prompt_rows),
                                                       "decoding": {"do_sample": False}})
    build_pairs(first, second, tmp_path / "a.jsonl", "hh")
    build_pairs(first, second, tmp_path / "b.jsonl", "hh")
    assert read_jsonl(tmp_path / "a.jsonl") == read_jsonl(tmp_path / "b.jsonl")
    assert {r["hidden_metadata"]["model_a"] for r in read_jsonl(tmp_path / "a.jsonl")} == {"sft", "dpo"}
    metadata = read_json(second.with_suffix(".manifest.json"))
    metadata["decoding"]["do_sample"] = True
    atomic_json(second.with_suffix(".manifest.json"), metadata)
    with pytest.raises(ValueError, match="identical decoding"):
        build_pairs(first, second, tmp_path / "c.jsonl", "hh")
