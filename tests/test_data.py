from alignment.data import normalize_hh, normalize_sft, normalize_uf, prepare_persona, select_rows
from alignment.utils import read_json


def test_hh_rejects_multiturn_and_mismatched_prompts():
    pair = {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello", "rejected": "\n\nHuman: Hi\n\nAssistant: Go away"}
    assert normalize_hh(pair) == {"prompt": "Hi", "chosen": "Hello", "rejected": "Go away"}
    assert normalize_hh({**pair, "chosen": pair["chosen"] + "\n\nHuman: More\n\nAssistant: Okay"}) is None
    assert normalize_hh({**pair, "rejected": pair["rejected"].replace("Hi", "Bye")}) is None


def test_first_exchange_and_preference_roles():
    user = {"role": "user", "content": "Question"}
    answer = {"role": "assistant", "content": "Answer"}
    assert normalize_sft({"messages": [user, answer, user]}) == {"prompt": "Question", "response": "Answer"}
    assert normalize_uf({"chosen": [user, answer], "rejected": [user, {**answer, "content": "Wrong"}]})["chosen"] == "Answer"
    assert normalize_uf({"chosen": [user, answer, user], "rejected": [user, answer]}) is None


def test_subset_is_deterministic_and_removes_overlap():
    source = [{"prompt": str(i), "response": "x"} for i in range(10)]
    source.append(source[0])
    a, counts = select_rows(source, lambda x: x.copy(), 4, 42, excluded=["1"])
    b, _ = select_rows(source, lambda x: x.copy(), 4, 42, excluded=["1"])
    assert a == b and all(r["prompt"] != "1" for r in a)
    assert counts["duplicate_or_overlapping_prompt"] == 2


def test_persona_split_and_eval_are_disjoint(tmp_path):
    destination = prepare_persona(tmp_path)
    train, validation = read_json(destination / "train.json"), read_json(destination / "validation.json")
    assert len(train) + len(validation) == 220
    assert len(validation) == 22
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in validation}
    from alignment.utils import ROOT
    evaluation = read_json(ROOT / "eval_sets/persona_eval_40_seed42.json")
    assert len(evaluation) == 40
    assert not {r["prompt"] for r in train + validation} & {r["instruction"] for r in evaluation}
