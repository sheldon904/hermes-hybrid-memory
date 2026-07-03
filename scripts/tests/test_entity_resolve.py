"""entity_resolve: merge behavior unchanged + near-miss alias capture."""

from entity_resolve import (ALIAS_CANDIDATE_THRESHOLD, FUZZY_THRESHOLD, Resolver)


def _fresh():
    return Resolver(index={"keys": {}, "canon": {}})


def test_exact_and_strong_fuzzy_still_merge():
    rz = _fresh()
    a = rz.canonical("Acme Construction Group", hint_type="company")
    b = rz.canonical("Acmee Construction Group", hint_type="company")  # ~0.98
    assert a == b
    assert rz.pending_aliases == []


def test_near_miss_recorded_not_merged():
    rz = _fresh()
    a = rz.canonical("Nimbus Platform", hint_type="project")
    b = rz.canonical("Nimbo Platform", hint_type="project")  # ratio ~0.889
    assert a != b
    assert len(rz.pending_aliases) == 1
    new, near, score = rz.pending_aliases[0]
    assert {new, near} == {a, b}
    assert ALIAS_CANDIDATE_THRESHOLD <= score < FUZZY_THRESHOLD


def test_distant_names_no_alias_record():
    rz = _fresh()
    rz.canonical("Acme Labs", hint_type="company")
    rz.canonical("Acme Digital", hint_type="company")  # ratio ~0.69
    assert rz.pending_aliases == []
