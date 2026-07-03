"""memory_abstract cluster rules: hub guard, size bounds, age gate, never-delete."""

import memory_abstract as ma
from conftest import add_fact


def _cluster_ids(clusters):
    return [sorted(f["fact_id"] for f in facts) for _, facts in clusters]


def test_shared_entity_pair_clusters(facts_db):
    path, con = facts_db
    ids = [add_fact(con, f"alex fact {i}", entities=["Alex", "Riverside U"]) for i in range(4)]
    clusters = ma.find_clusters(con)
    assert _cluster_ids(clusters) == [sorted(ids)]


def test_hub_entity_does_not_connect(facts_db):
    path, con = facts_db
    # 20 facts sharing ONLY one big entity (> HUB_MAX_FACTS) must not cluster.
    for i in range(20):
        add_fact(con, f"hub fact {i}", entities=["Owner"])
    assert ma.find_clusters(con) == []


def test_too_small_and_too_large_skipped(facts_db):
    path, con = facts_db
    for i in range(2):  # size 2 < MIN_CLUSTER
        add_fact(con, f"small {i}", entities=["TinyCo", "TinyTown"])
    for i in range(ma.MAX_CLUSTER + 1):  # size 13 > MAX_CLUSTER
        add_fact(con, f"big {i}", entities=["BigCo", "BigTown"])
    assert ma.find_clusters(con) == []


def test_young_facts_excluded(facts_db):
    path, con = facts_db
    add_fact(con, "old a", entities=["X Corp", "Y City"], created_days_ago=10)
    add_fact(con, "old b", entities=["X Corp", "Y City"], created_days_ago=10)
    add_fact(con, "young c", entities=["X Corp", "Y City"], created_days_ago=1)
    # only 2 old facts -> below MIN_CLUSTER, so nothing
    assert ma.find_clusters(con) == []


def test_chunked_members_excluded_and_never_deleted(facts_db):
    path, con = facts_db
    ids = [add_fact(con, f"nimbus fact {i}", entities=["Nimbus", "Acme Construction"])
           for i in range(3)]
    con.execute("INSERT INTO facts(content, category, tags) VALUES ('chunk summary', 'chunk', 'chunk,sig:abc')")
    chunk_id = con.execute("SELECT fact_id FROM facts WHERE category='chunk'").fetchone()[0]
    for mid in ids:
        con.execute("INSERT INTO chunk_members(chunk_fact_id, member_fact_id) VALUES (?,?)",
                    (chunk_id, mid))
    con.commit()
    assert ma.find_clusters(con) == []  # members not re-clustered
    # never-delete invariant: all members still present
    n = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert n == 4


def test_fingerprint_cosine_groups():
    from collections import Counter
    a = Counter({("out", "OFFERS_JOB", "idea"): 2, ("in", "ABOUT", "company"): 1})
    b = Counter({("out", "OFFERS_JOB", "idea"): 2, ("in", "ABOUT", "company"): 1})
    c = Counter({("out", "LOCATED_IN", "location"): 1})
    assert abs(ma._cos(a, b) - 1.0) < 1e-9
    assert ma._cos(a, c) == 0.0
