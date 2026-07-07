#!/usr/bin/env python3
"""entity_resolve.py, write-time entity canonicalization for the memory graph.

Collapses name variants / shared domains / shared contact info to ONE canonical
entity, so the same company or person seen via email, calls, agency leads, and
curated memory resolves to a single graph node, while keeping genuinely distinct
siblings (e.g. "Acme Labs" vs "Acme Digital") separate.

Stdlib-only (difflib for the optional fuzzy gate). Persistent index at
~/.hermes/knowledge_graph/entity_index.json (reversible, delete to reset).

Used at WRITE time by memory_ingest._append_overlay and build_kg._merge_overlay.
The live hybrid provider resolves by exact match at READ time, so baking canonical
names in at write time is sufficient and never touches the install tree.

Index shape:
  {"keys":  {"<mergekey>": "<Canonical Display>"},
   "canon": {"<Canonical Display>": {"type": "company|person|...", "norm": "...",
                                      "keys": ["norm:...","domain:...", ...]}}}
Merge keys: norm:<normalized-name> | domain:<d> | email:<e> | phone:<last10>
"""

import difflib
import json
import os
import re
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
KG_DIR = HERMES_HOME / "knowledge_graph"
INDEX_PATH = KG_DIR / "entity_index.json"

FUZZY_THRESHOLD = 0.92
# Fuzzy scores in [ALIAS_CANDIDATE_THRESHOLD, FUZZY_THRESHOLD) are "close-call"
# resolutions: not merged, but recorded on Resolver.pending_aliases so callers
# can write reviewable POSSIBLE_ALIAS edges instead of deciding silently.
ALIAS_CANDIDATE_THRESHOLD = 0.85
# Corporate suffixes / legal forms stripped during normalization (standalone words only).
_SUFFIX = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|co|corp|corporation|company|"
    r"pllc|pa|plc|llp|lp|gmbh|sa|nv|bv|pty)\b", re.I)
_PUNCT = re.compile(r"[.,/&'\"()\-_:;|]+")


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def normalize(name) -> str:
    """Lowercase, drop legal suffixes + punctuation + leading 'the', collapse space."""
    if not isinstance(name, str):
        return ""
    n = name.lower()
    n = _PUNCT.sub(" ", n)
    n = _SUFFIX.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    if n.startswith("the "):
        n = n[4:].strip()
    return n


def domain_of(url_or_email):
    """Extract a bare domain from a URL or email; None if not derivable."""
    if not isinstance(url_or_email, str) or not url_or_email.strip():
        return None
    s = url_or_email.strip().lower()
    if "@" in s:
        d = s.rsplit("@", 1)[1]
    else:
        d = re.sub(r"^[a-z]+://", "", s)
        d = d.split("/", 1)[0]
    d = re.sub(r"^www\.", "", d).split(":", 1)[0].strip()
    return d if ("." in d and 3 <= len(d) <= 60) else None


def phone_key(s):
    """Last 10 digits of a phone number; None if fewer than 10 digits."""
    if not isinstance(s, str):
        return None
    digits = re.sub(r"\D", "", s)
    return digits[-10:] if len(digits) >= 10 else None


def clean_display(name) -> str:
    """Display name that satisfies build_kg._clean_name (<=40 chars, keeps a capital)."""
    n = " ".join(str(name).strip().strip(".,;:'\"").split())
    if len(n) > 40:
        n = n[:40].rstrip()
    return n


def _merge_keys(norm, domain=None, email=None, phone=None):
    """Ordered strongest-first so identity keys (domain/email/phone) win over name."""
    keys = []
    d = domain_of(domain) or domain_of(email)
    if d:
        keys.append(f"domain:{d}")
    em = (email or "").strip().lower()
    if em and "@" in em:
        keys.append(f"email:{em}")
    pk = phone_key(phone)
    if pk:
        keys.append(f"phone:{pk}")
    if norm:
        keys.append(f"norm:{norm}")
    return keys


# ---------------------------------------------------------------------------
# Resolver (load once, resolve many, save once)
# ---------------------------------------------------------------------------

class Resolver:
    def __init__(self, index: dict | None = None):
        self.idx = index if index is not None else _load_index()
        self.idx.setdefault("keys", {})
        self.idx.setdefault("canon", {})
        # (new_display, near_miss_canonical, score) tuples from close-call
        # fuzzy matches this session; drained by memory_ingest._append_overlay.
        self.pending_aliases = []

    def canonical(self, name, *, domain=None, email=None, phone=None, hint_type=None):
        """Return the canonical display name for an entity, registering/merging it.

        Resolution order: exact merge-key hit (domain>email>phone>name) → fuzzy
        name match (same type, >=0.90) → register a new canonical.
        """
        norm = normalize(name)
        if not norm:
            return clean_display(name)
        keys = self.idx["keys"]
        canon = self.idx["canon"]
        mks = _merge_keys(norm, domain, email, phone)

        hit = None
        near_miss = None
        for k in mks:                      # strongest-first direct hit
            if k in keys:
                hit = keys[k]
                break
        if hit is None:                    # fuzzy fallback on normalized name
            new_domains = {k[7:] for k in mks if k.startswith("domain:")}
            best, cand = 0.0, None
            for cname, meta in canon.items():
                if hint_type and meta.get("type") and meta["type"] != hint_type:
                    continue
                cnorm = meta.get("norm", "")
                if not cnorm or cnorm[0] != norm[0]:   # first-char bucket bounds cost
                    continue
                # distinct-domain guard: never fuzzy-merge two orgs whose domains differ
                # (most business records carry a domain, so this stops same-look
                # near-namesakes in the same vertical from merging incorrectly).
                cand_domains = {k[7:] for k in meta.get("keys", []) if k.startswith("domain:")}
                if new_domains and cand_domains and not (new_domains & cand_domains):
                    continue
                r = difflib.SequenceMatcher(None, norm, cnorm).ratio()
                if r > best:
                    best, cand = r, cname
            if cand and best >= FUZZY_THRESHOLD:
                hit = cand
            elif cand and best >= ALIAS_CANDIDATE_THRESHOLD:
                near_miss = (cand, round(best, 3))

        if hit is not None:
            meta = canon.setdefault(hit, {"type": hint_type, "norm": normalize(hit), "keys": []})
            if hint_type and not meta.get("type"):
                meta["type"] = hint_type
            for k in mks:
                keys[k] = hit
                if k not in meta["keys"]:
                    meta["keys"].append(k)
            return hit

        disp = clean_display(name)
        canon[disp] = {"type": hint_type, "norm": norm, "keys": list(mks)}
        for k in mks:
            keys[k] = disp
        if near_miss:
            self.pending_aliases.append((disp, near_miss[0], near_miss[1]))
        return disp

    def add_alias(self, variant, canonical_display, hint_type=None):
        """Force a name variant to resolve to an existing canonical (seed curated aliases)."""
        nv = normalize(variant)
        meta = self.idx["canon"].setdefault(
            canonical_display, {"type": hint_type, "norm": normalize(canonical_display), "keys": []})
        if hint_type and not meta.get("type"):
            meta["type"] = hint_type
        if nv:
            self.idx["keys"][f"norm:{nv}"] = canonical_display
            if f"norm:{nv}" not in meta["keys"]:
                meta["keys"].append(f"norm:{nv}")

    def save(self):
        _save_index(self.idx)


def _load_index() -> dict:
    if INDEX_PATH.exists():
        try:
            d = json.loads(INDEX_PATH.read_text())
            d.setdefault("keys", {})
            d.setdefault("canon", {})
            return d
        except Exception:
            pass
    return {"keys": {}, "canon": {}}


def _save_index(idx: dict) -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(idx, ensure_ascii=False, indent=1))


def canonical(name, **kw) -> str:
    """One-off convenience (loads + saves the index each call)."""
    r = Resolver()
    out = r.canonical(name, **kw)
    r.save()
    return out


# ---------------------------------------------------------------------------
# Relation ontology, collapse ad-hoc LLM predicates to a CLOSED controlled
# vocabulary so multi-hop paths, fingerprints, and analogy search stay
# consistent. Since 2026-07-06 unknown relations no longer pass through (that
# grew 344 distinct relations for 1,607 edges): exact map -> flip map (inverse
# forms, caller swaps src/dst) -> ordered stem rules -> RELATED_TO. The raw
# extractor string is preserved in edges.rel_orig.
# ---------------------------------------------------------------------------

_RELATION_CANON = {
    # work / org
    "works_for": "WORKS_FOR", "works_at": "WORKS_FOR", "employed_by": "WORKS_FOR",
    "employee_of": "WORKS_FOR", "staff_of": "WORKS_FOR",
    "founded": "FOUNDED", "founder_of": "FOUNDED", "co_founded": "FOUNDED", "cofounded": "FOUNDED",
    "owns": "OWNS", "owner_of": "OWNS",
    "ceo_of": "LEADS", "cto_of": "LEADS", "coo_of": "LEADS", "cfo_of": "LEADS",
    "leads": "LEADS", "runs": "LEADS", "director_of": "LEADS", "manages": "LEADS",
    "co_founder_of": "FOUNDED", "cofounder_of": "FOUNDED",
    "founded_with": "AFFILIATED_WITH", "acquired": "OWNS", "equity_holder": "OWNS",
    "contact_at": "CONTACT_AT", "point_of_contact": "CONTACT_AT", "poc_for": "CONTACT_AT",
    "has_point_of_contact": "CONTACT_AT",
    # commercial
    "client_of": "CLIENT_OF", "customer_of": "CLIENT_OF",
    "prospect": "PROSPECT", "prospect_of": "PROSPECT", "potential_client": "PROSPECT", "lead": "PROSPECT",
    "contacted": "CONTACTED", "reached_out_to": "CONTACTED",
    "vendor_of": "VENDOR_OF", "supplier_of": "VENDOR_OF",
    "partner_of": "PARTNER_OF", "reseller_of": "PARTNER_OF", "distributor_for": "PARTNER_OF",
    "applied_to": "APPLIED_TO", "interviewing_with": "APPLIED_TO",
    "interviewed_with": "APPLIED_TO", "interviewed_at": "APPLIED_TO",
    "interviewed_by": "APPLIED_TO",
    "met": "CONTACTED", "met_with": "CONTACTED", "worked_on": "HAS_PROJECT",
    "application_deadline_is": "RELATED_TO",
    "interested_in": "INTERESTED_IN", "wants": "INTERESTED_IN", "evaluating": "INTERESTED_IN",
    "offers": "OFFERS", "provides": "OFFERS", "sells": "OFFERS",
    "owes": "OWES", "has_outstanding_invoice": "OWES",
    "referral_from": "REFERRAL_FROM", "referred_by": "REFERRAL_FROM",
    # place / category
    "located_in": "LOCATED_IN", "based_in": "LOCATED_IN", "in_city": "LOCATED_IN",
    "operates_in": "LOCATED_IN", "headquartered_in": "LOCATED_IN", "serves": "LOCATED_IN",
    "lives_in": "LIVES_IN", "resides_in": "LIVES_IN",
    "in_industry": "IN_INDUSTRY", "industry": "IN_INDUSTRY", "sector": "IN_INDUSTRY",
    "category": "IN_INDUSTRY", "targets": "TARGETS",
    # project / tech
    "has_project": "HAS_PROJECT", "works_on": "HAS_PROJECT", "building": "HAS_PROJECT",
    "uses": "USES", "uses_stack": "USES", "uses_tool": "USES", "built_with": "USES",
    "deployment_of": "DEPLOYS", "deploys": "DEPLOYS",
    # personal
    "married_to": "SPOUSE_OF", "spouse_of": "SPOUSE_OF", "dating": "PARTNER_OF",
    "girlfriend_of": "PARTNER_OF", "boyfriend_of": "PARTNER_OF",
    "in_relationship_with": "PARTNER_OF",
    "mother_of": "PARENT_OF", "father_of": "PARENT_OF",
    "parent_of": "PARENT_OF", "sibling_of": "SIBLING_OF",
    "step_child_of": "STEP_CHILD_OF",
    "attended": "ATTENDS", "attends": "ATTENDS", "studies_at": "ATTENDS",
    "student_at": "ATTENDS", "graduated_from": "ATTENDS",
    "mentioned": "ABOUT", "about": "ABOUT", "related_to": "RELATED_TO",
    # jobs pipeline (observed variants, 2026-07-06 vocabulary closure)
    "has_job": "HIRING", "offers_job": "HIRING", "offers_role": "HIRING",
    "has_job_posting": "HIRING", "has_job_opening": "HIRING", "posted_job": "HIRING",
    "posted": "HIRING", "hiring": "HIRING", "hiring_for": "HIRING",
    "applied_to": "APPLIED_TO", "applied_for": "APPLIED_TO", "interviewed_for": "APPLIED_TO",
    "applied_via": "USES", "posted_on": "POSTED_ON",
    "recruiter_at": "WORKS_FOR", "recruiter_for": "WORKS_FOR",
    # location variants
    "location": "LOCATED_IN", "is_location": "LOCATED_IN", "in_location": "LOCATED_IN",
    "at_location": "LOCATED_IN", "is_located_in": "LOCATED_IN",
    "location_of_job_opening": "LOCATED_IN", "search_location": "LOCATED_IN",
    "work_location": "LOCATED_IN", "for_positions_in": "LOCATED_IN",
    "has_job_posting_in": "LOCATED_IN", "offered_in": "LOCATED_IN",
    "offers_onsite_in": "LOCATED_IN", "offers_hybrid_in": "LOCATED_IN",
    # attribute-shaped noise -> generic
    "has_salary": "RELATED_TO", "effective_date": "RELATED_TO",
    "expires_on": "RELATED_TO", "application_deadline": "RELATED_TO",
    # identity entries so the closed vocabulary + system relations round-trip
    "travels_to": "TRAVELS_TO", "part_of": "PART_OF", "member_of": "MEMBER_OF",
    "affiliated_with": "AFFILIATED_WITH", "involves": "INVOLVES", "leads": "LEADS",
    "instance_of": "INSTANCE_OF", "possible_alias": "POSSIBLE_ALIAS",
}

# Inverse/passive forms: canonical relation + the edge direction must be
# swapped by the caller (canon_relation returns flipped=True).
_RELATION_FLIP = {
    "offered_by": "OFFERS", "provided_by": "OFFERS", "issued_by": "OFFERS",
    "operated_by": "OFFERS", "developed_by": "OFFERS",
    "posted_by": "HIRING", "job_posted_by": "HIRING",
    "role_at": "HIRING", "is_role_at": "HIRING",
    "led_by": "LEADS", "managed_by": "LEADS", "run_by": "LEADS",
    "owned_by": "OWNS", "founded_by": "FOUNDED", "co_founded_by": "FOUNDED",
    "child_of": "PARENT_OF",
    "employs": "WORKS_FOR", "employer_of": "WORKS_FOR",
    "has_cto": "LEADS", "has_ceo": "LEADS",
}

# Ordered stem fallback (first match wins, order matters: location before
# OFFER, APPLI before JOB, COFOUNDER before FOUND). Matched against the
# normalized lower_snake string.
_RELATION_STEMS = (
    ("locat|based|headquarter|address", "LOCATED_IN"),
    ("appl[iy]", "APPLIED_TO"),
    ("hiring|vacanc|job|position|recruit", "HIRING"),
    ("post", "POSTED_ON"),
    ("cofounder|co_founder", "AFFILIATED_WITH"),
    ("work|employ|staff", "WORKS_FOR"),
    ("found", "FOUNDED"),
    ("contact|reached|email", "CONTACTED"),
    ("stud|attend|enroll|graduat", "ATTENDS"),
    ("own", "OWNS"),
    ("lead|ceo|director|manag", "LEADS"),
    ("partner", "PARTNER_OF"),
    ("offer|provid|sell", "OFFERS"),
    ("member", "MEMBER_OF"),
    ("travel|visit", "TRAVELS_TO"),
    ("use|deploy", "USES"),
    ("interest", "INTERESTED_IN"),
    ("sent|receiv|notif|updat|date|deadline|via|salary", "RELATED_TO"),
)
_STEM_RES = [(re.compile(pat), canon) for pat, canon in _RELATION_STEMS]


def canon_relation(rel) -> tuple:
    """Map any relation string into the closed vocabulary.

    Returns (canonical, flipped). flipped=True means the relation was a
    passive/inverse form and the CALLER must swap the edge's src/dst (this
    function only sees the string). Unknown relations no longer pass through
    (that was the vocabulary-sprawl engine), they fall to RELATED_TO; the
    original string survives in edges.rel_orig for observability.
    """
    if not rel:
        return "RELATED_TO", False
    r = re.sub(r"[^a-z0-9]+", "_", str(rel).strip().lower()).strip("_")
    if r in _RELATION_CANON:
        return _RELATION_CANON[r], False
    if r in _RELATION_FLIP:
        return _RELATION_FLIP[r], True
    # Generic passive marker: a trailing _by means subject/object are inverted
    # ("X IS_DEVELOPED_AND_OFFERED_BY Y" == "Y OFFERS X").
    passive = r.endswith("_by")
    for cre, canon in _STEM_RES:
        if cre.search(r):
            return canon, passive
    return "RELATED_TO", False


def normalize_relation(rel) -> str:
    """Back-compat string form of canon_relation (cannot express direction
    flips, callers that own (src, dst) should use canon_relation)."""
    return canon_relation(rel)[0]


# ---------------------------------------------------------------------------
# CLI (for verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--normalize":
        for a in args[1:]:
            print(f"{a!r} -> {normalize(a)!r}")
    elif args and args[0] == "--canon":
        # --canon "Name" [domain=] [email=] [phone=] [type=]
        name = args[1]
        kw = dict(kv.split("=", 1) for kv in args[2:] if "=" in kv)
        ht = kw.pop("type", None)
        print(canonical(name, hint_type=ht, **kw))
    else:
        print("usage: entity_resolve.py --normalize <name>... | --canon <name> [domain=] [email=] [phone=] [type=]")
