"""Offline ingestion tests — FAKE mode, tiny fixtures, no network, no real DB.

A small in-memory `GraphStore` interprets exactly the MERGE query shapes that
graph.merge_relation and ingestion.project emit, so node/edge creation and
idempotency can be asserted without a live Neo4j. vectors.embed_one is stubbed.
"""

import re
from pathlib import Path

import pytest

from app.config import EMBED_DIM

FIXTURES = Path(__file__).parent / "fixtures"

# Query-shape detectors (match the literal Cypher built in app/graph + project).
_EDGE_RE = re.compile(r"MERGE \(s\)-\[r:(\w+)\]->\(t\)")
_SRC_RE = re.compile(r"MERGE \(s:(\w+) \{uid: \$suid\}\)")
_TGT_RE = re.compile(r"MERGE \(t:(\w+) \{uid: \$tuid\}\)")
_NODE_RE = re.compile(r"MERGE \((\w+):(\w+) \{uid: \$uid\}\)")
# SET assignments: map the Cypher PROPERTY name (not the param name) to a value.
_SET_PARAM_RE = re.compile(r"\.(\w+) = \$(\w+)")
_SET_STR_RE = re.compile(r"\.(\w+) = '([^']*)'")
_SET_BOOL_RE = re.compile(r"\.(\w+) = (true|false)\b")


def _node_props(query: str, params: dict) -> dict:
    """Resolve a node's properties from its SET clause (property name -> value).

    Handles `prop = $param` (property name often differs from the param name,
    e.g. manuscript_id <- $mid, number <- $n) and baked constants/booleans.
    """
    props: dict = {}
    for m in _SET_PARAM_RE.finditer(query):
        prop, param = m.group(1), m.group(2)
        if param in params:
            props[prop] = params[param]
    for m in _SET_STR_RE.finditer(query):
        props[m.group(1)] = m.group(2)
    for m in _SET_BOOL_RE.finditer(query):
        props[m.group(1)] = m.group(2) == "true"
    return props


class GraphStore:
    """In-memory graph: nodes by uid, edges by (source_uid, type, target_uid)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}

    def _upsert_node(self, uid: str, label: str, props: dict, on_create: bool) -> None:
        clean = {k: v for k, v in props.items() if v is not None}
        if uid not in self.nodes:
            self.nodes[uid] = {"label": label, "props": dict(clean)}
        elif not on_create:
            self.nodes[uid]["props"].update(clean)

    def apply(self, query: str, params: dict) -> None:
        edge = _EDGE_RE.search(query)
        if edge:
            rtype = edge.group(1)
            s_label = _SRC_RE.search(query).group(1)
            t_label = _TGT_RE.search(query).group(1)
            suid, tuid = params["suid"], params["tuid"]
            self._upsert_node(
                suid,
                s_label,
                {"name": params.get("sname"), "manuscript_id": params.get("mid")},
                on_create=True,
            )
            self._upsert_node(
                tuid,
                t_label,
                {"name": params.get("tname"), "manuscript_id": params.get("mid")},
                on_create=True,
            )
            key = (suid, rtype, tuid)
            if key not in self.edges:  # ON CREATE SET r += $bundle
                self.edges[key] = dict(params.get("bundle") or {})
            return
        node = _NODE_RE.search(query)
        if node:
            label = node.group(2)
            self._upsert_node(
                params["uid"], label, _node_props(query, params), on_create=False
            )
            return
        # Reads / other statements are irrelevant to projection assertions.

    def count_label(self, label: str) -> int:
        return sum(1 for n in self.nodes.values() if n["label"] == label)


class _FakeResult:
    def __init__(self, records: list) -> None:
        self.records = records

    def single(self):
        return self.records[0] if self.records else None


class _FakeTx:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def run(self, query: str, **params):
        # commit_chapter reads max(sequence_index) first; emulate an empty graph
        # (no prior events) so the write path proceeds deterministically.
        if "max_seq" in query:
            return _FakeResult([{"max_seq": -1}])
        self.store.apply(query, params)
        return _FakeResult([])


class _FakeSession:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, *args, **kwargs):
        return fn(_FakeTx(self.store), *args, **kwargs)


class _FakeDriver:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def session(self):
        return _FakeSession(self.store)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> GraphStore:
    s = GraphStore()
    monkeypatch.setattr("app.graph.init_driver", lambda: _FakeDriver(s))
    monkeypatch.setattr("app.vectors.embed_one", lambda text: [0.01] * EMBED_DIM)
    return s


def _project_pdnc(mid_override: str | None = None) -> str:
    from ingestion.pdnc import PDNCLoader
    from ingestion.project import project_to_graph

    loader = PDNCLoader(FIXTURES / "pdnc" / "TinyNovel")
    project_to_graph(loader, mid_override)
    return mid_override or loader.namespace()


def _project_litbank(mid_override: str | None = None) -> str:
    from ingestion.litbank import LitBankLoader
    from ingestion.project import project_to_graph

    loader = LitBankLoader(FIXTURES / "litbank", "tinydoc")
    project_to_graph(loader, mid_override)
    return mid_override or loader.namespace()


class TestPDNC:
    def test_quotation_maps_to_event_with_involves_and_alias_resolution(
        self, store: GraphStore
    ) -> None:
        mid = _project_pdnc()
        ev = f"{mid}:Event:quote:q1"
        assert ev in store.nodes
        assert store.nodes[ev]["props"]["quote_type"] == "explicit"

        # speaker 'Lizzy' resolved to canonical 'Elizabeth Bennet' (agent)
        speaker = f"{mid}:Character:Elizabeth Bennet"
        assert (ev, "INVOLVES", speaker) in store.edges
        assert store.edges[(ev, "INVOLVES", speaker)]["role"] == "agent"

        # addressee 'Mr. Darcy' resolved to canonical 'Fitzwilliam Darcy' (patient)
        addressee = f"{mid}:Character:Fitzwilliam Darcy"
        assert (ev, "INVOLVES", addressee) in store.edges
        assert store.edges[(ev, "INVOLVES", addressee)]["role"] == "patient"

    def test_identified_as_alias_edges(self, store: GraphStore) -> None:
        mid = _project_pdnc()
        char = f"{mid}:Character:Elizabeth Bennet"
        for alias in ("Lizzy", "Eliza"):
            alias_uid = f"{mid}:Alias:{alias}"
            assert store.nodes[alias_uid]["label"] == "Alias"
            assert (char, "IDENTIFIED_AS", alias_uid) in store.edges
            assert store.edges[(char, "IDENTIFIED_AS", alias_uid)]["kind"] == "alias"

    def test_eval_namespacing_and_source_tag(self, store: GraphStore) -> None:
        mid = _project_pdnc()
        assert mid == "pdnc:TinyNovel"
        manuscript = store.nodes[f"{mid}:Manuscript:{mid}"]
        assert manuscript["props"]["source"] == "eval"
        assert manuscript["props"]["manuscript_id"].startswith("pdnc:")

    def test_idempotent_reprojection(self, store: GraphStore) -> None:
        _project_pdnc()
        n_nodes, n_edges = len(store.nodes), len(store.edges)
        _project_pdnc()  # re-run: MERGE on uid must not duplicate
        assert len(store.nodes) == n_nodes
        assert len(store.edges) == n_edges


class TestLitBank:
    def test_entity_type_mapping(self, store: GraphStore) -> None:
        mid = _project_litbank()
        assert store.nodes[f"{mid}:Character:Elizabeth"]["label"] == "Character"
        assert store.nodes[f"{mid}:Location:London"]["label"] == "Location"
        assert store.nodes[f"{mid}:Character:Mr. Darcy"]["label"] == "Character"
        assert (
            store.nodes[f"{mid}:Organization:Bennet family"]["label"] == "Organization"
        )

    def test_event_involves_with_property_bundle(self, store: GraphStore) -> None:
        mid = _project_litbank()
        event = f"{mid}:Event:0"
        elizabeth = f"{mid}:Character:Elizabeth"
        key = (event, "INVOLVES", elizabeth)
        assert key in store.edges
        bundle = store.edges[key]
        assert bundle["source"] == "eval"
        assert bundle["seq_introduced"] == 0
        assert "source_ref" in bundle
        assert bundle["role"] == "patient"

    def test_passages_written_as_committed_chapters(self, store: GraphStore) -> None:
        """Eval passages are Chapter nodes with embeddings -> reuse passage_index."""
        mid = _project_litbank()
        chapter = store.nodes[f"{mid}:Chapter:0"]["props"]
        assert chapter["status"] == "COMMITTED"
        assert chapter["source"] == "eval"
        assert len(chapter["embedding"]) == EMBED_DIM


class TestSchemaEdges:
    def test_has_trait_shares_one_attribute_node(self, store: GraphStore) -> None:
        """Two characters with the same trait point at ONE Attribute node."""
        from app.graph import merge_relation
        from app.models import ExtractedRelation, RelationType

        mid = "m_demo"
        tx = _FakeTx(store)
        for character in ("Mara", "Sera"):
            merge_relation(
                tx,
                mid,
                ExtractedRelation(
                    type=RelationType.HAS_TRAIT,
                    source=character,
                    target="brave",
                    category="personality",
                ),
                source_ref="trait",
                seq_introduced=0,
                source="eval",
            )

        attribute = f"{mid}:Attribute:brave"
        assert store.nodes[attribute]["label"] == "Attribute"
        assert store.count_label("Attribute") == 1  # shared, not duplicated
        assert (f"{mid}:Character:Mara", "HAS_TRAIT", attribute) in store.edges
        assert (f"{mid}:Character:Sera", "HAS_TRAIT", attribute) in store.edges
        assert (
            store.edges[(f"{mid}:Character:Mara", "HAS_TRAIT", attribute)]["category"]
            == "personality"
        )

    def test_structured_trait_node_keyed_by_dimension_not_value(
        self, store: GraphStore
    ) -> None:
        """A trait_key dimension is ONE Attribute node even across different
        values — the value rides on the edge, so the Attribute isn't fragmented."""
        from app.graph import merge_relation
        from app.models import ExtractedRelation, RelationType

        mid = "m_demo"
        tx = _FakeTx(store)
        for character, value in (("Mara", "blue"), ("Sera", "green")):
            merge_relation(
                tx,
                mid,
                ExtractedRelation(
                    type=RelationType.HAS_TRAIT,
                    source=character,
                    target=value,
                    category="physical",
                    trait_key="eye_color",
                    trait_value=value,
                ),
                source_ref="ch1",
                seq_introduced=1,
                source="gen",
            )

        dimension = f"{mid}:Attribute:eye_color"
        assert store.nodes[dimension]["label"] == "Attribute"
        assert store.count_label("Attribute") == 1  # one dimension node, not two
        mara_edge = store.edges[(f"{mid}:Character:Mara", "HAS_TRAIT", dimension)]
        sera_edge = store.edges[(f"{mid}:Character:Sera", "HAS_TRAIT", dimension)]
        assert mara_edge["trait_value"] == "blue"
        assert sera_edge["trait_value"] == "green"

    def test_property_bundle_drops_nulls_keeps_active_seq_invalidated(
        self, store: GraphStore
    ) -> None:
        """seq_invalidated=None (currently active) is absent, not stored as null."""
        from app.graph import merge_relation
        from app.models import ExtractedRelation, RelationType

        mid = "m_demo"
        tx = _FakeTx(store)
        merge_relation(
            tx,
            mid,
            ExtractedRelation(
                type=RelationType.CONTROLS, source="Mara", target="The Ledger"
            ),
            source_ref="ref1",
            seq_introduced=5,
            source="eval",
            target_label="Object",
        )
        edge = store.edges[
            (f"{mid}:Character:Mara", "CONTROLS", f"{mid}:Object:The Ledger")
        ]
        assert edge["seq_introduced"] == 5
        assert edge["source"] == "eval"
        assert "seq_invalidated" not in edge  # None dropped == active

    def test_unknown_label_rejected(self, store: GraphStore) -> None:
        from app.graph import merge_relation
        from app.models import ExtractedRelation, RelationType

        with pytest.raises(ValueError):
            merge_relation(
                _FakeTx(store),
                "m_demo",
                ExtractedRelation(
                    type=RelationType.CONTROLS, source="Mara", target="X"
                ),
                source_ref="r",
                seq_introduced=0,
                target_label="NotARealLabel",
            )


class TestEvalGuardUnit:
    def test_is_eval_manuscript_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _D:
            def execute_query(self, query, **params):
                return _FakeResult([{"is_eval": True}])

        monkeypatch.setattr("app.graph.init_driver", lambda: _D())
        from app import graph

        assert graph.is_eval_manuscript("pdnc:X") is True

    def test_is_eval_manuscript_false_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _D:
            def execute_query(self, query, **params):
                return _FakeResult([])

        monkeypatch.setattr("app.graph.init_driver", lambda: _D())
        from app import graph

        assert graph.is_eval_manuscript("nonexistent") is False


class TestChunkText:
    def test_packs_paragraphs_in_order(self) -> None:
        from ingestion.base import chunk_text

        segs = chunk_text("Alpha para.\n\nBeta para.\n\nGamma para.", max_chars=14)
        assert [s.index for s in segs] == list(range(len(segs)))
        joined = " ".join(s.text for s in segs)
        for word in ("Alpha", "Beta", "Gamma"):
            assert word in joined


class TestSpeaksToEdge:
    def test_speaks_to_in_allowlist_and_merges_character_to_character(
        self, store: GraphStore
    ) -> None:
        from app.graph import merge_relation
        from app.models import ExtractedRelation, RelationType

        merge_relation(
            _FakeTx(store),
            "m_d",
            ExtractedRelation(
                type=RelationType.SPEAKS_TO,
                source="Mara",
                target="Sera",
                quote_type="anaphoric",
            ),
            source_ref="q1",
            seq_introduced=1,
            source="gen",
        )
        key = ("m_d:Character:Mara", "SPEAKS_TO", "m_d:Character:Sera")
        assert key in store.edges
        assert store.edges[key]["quote_type"] == "anaphoric"
        assert store.edges[key]["source"] == "gen"
        # thin direct edge: both endpoints are Character
        assert store.nodes["m_d:Character:Mara"]["label"] == "Character"
        assert store.nodes["m_d:Character:Sera"]["label"] == "Character"


class TestCommitWritesRelations:
    def test_commit_writes_extraction_relations_source_gen(
        self, store: GraphStore
    ) -> None:
        from app.graph import commit_chapter
        from app.models import (
            ChapterExtraction,
            ExtractedCharacter,
            ExtractedRelation,
            RelationType,
        )

        extraction = ChapterExtraction(
            characters=[
                ExtractedCharacter(name="Mara", status="alive", note="x"),
                ExtractedCharacter(name="Sera", status="alive", note="x"),
            ],
            events=[],
            relations=[
                ExtractedRelation(
                    type=RelationType.SPEAKS_TO,
                    source="Mara",
                    target="Sera",
                    quote_type="explicit",
                ),
                ExtractedRelation(
                    type=RelationType.HAS_TRAIT,
                    source="Mara",
                    target="tall",
                    category="physical",
                    trait_key="height",
                    trait_value="tall",
                ),
            ],
        )
        # `store` fixture monkeypatches init_driver + embed_one.
        commit_chapter("m_gen", 3, "prose text", extraction, None)

        speaks = ("m_gen:Character:Mara", "SPEAKS_TO", "m_gen:Character:Sera")
        assert speaks in store.edges
        assert store.edges[speaks]["source"] == "gen"
        assert store.edges[speaks]["seq_introduced"] == 3

        # HAS_TRAIT Attribute node is keyed by the DIMENSION (trait_key='height'),
        # not the value, so values don't fragment it. Value rides on the edge.
        trait = ("m_gen:Character:Mara", "HAS_TRAIT", "m_gen:Attribute:height")
        assert trait in store.edges
        assert store.nodes["m_gen:Attribute:height"]["label"] == "Attribute"
        assert store.edges[trait]["trait_key"] == "height"
        assert store.edges[trait]["trait_value"] == "tall"

    def test_commit_with_no_relations_is_unchanged(self, store: GraphStore) -> None:
        """Backward compat: an extraction with no relations commits with 0 edges."""
        from app.graph import commit_chapter
        from app.models import ChapterExtraction, ExtractedCharacter

        extraction = ChapterExtraction(
            characters=[ExtractedCharacter(name="Mara", status="alive", note="x")],
            events=[],
        )
        commit_chapter("m_norel", 1, "text", extraction, None)
        assert store.edges == {}
        assert "m_norel:Character:Mara" in store.nodes


class TestCanonSurfacesRelations:
    def test_get_active_relations_shape_and_active_filter(self) -> None:
        from app.graph import _get_active_relations

        seen_queries: list[str] = []

        class _RelDriver:
            def execute_query(self, query, **params):
                seen_queries.append(query)
                table = {
                    "LOCATED_AT": [{"character": "Mara", "location": "Docks"}],
                    "RELATES_TO": [
                        {"source": "Mara", "target": "Brann", "stance": "enemy"}
                    ],
                    "FAMILY_OF": [{"source": "A", "target": "B", "subtype": "parent"}],
                    "POSSESSES": [{"character": "Mara", "object": "Key"}],
                    "IDENTIFIED_AS": [{"character": "Mara", "alias": "M"}],
                    "HAS_TRAIT": [
                        {
                            "character": "Mara",
                            "trait": "blue",
                            "category": "physical",
                            "trait_key": "eye_color",
                            "trait_value": "blue",
                        }
                    ],
                }
                for rel, recs in table.items():
                    if rel in query:
                        return _FakeResult(recs)
                return _FakeResult([])

        rels = _get_active_relations(_RelDriver(), "m_x")

        assert rels["located_at"] == [{"character": "Mara", "location": "Docks"}]
        assert rels["relates_to"][0]["stance"] == "enemy"
        assert rels["family_of"][0]["subtype"] == "parent"
        assert rels["possesses"][0]["object"] == "Key"
        assert rels["identified_as"][0]["alias"] == "M"
        assert rels["has_trait"][0]["trait_key"] == "eye_color"
        assert rels["has_trait"][0]["trait_value"] == "blue"
        # Every relation query must filter to ACTIVE edges only.
        assert all("seq_invalidated IS NULL" in q for q in seen_queries)


class TestPDNCRealFormat:
    """Regression tests for divergences found against the real PDNC data:
    the Aliases column uses Python *set* literals for some rows.
    """

    def test_parse_set_literal_aliases(self) -> None:
        from ingestion.pdnc import _parse_list_cell

        # Real PDNC: Charlotte Lucas aliases are a set literal.
        result = _parse_list_cell("{'Mrs. Collins', 'Miss Lucas', 'Charlotte'}")
        assert result == ["Charlotte", "Miss Lucas", "Mrs. Collins"]  # set -> sorted

    def test_parse_list_literal_aliases(self) -> None:
        from ingestion.pdnc import _parse_list_cell

        # Other real rows use list literals (order preserved).
        assert _parse_list_cell("['Colonel Forster']") == ["Colonel Forster"]
        assert _parse_list_cell("['Mary', 'Elizabeth']") == ["Mary", "Elizabeth"]

    def test_empty_and_plain_values(self) -> None:
        from ingestion.pdnc import _parse_list_cell

        assert _parse_list_cell("") == []
        assert _parse_list_cell(None) == []
        assert _parse_list_cell("Solo") == ["Solo"]


class TestLitBankRealFormat:
    """Regression test for the real LitBank '_brat' filename suffix."""

    def test_resolver_handles_brat_suffix(self, tmp_path) -> None:
        from ingestion.litbank import LitBankLoader

        ent = tmp_path / "entities" / "tsv"
        ev = tmp_path / "events" / "tsv"
        ent.mkdir(parents=True)
        ev.mkdir(parents=True)
        # Real files carry the _brat suffix.
        (ent / "99_demo_brat.tsv").write_text("Anna\tB-PER\nran\tO\n")
        (ev / "99_demo_brat.tsv").write_text("Anna\tO\nran\tEVENT\n")

        # Caller passes the clean stem; loader still finds the _brat file.
        loader = LitBankLoader(tmp_path, "99_demo")
        assert loader.namespace() == "litbank:99_demo"
        assert loader.entities_path.name == "99_demo_brat.tsv"
        ir = loader.load()
        assert any(e.name == "Anna" and e.type == "CHARACTER" for e in ir.entities)
        assert len(ir.events) == 1
