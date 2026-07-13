"""Strict Hirth / two-stroke knowledge graph ontology.

The seed graph is intentionally small: it gives sparse uploaded corpora a
coherent diagnostic backbone without pretending that document-derived facts
have been extracted.
"""
from __future__ import annotations

from typing import Any

NODE_TYPES = {
    "engine",
    "system",
    "part",
    "symptom",
    "cause",
    "fix",
    "spec",
    "source",
    "procedure",
    "test",
    "warning",
    "unknown",
}

EDGE_TYPES = {
    "HAS_SYSTEM",
    "HAS_PART",
    "HAS_SYMPTOM",
    "CAUSED_BY",
    "FIXED_BY",
    "HAS_SPEC",
    "MENTIONED_IN",
    "EVIDENCED_BY",
    "REQUIRES_PROCEDURE",
    "TESTED_BY",
    "RELATED_TO",
}

SEED_PROPS = {
    "extraction_method": "manual_seed",
    "confidence": 0.95,
    "evidence": "Built-in two-stroke diagnostic ontology",
    "doc_id": None,
    "source_chunk_id": None,
}

SYSTEM_PARTS: dict[str, list[str]] = {
    "Ignition System": [
        "Spark Plug",
        "Ignition Coil",
        "CDI / Ignition Module",
    ],
    "Fuel System": [
        "Carburetor",
        "Fuel Pump",
        "Fuel Filter",
        "Fuel Line",
    ],
    "Air Intake System": [
        "Reed Valve",
        "Air Filter",
    ],
    "Exhaust System": [
        "Exhaust Manifold",
        "Muffler",
    ],
    "Crankcase": [
        "Crankshaft",
        "Crankcase Seal",
    ],
    "Cylinder / Piston Assembly": [
        "Piston",
        "Piston Ring",
        "Cylinder Head",
    ],
    "Lubrication System": [
        "Oil Injection Pump",
    ],
    "Cooling System": [
        "Cooling Fan",
        "Temperature Sensor",
    ],
    "Certification / Testing": [],
    "Maintenance": [],
}

SYMPTOMS = [
    "Won't Start",
    "Hard Starting",
    "Misfire",
    "Rough Idle",
    "Loss of Power",
    "Overheating",
    "Excessive Smoke",
    "Detonation / Knocking",
    "High EGT",
    "Fuel Starvation",
    "Poor Compression",
]

CAUSE_FIXES: dict[str, list[str]] = {
    "Spark Plug Fouling": ["Clean Spark Plug", "Replace Spark Plug"],
    "Weak Spark": ["Check Ignition"],
    "Incorrect Mixture": ["Adjust Mixture"],
    "Fuel Restriction": ["Replace Fuel Filter", "Inspect Fuel Lines"],
    "Air Leak": ["Inspect Fuel Lines"],
    "Blocked Jet": ["Clean Carburetor"],
    "Low Compression": ["Check Compression"],
    "Overheating": ["Inspect Cooling System"],
    "Incorrect Timing": ["Check Ignition"],
    "Carbon Buildup": ["Decarbonize Exhaust"],
}

SYMPTOM_CAUSES: dict[str, list[str]] = {
    "Won't Start": ["Weak Spark", "Fuel Restriction", "Low Compression"],
    "Hard Starting": ["Spark Plug Fouling", "Incorrect Mixture", "Low Compression"],
    "Misfire": ["Weak Spark", "Spark Plug Fouling", "Incorrect Timing"],
    "Rough Idle": ["Incorrect Mixture", "Blocked Jet", "Air Leak"],
    "Loss of Power": ["Fuel Restriction", "Low Compression", "Carbon Buildup"],
    "Overheating": ["Incorrect Mixture", "Carbon Buildup"],
    "Excessive Smoke": ["Incorrect Mixture"],
    "Detonation / Knocking": ["Incorrect Timing", "Overheating"],
    "High EGT": ["Incorrect Mixture", "Fuel Restriction"],
    "Fuel Starvation": ["Fuel Restriction", "Blocked Jet"],
    "Poor Compression": ["Low Compression"],
}


def seed_graph() -> dict[str, list[dict[str, Any]]]:
    """Return seed nodes and edges in graph_builder-compatible shape."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[tuple[str, str]] = set()

    def add_node(ntype: str, name: str) -> None:
        key = (ntype, name)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        nodes.append({"type": ntype, "name": name, "props": dict(SEED_PROPS)})

    node_types_by_name: dict[str, str] = {}

    def add_edge(src: str, dst: str, relation: str, src_type: str | None = None, dst_type: str | None = None) -> None:
        edges.append({
            "src": src,
            "dst": dst,
            "src_type": src_type or node_types_by_name.get(src),
            "dst_type": dst_type or node_types_by_name.get(dst),
            "relation": relation,
            "props": dict(SEED_PROPS),
        })

    add_node("engine", "Two-Stroke Engine")
    node_types_by_name["Two-Stroke Engine"] = "engine"

    for system, parts in SYSTEM_PARTS.items():
        add_node("system", system)
        node_types_by_name[system] = "system"
        add_edge("Two-Stroke Engine", system, "HAS_SYSTEM", "engine", "system")
        for part in parts:
            add_node("part", part)
            node_types_by_name[part] = "part"
            add_edge(system, part, "HAS_PART", "system", "part")

    for symptom, causes in SYMPTOM_CAUSES.items():
        add_node("symptom", symptom)
        add_edge("Two-Stroke Engine", symptom, "HAS_SYMPTOM", "engine", "symptom")
        for cause in causes:
            add_node("cause", cause)
            add_edge(symptom, cause, "CAUSED_BY", "symptom", "cause")

    for cause, fixes in CAUSE_FIXES.items():
        add_node("cause", cause)
        for fix in fixes:
            add_node("fix", fix)
            add_edge(cause, fix, "FIXED_BY", "cause", "fix")

    add_node("procedure", "Maintenance Inspection")
    add_edge("Maintenance", "Maintenance Inspection", "REQUIRES_PROCEDURE", "system", "procedure")
    add_node("test", "Endurance Test")
    add_edge("Certification / Testing", "Endurance Test", "TESTED_BY", "system", "test")

    return {"nodes": nodes, "edges": edges}


def normalize_node_type(value: str | None) -> str:
    ntype = (value or "unknown").strip().lower()
    return ntype if ntype in NODE_TYPES else "unknown"


def normalize_relation(value: str | None) -> str:
    relation = (value or "RELATED_TO").strip().upper().replace(" ", "_")
    return relation if relation in EDGE_TYPES else "RELATED_TO"


def is_valid_seed_graph(graph: dict[str, list[dict[str, Any]]]) -> bool:
    """Small sanity helper used by tests and ad hoc scripts."""
    node_names = {str(n.get("name", "")).strip() for n in graph.get("nodes", [])}
    if not node_names:
        return False
    for node in graph.get("nodes", []):
        if normalize_node_type(str(node.get("type", ""))) not in NODE_TYPES:
            return False
        if not str(node.get("name", "")).strip():
            return False
    for edge in graph.get("edges", []):
        if str(edge.get("src", "")).strip() not in node_names:
            return False
        if str(edge.get("dst", "")).strip() not in node_names:
            return False
        if normalize_relation(str(edge.get("relation", ""))) not in EDGE_TYPES:
            return False
    return True
