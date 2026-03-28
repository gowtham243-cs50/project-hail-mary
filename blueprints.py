"""
Blueprint Ingestion — DXF Parser + Embedding
=============================================
Parses a DXF file, extracts room/area information,
and embeds each room description using OpenRouterBGEEmbeddingFunction.

Output: list of dicts, each with room metadata + its embedding vector.

Usage:
    python blueprint_ingest.py demo_rig.dxf
    python blueprint_ingest.py          # auto-generates a demo DXF
"""

import ezdxf
import json
import math
import sys
from bgme import OpenRouterBGEEmbeddingFunction


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tune these to match your actual DXF layer names
# ─────────────────────────────────────────────────────────────────────────────

ROOM_LABEL_LAYERS = {"ROOM_NAMES", "ROOMS", "TEXT", "A-AREA-IDEN", "LABELS"}

AREA_KEYWORDS = {
    "drilling": "Drilling Area",
    "rig base": "Rig Base",
    "control":  "Control Room",
    "pump":     "Pump Room",
    "storage":  "Storage Area",
    "office":   "Office Area",
    "medical":  "Medical Bay",
    "generator":"Generator Room",
    "mine":     "Mine Area",
    "shaft":    "Shaft Area",
    "workshop": "Workshop",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PARSE DXF
# ─────────────────────────────────────────────────────────────────────────────

def classify_area(label: str) -> str:
    lower = label.lower()
    for keyword, area in AREA_KEYWORDS.items():
        if keyword in lower:
            return area
    return "General Area"


def parse_dxf(filepath: str) -> list[dict]:
    """
    Parse a DXF file and return a list of room records.

    Each record:
        {
            "label":       "Room 304 Drilling Area",
            "area":        "Drilling Area",
            "x":           100.0,
            "y":           40.0,
            "description": "Room 304 Drilling Area, located in Drilling Area, coordinates (100.0, 40.0)"
        }
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    raw_texts = []
    for entity in msp:
        if entity.dxftype() not in ("TEXT", "MTEXT"):
            continue
        try:
            layer = entity.dxf.layer.upper()
            if entity.dxftype() == "TEXT":
                text = entity.dxf.text.strip()
                x, y = entity.dxf.insert.x, entity.dxf.insert.y
            else:
                text = entity.plain_mtext().strip()
                x, y = entity.dxf.insert.x, entity.dxf.insert.y

            if not text:
                continue

            # Keep if it's on a known label layer OR looks like a room name
            on_label_layer  = layer in ROOM_LABEL_LAYERS
            looks_like_room = any(kw in text.lower() for kw in
                                  ["room", "area", "bay", "lab", "office",
                                   "shaft", "pump", "store", "drill",
                                   "control", "workshop"])
            if on_label_layer or looks_like_room:
                raw_texts.append({"text": text, "x": x, "y": y})

        except Exception:
            continue

    # De-duplicate entries that are < 2 units apart (repeated annotations)
    seen, unique = [], []
    for t in raw_texts:
        if not any(math.dist((t["x"], t["y"]), (s["x"], s["y"])) < 2.0 for s in seen):
            unique.append(t)
            seen.append(t)

    rooms = []
    for t in unique:
        label = t["text"]
        area  = classify_area(label)
        rooms.append({
            "label":       label,
            "area":        area,
            "x":           round(t["x"], 2),
            "y":           round(t["y"], 2),
            "description": f"{label}, located in {area}, coordinates ({t['x']:.1f}, {t['y']:.1f})",
        })

    print(f"[parse] Extracted {len(rooms)} rooms from '{filepath}'")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENRICH DESCRIPTIONS WITH SPATIAL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

ADJACENCY_THRESHOLD = 50.0   # units — tune to match your DXF scale

DIRECTION_MAP = [
    # (angle_min, angle_max, label)
    (-22.5,   22.5,  "east"),
    ( 22.5,   67.5,  "northeast"),
    ( 67.5,  112.5,  "north"),
    (112.5,  157.5,  "northwest"),
    (157.5,  180.0,  "west"),
    (-180.0, -157.5, "west"),
    (-157.5, -112.5, "southwest"),
    (-112.5,  -67.5, "south"),
    (-67.5,   -22.5, "southeast"),
]

def _cardinal(dx: float, dy: float) -> str:
    """Return a cardinal/intercardinal direction label from a delta vector."""
    angle = math.degrees(math.atan2(dy, dx))
    for lo, hi, label in DIRECTION_MAP:
        if lo <= angle <= hi:
            return label
    return "nearby"


def enrich_descriptions(rooms: list[dict],
                        threshold: float = ADJACENCY_THRESHOLD) -> list[dict]:
    """
    Replace each room's plain 'description' with a natural language sentence
    that includes:
      - what zone/area it belongs to
      - which rooms are adjacent (within threshold) and in which direction

    This makes the embedding spatially aware so queries like
    "rooms near the pump room" or "what's between rig base and drilling area"
    retrieve meaningful results.
    """
    for room in rooms:
        neighbours = []
        for other in rooms:
            if other["label"] == room["label"]:
                continue
            dist = math.dist((room["x"], room["y"]), (other["x"], other["y"]))
            if dist <= threshold:
                direction = _cardinal(other["x"] - room["x"], other["y"] - room["y"])
                neighbours.append(f"{other['label']} ({direction})")

        adjacency = (
            "Adjacent rooms: " + ", ".join(neighbours) + "."
            if neighbours
            else "No rooms detected nearby."
        )

        room["description"] = (
            f"{room['label']} is located in the {room['area']} zone. {adjacency}"
        )

    print(f"[enrich] Descriptions enriched with spatial context for {len(rooms)} rooms")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EMBED
# ─────────────────────────────────────────────────────────────────────────────

def embed_rooms(rooms: list[dict]) -> list[dict]:
    """
    Embed each room's description using OpenRouterBGEEmbeddingFunction.
    Adds an 'embedding' key to each room dict and returns the list.
    """
    embedding_fn = OpenRouterBGEEmbeddingFunction()
    descriptions = [r["description"] for r in rooms]

    embeddings = embedding_fn(descriptions)   # list[list[float]]

    for room, vec in zip(rooms, embeddings):
        room["embedding"] = vec

    print(f"[embed] Embedded {len(rooms)} rooms")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DXF GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def create_demo_dxf(filepath: str = "demo_rig.dxf") -> str:
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    doc.layers.new("ROOMS", dxfattribs={"color": 2})

    room_data = [
        ("Main Room Rig Base",       0,   0),
        ("Control Room Rig Base",    0,  40),
        ("Storage Room Rig Base",   40,   0),
        ("Room 304 Drilling Area", 100,  40),
        ("Room 305 Drilling Area", 100,   0),
        ("Pump Room Drilling Area",  60,  40),
        ("Office 101",               0,  80),
        ("Medical Bay",             40,  80),
        ("Generator Room",         100,  80),
    ]
    for label, x, y in room_data:
        msp.add_text(label, dxfattribs={"layer": "ROOMS", "insert": (x, y), "height": 2.5})

    doc.saveas(filepath)
    print(f"[demo] Synthetic DXF saved to '{filepath}'")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dxf_path = sys.argv[1] if len(sys.argv) > 1 else create_demo_dxf()

    # 1. Parse
    rooms = parse_dxf(dxf_path)

    # 2. Enrich descriptions with spatial context
    rooms = enrich_descriptions(rooms)

    # 3. Embed
    rooms = embed_rooms(rooms)

    # 4. Hand off — rooms is now a list of dicts ready for your vector store.
    print("\n[result] Sample output (first room):")
    preview = {k: v for k, v in rooms[0].items() if k != "embedding"}
    preview["embedding"] = f"[{rooms[0]['embedding'][0]:.4f}, ... ] (dim={len(rooms[0]['embedding'])})"
    print(json.dumps(preview, indent=2))

    # To pass to your vector store, just use `rooms` directly:
    # vectorstore.add(rooms)