#!/usr/bin/env python3
"""
EXPLA Graph Chat Server
- Lädt den Knowledge Graph in den RAM
- Bietet REST-Endpoints für Graph-Abfragen
- Integriert Claude API mit Tool-Use für natürlichsprachliche Fragen
- Serviert das HTML-Frontend

Start: python3 graph_server.py
Dann: http://localhost:8000
"""

import json, gzip, re, os
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import uvicorn

# ── Graph laden ──────────────────────────────────────────────────────────────
# Suche: 1) ENV, 2) graph.json.gz im gleichen Ordner, 3) Lokaler Pfad
DEFAULT_LOCAL = str(Path(__file__).parent.parent / "BPW_Wissensgraph_Daten" / "Silver_Lake" / "nodes" / "silver_lake_merged.json")
DEFAULT_GZ = str(Path(__file__).parent / "graph.json.gz")
GRAPH_PATH = os.environ.get("GRAPH_PATH", DEFAULT_GZ if Path(DEFAULT_GZ).exists() else DEFAULT_LOCAL)

print(f"Lade Graph: {GRAPH_PATH}")
if GRAPH_PATH.endswith(".gz"):
    with gzip.open(GRAPH_PATH, "rt", encoding="utf-8") as f:
        G = json.load(f)
else:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        G = json.load(f)

TOTAL_N = sum(len(v) for v in G["nodes"].values())
TOTAL_E = sum(len(v) for v in G["edges"].values())
print(f"Graph geladen: {TOTAL_N} Knoten, {TOTAL_E} Kanten, Version {G.get('version','?')}")

# ── Indizes aufbauen ─────────────────────────────────────────────────────────
def normalize(pn):
    return re.sub(r'[\s\-\.\,]', '', str(pn)).upper()

# TN → Bauteil
TN_INDEX = {}
for bt in G["nodes"].get("Bauteil", []):
    tn = bt.get("teilenummer", "")
    if tn:
        TN_INDEX[normalize(tn)] = bt

# ID → Node (alle Typen)
ID_INDEX = {}
for typ, nodes in G["nodes"].items():
    for n in nodes:
        ID_INDEX[n["id"]] = {**n, "_typ": typ}

# Edges by von/nach
EDGES_BY_VON = defaultdict(list)
EDGES_BY_NACH = defaultdict(list)
for typ, edges in G["edges"].items():
    for e in edges:
        e_with_type = {**e, "_edge_typ": typ}
        EDGES_BY_VON[e.get("von", "")].append(e_with_type)
        EDGES_BY_NACH[e.get("nach", "")].append(e_with_type)

print(f"Indizes: {len(TN_INDEX)} TN, {len(ID_INDEX)} Knoten")

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Running Gear Knowledge Graph", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Graph-Abfrage-Funktionen ────────────────────────────────────────────────

def search_bauteile(query: str, hersteller: str = None, baugruppe: str = None, limit: int = 20):
    """Sucht Bauteile nach Teilenummer, Name oder Typ."""
    q = query.lower()
    q_norm = normalize(query)
    results = []

    # Exakter TN-Match
    if q_norm in TN_INDEX:
        results.append(TN_INDEX[q_norm])

    # Baugruppen-Match (z.B. "Bremsscheibe" → BG_BREMSSCHEIBE)
    bg_match = "BG_" + query.upper().replace(" ", "_").replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE")
    bg_names = {bg["id"]: bg.get("name", "") for bg in G["nodes"].get("Baugruppe", [])}

    # Suche in allen Bauteilen
    for bt in G["nodes"].get("Bauteil", []):
        if len(results) >= limit:
            break
        if bt in results:
            continue
        if hersteller and hersteller.lower() not in str(bt.get("hersteller", "")).lower():
            continue
        if baugruppe and bt.get("baugruppe_id") != baugruppe:
            continue
        name = str(bt.get("name", "")).lower()
        tn = str(bt.get("teilenummer", "")).lower()
        typ = str(bt.get("bauteil_typ", "")).lower()
        bg_id = str(bt.get("baugruppe_id", "")).lower()
        if q in name or q in tn or q in typ or q in bg_id or q_norm == normalize(bt.get("teilenummer", "")):
            results.append(bt)

    return [{"id": b["id"], "name": b.get("name", "Unbekannt"), "teilenummer": b.get("teilenummer", "—"),
             "hersteller": b.get("hersteller", "Unbekannt"), "bauteil_typ": b.get("bauteil_typ"),
             "baugruppe_id": b.get("baugruppe_id"),
             "beschreibung": b.get("beschreibung_einfach", ""),
             "konfidenz": b.get("konfidenz")} for b in results[:limit]]


def get_cross_references(teilenummer: str, hersteller_filter: str = None):
    """Findet Cross-References (kompatible Teile) für eine Teilenummer."""
    tn_norm = normalize(teilenummer)
    bt = TN_INDEX.get(tn_norm)
    if not bt:
        return {"error": f"Teilenummer {teilenummer} nicht gefunden", "results": []}

    results = []
    # M05 edges von und nach diesem Bauteil
    for e in EDGES_BY_VON.get(bt["id"], []):
        if e["_edge_typ"] == "M05_kompatibel_mit":
            other = ID_INDEX.get(e["nach"])
            if other and (not hersteller_filter or hersteller_filter.lower() in str(other.get("hersteller", "")).lower()):
                results.append({"id": other["id"], "teilenummer": other.get("teilenummer"),
                                "hersteller": other.get("hersteller"), "name": other.get("name"),
                                "konfidenz": e.get("konfidenz"), "quelle": e.get("quelle")})

    for e in EDGES_BY_NACH.get(bt["id"], []):
        if e["_edge_typ"] == "M05_kompatibel_mit":
            other = ID_INDEX.get(e["von"])
            if other and other["id"] != bt["id"] and (not hersteller_filter or hersteller_filter.lower() in str(other.get("hersteller", "")).lower()):
                results.append({"id": other["id"], "teilenummer": other.get("teilenummer"),
                                "hersteller": other.get("hersteller"), "name": other.get("name"),
                                "konfidenz": e.get("konfidenz"), "quelle": e.get("quelle")})

    return {"bauteil": {"id": bt["id"], "teilenummer": bt.get("teilenummer"),
                        "hersteller": bt.get("hersteller"), "name": bt.get("name")},
            "cross_references": results, "count": len(results)}


def get_zulieferer(kategorie: str = None):
    """Listet Zulieferer, optional gefiltert nach Kategorie."""
    results = []
    for org in G["nodes"].get("Organisation", []):
        tier = org.get("tier", "")
        if tier in ("T1", "T2"):
            rolle = org.get("rolle", [])
            if isinstance(rolle, str):
                rolle = [rolle]
            if kategorie:
                if not any(kategorie.lower() in r.lower() for r in rolle):
                    continue
            results.append({"id": org["id"], "name": org.get("name"), "sitz": org.get("sitz"),
                            "tier": tier, "rolle": rolle})
    return results


def get_mitbewerber():
    """Listet alle Mitbewerber."""
    results = []
    for org in G["nodes"].get("Organisation", []):
        tier = org.get("tier", "")
        rolle = str(org.get("rolle", ""))
        if tier == "Mitbewerber" or "Mitbewerber" in rolle:
            results.append({"id": org["id"], "name": org.get("name"), "sitz": org.get("sitz"),
                            "tier": tier, "rolle": org.get("rolle")})
    return results


def get_reklamationen(hersteller: str = None):
    """Listet Reklamationen, optional gefiltert nach Hersteller."""
    results = []
    for rek in G["nodes"].get("Reklamation", []):
        if hersteller:
            if hersteller.lower() not in str(rek.get("betroffene_hersteller", "")).lower() and \
               hersteller.lower() not in str(rek.get("beschreibung_einfach", "")).lower():
                continue
        results.append({"id": rek["id"], "name": rek.get("name"),
                        "beschreibung": rek.get("beschreibung_einfach"),
                        "hersteller": rek.get("betroffene_hersteller"),
                        "fahrzeuge": rek.get("fahrzeuge_betroffen"),
                        "jahr": rek.get("jahr")})
    return results


def get_dokumente_fuer_bauteil(teilenummer: str):
    """Findet alle Kataloge/Dokumente, in denen eine Teilenummer dokumentiert ist."""
    tn_norm = normalize(teilenummer)
    bt = TN_INDEX.get(tn_norm)
    if not bt:
        return {"error": f"Teilenummer {teilenummer} nicht gefunden", "results": []}

    docs = []
    for e in EDGES_BY_VON.get(bt["id"], []):
        if e["_edge_typ"] == "E12_dokumentiert_in":
            dok = ID_INDEX.get(e["nach"])
            if dok:
                docs.append({"id": dok["id"], "name": dok.get("name"),
                             "seite": e.get("seite"), "hersteller": dok.get("hersteller")})
    return {"bauteil": bt.get("teilenummer"), "dokumente": docs, "count": len(docs)}


def get_baugruppe_bauteile(baugruppe_id: str):
    """Listet alle Bauteile einer Baugruppe."""
    results = []
    for bt in G["nodes"].get("Bauteil", []):
        if bt.get("baugruppe_id") == baugruppe_id:
            results.append({"id": bt["id"], "teilenummer": bt.get("teilenummer"),
                            "hersteller": bt.get("hersteller"), "name": bt.get("name"),
                            "bauteil_typ": bt.get("bauteil_typ")})
    return {"baugruppe_id": baugruppe_id, "bauteile": results[:50], "count": len(results)}


def get_normen(thema: str = None):
    """Listet Normen, optional gefiltert nach Thema."""
    results = []
    for norm in G["nodes"].get("Norm", []):
        if thema:
            text = " ".join([
                str(norm.get("name", "")),
                str(norm.get("beschreibung_fachlich", "")),
                str(norm.get("beschreibung_einfach", "")),
                str(norm.get("id", "")),
            ])
            if thema.lower() not in text.lower():
                continue
        results.append({"id": norm["id"], "name": norm.get("name"),
                        "beschreibung": norm.get("beschreibung_einfach", norm.get("beschreibung_fachlich", ""))})
    return results


def get_trailer_modelle(achshersteller: str = None):
    """Listet Trailer-Modelle, optional gefiltert nach Achshersteller."""
    results = []
    for tm in G["nodes"].get("TrailerModell", []):
        if achshersteller:
            text = str(tm.get("beschreibung_fachlich", ""))
            # Check E01 edges
            has_match = achshersteller.lower() in text.lower()
            if not has_match:
                for e in EDGES_BY_NACH.get(tm["id"], []):
                    if e["_edge_typ"] == "E01_verbaut_in":
                        bt = ID_INDEX.get(e["von"])
                        if bt and achshersteller.lower() in str(bt.get("hersteller", "")).lower():
                            has_match = True
                            break
            if not has_match:
                continue
        results.append({"id": tm["id"], "name": tm.get("name"),
                        "beschreibung": tm.get("beschreibung_einfach", "")})
    return results


def get_graph_stats():
    """Liefert Statistiken zum Graphen."""
    # Baugruppen mit Beschreibung (nur Hauptbaugruppen = oberbaugruppe_id ist None oder BG_FAHRWERK)
    baugruppen = []
    for bg in G["nodes"].get("Baugruppe", []):
        baugruppen.append({
            "id": bg.get("id"),
            "name": bg.get("name"),
            "beschreibung": bg.get("beschreibung_einfach", ""),
            "oberbaugruppe_id": bg.get("oberbaugruppe_id"),
        })
    return {
        "version": G.get("version"),
        "knoten_gesamt": TOTAL_N,
        "kanten_gesamt": TOTAL_E,
        "knoten_nach_typ": {k: len(v) for k, v in G["nodes"].items()},
        "kanten_nach_typ": {k: len(v) for k, v in G["edges"].items()},
        "baugruppen": baugruppen,
    }


# ── Tool-Definitionen für Claude ─────────────────────────────────────────────

TOOLS = [
    {"name": "search_bauteile", "description": "Sucht Bauteile im Graph nach Teilenummer, Name, Typ oder Hersteller. Nutze dies für Fragen wie 'Welche Bremsscheiben gibt es?' oder 'Finde Teil 09.801.06.25.0'.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Suchbegriff (Teilenummer, Name, Typ)"},
         "hersteller": {"type": "string", "description": "Optional: Nach Hersteller filtern (z.B. 'BPW', 'SAF')"},
         "baugruppe": {"type": "string", "description": "Optional: Nach Baugruppe filtern (z.B. 'BG_BREMSSCHEIBE')"},
     }, "required": ["query"]}},

    {"name": "get_cross_references", "description": "Findet kompatible/alternative Teile anderer Hersteller für eine Teilenummer. Nutze dies für Cross-Reference-Fragen.",
     "input_schema": {"type": "object", "properties": {
         "teilenummer": {"type": "string", "description": "Die Teilenummer, für die Alternativen gesucht werden"},
         "hersteller_filter": {"type": "string", "description": "Optional: Nur Alternativen eines bestimmten Herstellers"},
     }, "required": ["teilenummer"]}},

    {"name": "get_zulieferer", "description": "Listet alle Zulieferer (Tier 1 und Tier 2), optional nach Kategorie gefiltert.",
     "input_schema": {"type": "object", "properties": {
         "kategorie": {"type": "string", "description": "Optional: Kategorie (z.B. 'Bremsen', 'Lager', 'Elektronik')"},
     }}},

    {"name": "get_mitbewerber", "description": "Listet alle Mitbewerber/Wettbewerber im Markt.",
     "input_schema": {"type": "object", "properties": {}}},

    {"name": "get_reklamationen", "description": "Listet KBA-Rückrufe und Reklamationen, optional nach Hersteller gefiltert.",
     "input_schema": {"type": "object", "properties": {
         "hersteller": {"type": "string", "description": "Optional: Nach Hersteller filtern (z.B. 'Krone', 'BPW')"},
     }}},

    {"name": "get_dokumente_fuer_bauteil", "description": "Findet alle Kataloge/Dokumente, in denen eine Teilenummer dokumentiert ist.",
     "input_schema": {"type": "object", "properties": {
         "teilenummer": {"type": "string", "description": "Die Teilenummer"},
     }, "required": ["teilenummer"]}},

    {"name": "get_baugruppe_bauteile", "description": "Listet alle Bauteile einer Baugruppe (z.B. BG_BREMSSCHEIBE, BG_EBS, BG_RADLAGER).",
     "input_schema": {"type": "object", "properties": {
         "baugruppe_id": {"type": "string", "description": "Baugruppen-ID (z.B. 'BG_BREMSSCHEIBE', 'BG_EBS', 'BG_RADLAGER', 'BG_LUFTFEDERUNG')"},
     }, "required": ["baugruppe_id"]}},

    {"name": "get_normen", "description": "Listet technische Normen, optional nach Thema gefiltert.",
     "input_schema": {"type": "object", "properties": {
         "thema": {"type": "string", "description": "Optional: Thema (z.B. 'Bremse', 'Beleuchtung')"},
     }}},

    {"name": "get_trailer_modelle", "description": "Listet Trailer-Modelle, optional gefiltert nach Achshersteller.",
     "input_schema": {"type": "object", "properties": {
         "achshersteller": {"type": "string", "description": "Optional: Achshersteller (z.B. 'BPW', 'SAF')"},
     }}},

    {"name": "get_graph_stats", "description": "Liefert Statistiken zum Knowledge Graph (Anzahl Knoten, Kanten, Versionen).",
     "input_schema": {"type": "object", "properties": {}}},
]

TOOL_FUNCTIONS = {
    "search_bauteile": search_bauteile,
    "get_cross_references": get_cross_references,
    "get_zulieferer": get_zulieferer,
    "get_mitbewerber": get_mitbewerber,
    "get_reklamationen": get_reklamationen,
    "get_dokumente_fuer_bauteil": get_dokumente_fuer_bauteil,
    "get_baugruppe_bauteile": get_baugruppe_bauteile,
    "get_normen": get_normen,
    "get_trailer_modelle": get_trailer_modelle,
    "get_graph_stats": get_graph_stats,
}

# ── Smart Local Query Engine (kein API Key nötig) ───────────────────────────

# TN-Pattern: erkennt Teilenummern wie 05.397.28.01.0, AS-0098-01F, 235 521 0001
TN_PATTERN = re.compile(r'\b\d{2}[\.\-]\d{3}[\.\-]\d{2}[\.\-]\d{2}[\.\-]\d\b|'
                        r'\b[A-Z]{2,3}[\-\s]?\d{3,6}[\-\s]?\d{0,4}[A-Z]?\b|'
                        r'\b\d{3}\s?\d{3}\s?\d{4}\b')

# Synonyme für Intent-Erkennung
INTENT_KEYWORDS = {
    "cross_ref": ["cross-ref", "crossref", "cross ref", "alternative", "alternativ", "kompatib", "ersatz", "vergleich", "statt", "anstelle", "tausch"],
    "zulieferer": ["zulieferer", "lieferant", "liefert", "supplier", "tier 1", "tier 2", "tier-1", "tier-2", "wer liefert", "wer stellt her"],
    "mitbewerber": ["mitbewerber", "wettbewerb", "konkurrenz", "competitor", "rivale"],
    "reklamation": ["reklamation", "rückruf", "kba", "recall", "mangel", "defekt", "rückrufaktion", "beanstandung"],
    "normen": ["norm", "standard", "vorschrift", "ece", "un r", "din", "iso", "fmvss", "regelung", "richtlinie"],
    "trailer": ["trailer", "auflieger", "anhänger", "sattelauflieger", "modell"],
    "stats": ["statistik", "übersicht", "zusammenfassung", "wie viele", "wieviele", "anzahl", "graph", "gesamt", "überblick"],
    "dokument": ["dokument", "katalog", "kataloge", "handbuch", "dokumentiert", "in welchem"],
    "baugruppe": ["baugruppe", "komponente", "system", "alle teile von", "teile der"],
    "search": ["suche", "finde", "zeig", "welche", "gibt es", "list"],
}

# Baugruppen-Synonyme
BG_SYNONYMS = {
    "bremsscheibe": "BG_BREMSSCHEIBE", "bremsscheiben": "BG_BREMSSCHEIBE", "brake disc": "BG_BREMSSCHEIBE",
    "bremsbelag": "BG_BREMSBELAG", "bremsbeläge": "BG_BREMSBELAG", "brake pad": "BG_BREMSBELAG",
    "bremse": "BG_BREMSE", "bremsen": "BG_BREMSE", "brake": "BG_BREMSE",
    "fahrwerk": "BG_FAHRWERK", "suspension": "BG_FAHRWERK",
    "luftfeder": "BG_LUFTFEDERUNG", "luftfederung": "BG_LUFTFEDERUNG", "air spring": "BG_LUFTFEDERUNG",
    "radlager": "BG_RADLAGER", "wheel bearing": "BG_RADLAGER", "lager": "BG_RADLAGER",
    "achse": "BG_ACHSE", "achsen": "BG_ACHSE", "axle": "BG_ACHSE",
    "sattelkupplung": "BG_SATTELKUPPLUNG", "fifth wheel": "BG_SATTELKUPPLUNG",
    "stützwinde": "BG_STUETZWINDE", "stützwinden": "BG_STUETZWINDE", "landing gear": "BG_STUETZWINDE",
    "nachsteller": "BG_NACHSTELLER", "slack adjuster": "BG_NACHSTELLER",
    "ebs": "BG_EBS", "abs": "BG_EBS", "elektronisch": "BG_EBS",
    "beleuchtung": "BG_BELEUCHTUNG", "licht": "BG_BELEUCHTUNG", "lampe": "BG_BELEUCHTUNG",
    "lenkung": "BG_LENKUNG", "steering": "BG_LENKUNG",
}

# Hersteller-Erkennung
KNOWN_HERSTELLER = {o.get("name", "").lower(): o for o in G["nodes"].get("Organisation", [])}


def detect_intent(msg: str):
    """Erkennt den Intent und extrahiert Parameter aus einer natürlichsprachlichen Frage."""
    msg_lower = msg.lower()

    # 1. Teilenummer erkennen
    tn_match = TN_PATTERN.search(msg)
    teilenummer = tn_match.group(0) if tn_match else None

    # 2. Intent erkennen
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[intent] = score

    # Cross-Ref hat Vorrang wenn TN vorhanden + cross-ref Kontext
    if teilenummer and "cross_ref" not in scores:
        if not scores:
            scores["cross_ref"] = 1

    # Bei Gleichstand: spezifischere Intents bevorzugen
    # "Wie viele Zulieferer" → zulieferer > stats
    priority_order = ["cross_ref", "reklamation", "normen", "zulieferer", "mitbewerber", "trailer", "dokument", "baugruppe", "search", "stats"]
    if scores:
        max_score = max(scores.values())
        candidates = [k for k, v in scores.items() if v == max_score]
        if len(candidates) > 1:
            for p in priority_order:
                if p in candidates:
                    primary_intent = p
                    break
            else:
                primary_intent = candidates[0]
        else:
            primary_intent = candidates[0]
    else:
        primary_intent = "search" if teilenummer else "search"

    # 3. Hersteller erkennen (kurzer Name für Suche, damit z.B. "Krone" statt voller Orgname)
    hersteller = None
    hersteller_short = None  # Kurzname für flexible Suche
    SHORT_NAMES = {
        "bpw": "BPW", "saf": "SAF", "krone": "Krone", "schmitz": "Schmitz",
        "kögel": "Kögel", "koegel": "Kögel", "daf": "DAF", "man": "MAN",
        "volvo": "Volvo", "scania": "Scania", "mercedes": "Mercedes",
        "iveco": "Iveco", "knorr": "Knorr", "wabco": "WABCO",
        "haldex": "Haldex", "meyle": "MEYLE", "jost": "JOST",
        "hendrickson": "Hendrickson", "gigant": "Gigant", "continental": "Continental",
        "bosch": "Bosch", "skf": "SKF", "schaeffler": "Schaeffler",
        "timken": "Timken", "al-ko": "AL-KO", "alko": "AL-KO",
    }
    for short, display in SHORT_NAMES.items():
        if short in msg_lower:
            hersteller_short = display
            # Finde vollen Orgnamen
            for o in G["nodes"].get("Organisation", []):
                if short in o.get("name", "").lower():
                    hersteller = o.get("name")
                    break
            if not hersteller:
                hersteller = display
            break

    # 4. Baugruppe erkennen
    baugruppe = None
    for syn, bg_id in BG_SYNONYMS.items():
        if syn in msg_lower:
            baugruppe = bg_id
            break

    # 5. Thema für Normen
    thema = None
    if primary_intent == "normen":
        for word in ["brems", "licht", "beleuchtung", "achse", "luft", "federung", "lenkung", "ebs", "abs", "reifen"]:
            if word in msg_lower:
                thema = word
                break

    return {
        "intent": primary_intent,
        "teilenummer": teilenummer,
        "hersteller": hersteller,
        "hersteller_short": hersteller_short,
        "baugruppe": baugruppe,
        "thema": thema,
        "query": msg,
    }


def format_bauteile_response(results, query_info):
    """Formatiert Bauteil-Suchergebnisse als lesbare Antwort."""
    if not results:
        return f"Keine Bauteile gefunden für \"{query_info.get('query', '')}\"."

    lines = [f"**{len(results)} Bauteile gefunden:**\n"]
    for i, bt in enumerate(results[:15], 1):
        tn = bt.get("teilenummer", "—")
        name = bt.get("name", "")
        herst = bt.get("hersteller", "")
        typ = bt.get("bauteil_typ", "")
        conf = bt.get("konfidenz")
        line = f"{i}. **{tn}**"
        if herst:
            line += f" ({herst})"
        if name and name != tn:
            line += f" — {name}"
        if typ:
            line += f" [{typ}]"
        if conf:
            line += f" | Konfidenz: {conf}"
        lines.append(line)

    if len(results) > 15:
        lines.append(f"\n... und {len(results) - 15} weitere.")
    return "\n".join(lines)


def format_crossref_response(result):
    """Formatiert Cross-Reference-Ergebnisse."""
    if "error" in result and result["error"]:
        return f"Teilenummer nicht gefunden: {result['error']}"

    bt = result.get("bauteil", {})
    refs = result.get("cross_references", [])

    lines = [f"**Cross-References für {bt.get('teilenummer', '?')}** ({bt.get('hersteller', '?')}, {bt.get('name', '')})\n"]

    if not refs:
        lines.append("Keine kompatiblen Alternativen im Graph gefunden.")
    else:
        lines.append(f"**{len(refs)} kompatible Alternativen:**\n")
        for i, ref in enumerate(refs, 1):
            line = f"{i}. **{ref.get('teilenummer', '?')}** ({ref.get('hersteller', '?')})"
            if ref.get("name"):
                line += f" — {ref['name']}"
            if ref.get("konfidenz"):
                line += f" | Konfidenz: {ref['konfidenz']}"
            if ref.get("quelle"):
                line += f" | Quelle: {ref['quelle']}"
            lines.append(line)

    return "\n".join(lines)


def format_zulieferer_response(results, kategorie=None):
    """Formatiert Zulieferer-Liste."""
    if not results:
        return f"Keine Zulieferer gefunden{' für Kategorie ' + kategorie if kategorie else ''}."

    title = f"**{len(results)} Zulieferer"
    if kategorie:
        title += f" (Kategorie: {kategorie})"
    title += ":**\n"

    lines = [title]
    for i, z in enumerate(results, 1):
        rolle = z.get("rolle", [])
        if isinstance(rolle, list):
            rolle = ", ".join(rolle)
        line = f"{i}. **{z.get('name', '?')}** — Tier: {z.get('tier', '?')}"
        if z.get("sitz"):
            line += f", Sitz: {z['sitz']}"
        if rolle:
            line += f" | Rollen: {rolle}"
        lines.append(line)
    return "\n".join(lines)


def format_mitbewerber_response(results):
    """Formatiert Mitbewerber-Liste."""
    if not results:
        return "Keine Mitbewerber im Graph gefunden."

    lines = [f"**{len(results)} Mitbewerber:**\n"]
    for i, m in enumerate(results, 1):
        line = f"{i}. **{m.get('name', '?')}**"
        if m.get("sitz"):
            line += f" — {m['sitz']}"
        lines.append(line)
    return "\n".join(lines)


def format_reklamationen_response(results, hersteller=None):
    """Formatiert Reklamations-Liste."""
    if not results:
        return f"Keine Reklamationen/KBA-Rückrufe gefunden{' für ' + hersteller if hersteller else ''}."

    title = f"**{len(results)} Reklamationen/KBA-Rückrufe"
    if hersteller:
        title += f" ({hersteller})"
    title += ":**\n"

    lines = [title]
    for i, r in enumerate(results, 1):
        line = f"{i}. **{r.get('name', r.get('id', '?'))}**"
        if r.get("jahr"):
            line += f" ({r['jahr']})"
        if r.get("beschreibung"):
            line += f"\n   {r['beschreibung']}"
        if r.get("hersteller"):
            line += f"\n   Hersteller: {r['hersteller']}"
        if r.get("fahrzeuge"):
            line += f" | Fahrzeuge: {r['fahrzeuge']}"
        lines.append(line)
    return "\n".join(lines)


def format_normen_response(results, thema=None):
    """Formatiert Normen-Liste."""
    if not results:
        return f"Keine Normen gefunden{' zum Thema ' + thema if thema else ''}."

    lines = [f"**{len(results)} Normen{' zum Thema ' + thema.title() if thema else ''}:**\n"]
    for i, n in enumerate(results[:20], 1):
        line = f"{i}. **{n.get('name', '?')}**"
        if n.get("beschreibung"):
            line += f" — {n['beschreibung'][:120]}"
        lines.append(line)
    return "\n".join(lines)


def format_trailer_response(results, achshersteller=None):
    """Formatiert Trailer-Modell-Liste."""
    if not results:
        return f"Keine Trailer-Modelle gefunden{' mit ' + achshersteller + '-Achsen' if achshersteller else ''}."

    lines = [f"**{len(results)} Trailer-Modelle{' mit ' + achshersteller + '-Achsen' if achshersteller else ''}:**\n"]
    for i, t in enumerate(results, 1):
        line = f"{i}. **{t.get('name', '?')}**"
        if t.get("beschreibung"):
            line += f" — {t['beschreibung'][:100]}"
        lines.append(line)
    return "\n".join(lines)


def format_stats_response(stats):
    """Formatiert Graph-Statistiken."""
    lines = [
        f"**EXPLA Knowledge Graph — Übersicht (v{stats.get('version', '?')})**\n",
        f"**{stats['knoten_gesamt']:,}** Knoten und **{stats['kanten_gesamt']:,}** Kanten\n",
        "**Knoten nach Typ:**",
    ]
    for typ, count in sorted(stats["knoten_nach_typ"].items(), key=lambda x: -x[1]):
        lines.append(f"  • {typ}: {count:,}")

    lines.append("\n**Kanten nach Typ:**")
    for typ, count in sorted(stats["kanten_nach_typ"].items(), key=lambda x: -x[1]):
        lines.append(f"  • {typ}: {count:,}")

    lines.append(f"\n**Datenquellen:** 52 geparste Ersatzteilkataloge von 30+ Herstellern")
    return "\n".join(lines)


def format_dokumente_response(result):
    """Formatiert Dokument-Ergebnisse."""
    if "error" in result:
        return result["error"]

    docs = result.get("dokumente", [])
    tn = result.get("bauteil", "?")

    if not docs:
        return f"Keine Dokumente/Kataloge gefunden für Teilenummer {tn}."

    lines = [f"**{len(docs)} Dokumente für {tn}:**\n"]
    for i, d in enumerate(docs, 1):
        line = f"{i}. **{d.get('name', '?')}**"
        if d.get("hersteller"):
            line += f" ({d['hersteller']})"
        if d.get("seite"):
            line += f" — Seite {d['seite']}"
        lines.append(line)
    return "\n".join(lines)


def smart_answer(msg: str, history: list = None):
    """Beantwortet eine Frage rein aus dem Graph — ohne API Key."""
    msg_lower = msg.lower().strip()

    # ── Konversations-Kontext prüfen ──
    # Kurze Nachfragen, die sich auf den Gesprächsverlauf beziehen
    CONVERSATIONAL_PATTERNS = [
        "was war", "was hast du", "erinnerst du", "weißt du noch", "merk dir", "merke dir",
        "nochmal", "wiederhole", "und jetzt", "genau", "danke", "ok", "ja",
        "nein", "cool", "super", "hallo", "hi", "hey", "tschüss", "bye",
        "kannst du", "wer bist du", "was kannst du", "hilfe", "help",
        "vergiss nicht", "sage", "sag mir",
    ]
    is_conversational = any(p in msg_lower for p in CONVERSATIONAL_PATTERNS)

    # Auch kurze Nachrichten ohne klaren Graph-Bezug als konversational behandeln
    if len(msg_lower.split()) <= 4 and not re.search(r'\d{2}[\.\-]', msg):
        # Kurze Nachricht ohne Teilenummer → konversational prüfen
        graph_words = {"bauteil", "teil", "norm", "zulieferer", "hersteller", "bpw", "saf",
                       "bremsscheibe", "cross", "reklamation", "rückruf", "trailer", "achse",
                       "knorr", "wabco", "krone", "baugruppe", "katalog", "graph"}
        has_graph_word = any(gw in msg_lower for gw in graph_words)
        if not has_graph_word:
            is_conversational = True

    if is_conversational:
        # ── "Merke dir" — funktioniert auch ohne History ──
        if any(p in msg_lower for p in ["merk dir", "merke dir", "vergiss nicht"]):
            num_match = re.search(r'\b(\d+)\b', msg)
            if num_match:
                return (
                    f"Alles klar, ich merke mir die Zahl **{num_match.group(1)}**! Frag mich einfach danach. Kann ich dir sonst mit dem Knowledge Graph helfen?",
                    ["conversation_memory"]
                )
            return (
                f"Alright, ich merke es mir! Frag mich einfach danach. Kann ich dir sonst mit dem Knowledge Graph helfen?",
                ["conversation_memory"]
            )

        # ── Rückfragen, die History brauchen ──
        if history and len(history) >= 2:
            last_bot = None
            for h in reversed(history):
                if h.get("role") == "assistant":
                    last_bot = h.get("content", "")
                    break

            if any(p in msg_lower for p in ["was war", "erinnerst", "weißt du noch", "nochmal", "wiederhole"]):
                # Prüfe ob nach einer Zahl gefragt wird
                if any(w in msg_lower for w in ["zahl", "nummer", "number", "wert"]):
                    for h in reversed(history):
                        if h.get("role") == "user":
                            h_text = h.get("content", "")
                            h_lower = h_text.lower()
                            has_memory_word = any(w in h_lower for w in ["merk", "sag", "zahl", "nummer"])
                            if has_memory_word:
                                num_match = re.search(r'\b(\d+)\b', h_text)
                                if num_match:
                                    return (
                                        f"Die Zahl war **{num_match.group(1)}**.",
                                        ["conversation_memory"]
                                    )
                        # Auch in Bot-Antworten suchen (z.B. "ich merke mir die Zahl 3")
                        if h.get("role") == "assistant":
                            h_text = h.get("content", "")
                            num_in_bot = re.search(r'merke mir.*?(\d+)', h_text, re.IGNORECASE)
                            if num_in_bot:
                                return (
                                    f"Die Zahl war **{num_in_bot.group(1)}**.",
                                    ["conversation_memory"]
                                )

                # Generelle Rückfrage → letzte Bot-Antwort zeigen
                if last_bot:
                    return (
                        f"Hier nochmal meine letzte Antwort:\n\n{last_bot[:500]}",
                        ["conversation_memory"]
                    )

        # Grüße & Smalltalk
        if any(p in msg_lower for p in ["hallo", "hi", "hey", "moin", "servus", "guten"]):
            return (
                "Hey! Ich bin APON, der Assistent für den Running Gear Knowledge Graph. Frag mich etwas über Bauteile, Cross-References, Normen oder Zulieferer — ich durchsuche den Graphen für dich.",
                []
            )

        if any(p in msg_lower for p in ["danke", "super", "cool", "top", "klasse", "perfekt"]):
            return (
                "Gerne! Hast du noch eine Frage zum Graphen?",
                []
            )

        if any(p in msg_lower for p in ["wer bist du", "was bist du", "was kannst du", "hilfe", "help"]):
            return (
                "Ich bin **APON** — der KI-Assistent für den EXPLA Running Gear Knowledge Graph.\n\n"
                "Ich kann dir helfen mit:\n"
                "• **Bauteile suchen** — z.B. \"Welche Bremsscheiben bietet BPW an?\"\n"
                "• **Cross-References** — z.B. \"Alternativen für 05.397.28.01.0\"\n"
                "• **Zulieferer** — z.B. \"Wer liefert Radlager an BPW?\"\n"
                "• **Normen** — z.B. \"Welche Normen gelten für Bremsen?\"\n"
                "• **Reklamationen** — z.B. \"KBA-Rückrufe für Krone\"\n"
                "• **Trailer-Modelle** — z.B. \"Welche Trailer nutzt Schmitz?\"\n\n"
                f"Der Graph enthält **{TOTAL_N:,} Knoten** und **{TOTAL_E:,} Beziehungen** aus 52 Katalogen.",
                []
            )

        # Generische Antwort für unklare Fragen
        return (
            f"Das kann ich leider nicht direkt beantworten — ich bin auf den Knowledge Graph spezialisiert.\n\n"
            f"Versuch es mit einer konkreten Frage, z.B.:\n"
            f"• \"Welche Alternativen gibt es für BPW 05.397.28.01.0?\"\n"
            f"• \"Wer liefert Radlager an BPW?\"\n"
            f"• \"Welche Normen gelten für Trailer-Bremsen?\"",
            []
        )

    info = detect_intent(msg)
    intent = info["intent"]
    tools_used = []

    if intent == "cross_ref" and info["teilenummer"]:
        # Bei Cross-Ref den Hersteller NICHT als Filter nutzen (wir wollen ja Alternativen anderer Hersteller!)
        result = get_cross_references(info["teilenummer"])
        tools_used.append("get_cross_references")
        answer = format_crossref_response(result)

    elif intent == "zulieferer":
        # Kategorie aus Baugruppe ableiten + verwandte Begriffe
        kat = None
        KATEGORIE_SYNONYME = {
            "radlager": "Lager", "lager": "Lager", "bearing": "Lager",
            "bremse": "Brems", "bremsscheibe": "Brems", "bremsbelag": "Brems", "brake": "Brems",
            "elektronik": "Elektr", "ebs": "Elektr", "abs": "Elektr",
            "feder": "Feder", "luftfeder": "Feder", "air spring": "Feder",
            "achse": "Achs", "axle": "Achs",
        }
        for syn in KATEGORIE_SYNONYME:
            if syn in msg.lower():
                kat = KATEGORIE_SYNONYME[syn]
                break
        if not kat:
            for syn, bg in BG_SYNONYMS.items():
                if syn in msg.lower():
                    kat = syn.title()
                    break
        result = get_zulieferer(kat)
        tools_used.append("get_zulieferer")
        # Wenn mit Kategorie nichts gefunden, nochmal ohne
        if not result and kat:
            result = get_zulieferer(None)
            answer = f"Keine Zulieferer mit Rolle \"{kat}\" gefunden. Hier alle {len(result)} Zulieferer:\n\n"
            answer += format_zulieferer_response(result)
        else:
            answer = format_zulieferer_response(result, kat)

    elif intent == "mitbewerber":
        result = get_mitbewerber()
        tools_used.append("get_mitbewerber")
        answer = format_mitbewerber_response(result)

    elif intent == "reklamation":
        # Kurznamen für Reklamation-Suche (z.B. "Krone" statt "Fahrzeugwerk Bernard Krone GmbH")
        herst_search = info.get("hersteller_short") or info.get("hersteller")
        result = get_reklamationen(herst_search)
        tools_used.append("get_reklamationen")
        answer = format_reklamationen_response(result, herst_search)

    elif intent == "normen":
        result = get_normen(info.get("thema"))
        tools_used.append("get_normen")
        answer = format_normen_response(result, info.get("thema"))

    elif intent == "trailer":
        herst_search = info.get("hersteller_short") or info.get("hersteller")
        result = get_trailer_modelle(herst_search)
        tools_used.append("get_trailer_modelle")
        answer = format_trailer_response(result, herst_search)

    elif intent == "stats":
        result = get_graph_stats()
        tools_used.append("get_graph_stats")
        answer = format_stats_response(result)

    elif intent == "dokument" and info["teilenummer"]:
        result = get_dokumente_fuer_bauteil(info["teilenummer"])
        tools_used.append("get_dokumente_fuer_bauteil")
        answer = format_dokumente_response(result)

    elif intent == "baugruppe" and info["baugruppe"]:
        result = get_baugruppe_bauteile(info["baugruppe"])
        tools_used.append("get_baugruppe_bauteile")
        bt_list = result.get("bauteile", [])
        if bt_list:
            answer = format_bauteile_response(
                [{"teilenummer": b.get("teilenummer"), "name": b.get("name"),
                  "hersteller": b.get("hersteller"), "bauteil_typ": b.get("bauteil_typ")}
                 for b in bt_list],
                info
            )
        else:
            answer = f"Keine Bauteile in Baugruppe {info['baugruppe']} gefunden."

    else:
        # Generische Suche — Schlüsselwörter extrahieren statt ganzen Satz
        if info["teilenummer"]:
            query = info["teilenummer"]
        elif info.get("baugruppe"):
            # Baugruppe erkannt → suche alle Bauteile dieser Baugruppe
            result_bg = get_baugruppe_bauteile(info["baugruppe"])
            tools_used.append("get_baugruppe_bauteile")
            bt_list = result_bg.get("bauteile", [])
            herst_short = info.get("hersteller_short") or info.get("hersteller")
            if herst_short:
                bt_list = [b for b in bt_list if herst_short.lower() in str(b.get("hersteller", "")).lower()]
            if bt_list:
                answer = format_bauteile_response(
                    [{"teilenummer": b.get("teilenummer"), "name": b.get("name"),
                      "hersteller": b.get("hersteller"), "bauteil_typ": b.get("bauteil_typ")}
                     for b in bt_list],
                    info
                )
            else:
                bg_name = info["baugruppe"].replace("BG_", "").replace("_", " ").title()
                extra = f" von {herst_short}" if herst_short else ""
                answer = f"Keine Bauteile in der Baugruppe **{bg_name}**{extra} gefunden."
            return answer, tools_used
        else:
            # Schlüsselwörter aus dem Satz extrahieren
            stop_words = {"welche", "was", "wie", "wer", "gibt", "es", "an", "für", "von", "der", "die", "das",
                          "bietet", "hat", "sind", "werden", "kann", "alle", "viele", "wieviele", "zeig", "zeige",
                          "finde", "suche", "mir", "bitte", "im", "in", "zum", "zur", "bei", "mit", "und", "oder"}
            words = [w for w in re.split(r'[\s\?\!\.\,]+', msg.lower()) if w and w not in stop_words and len(w) > 1]
            # Entferne bekannte Hersteller aus Suchwörtern
            herst_lower = (info.get("hersteller_short") or "").lower()
            words = [w for w in words if w != herst_lower]
            query = " ".join(words[:3]) if words else msg.strip()

        bg = info.get("baugruppe")
        herst_search = info.get("hersteller_short") or info.get("hersteller")
        result = search_bauteile(query, herst_search, bg)
        tools_used.append("search_bauteile")

        if result:
            answer = format_bauteile_response(result, info)
        else:
            # Fallback: Stats + Hilfe
            answer = (
                f"Ich konnte keine direkten Ergebnisse zu \"{msg}\" finden.\n\n"
                f"**Tipps:** Versuche es mit:\n"
                f"• Einer **Teilenummer** (z.B. 05.397.28.01.0)\n"
                f"• Einem **Bauteil-Typ** (z.B. Bremsscheibe, Radlager, EBS)\n"
                f"• **Zulieferer**, **Mitbewerber**, **Reklamationen**, **Normen**\n"
                f"• **Cross-References** für eine Teilenummer\n\n"
                f"Der Graph enthält {TOTAL_N:,} Knoten und {TOTAL_E:,} Kanten."
            )

    return answer, tools_used


# ── Chat Endpoint ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""Du bist APON — der persönliche KI-Assistent für den EXPLA Knowledge Graph.

Du hilfst Nutzern, Informationen über BPW Running Gear (Trailer-Achsen und Fahrwerkskomponenten) zu finden. Du bist freundlich, kompetent und antwortest auf Deutsch.

DEIN VERHALTEN:
- Führe natürliche Gespräche. Frag nach, wenn etwas unklar ist.
- Nutze die Graph-Tools um Fragen zu beantworten — aber nur wenn nötig.
- Bei einfachen Folgefragen oder Smalltalk brauchst Du keine Tools.
- Fasse Ergebnisse übersichtlich zusammen. Nicht einfach rohe Daten raushauen.
- Wenn Du mehrere Tools brauchst, kombiniere die Ergebnisse zu einer zusammenhängenden Antwort.
- Nenne Konfidenzwerte und Quellen wenn verfügbar, aber baue sie natürlich in den Text ein.

DER GRAPH:
- {TOTAL_N:,} Knoten, {TOTAL_E:,} Kanten, basierend auf 52 geparsten Ersatzteilkatalogen
- Bauteile mit Teilenummern, Cross-References (kompatible Teile anderer Hersteller)
- 67 Organisationen (Zulieferer, OEMs, Mitbewerber)
- KBA-Reklamationen/Rückrufe
- 91 technische Normen, 39 Trailer-Modelle
- Baugruppen: Bremsscheibe, Bremsbelag, Bremse, Fahrwerk, Luftfederung, Radlager, Achse, Sattelkupplung, Stützwinde, Nachsteller, EBS, Beleuchtung, Lenkung

TIPPS FÜR GUTE ANTWORTEN:
- Bei "welche Teile von X nutzen Y-Produkte": Suche Cross-References zwischen den Herstellern
- Bei Zulieferer-Fragen: get_zulieferer liefert alle Tier-1/Tier-2 mit Rollen
- Bei Vergleichsfragen: Kombiniere mehrere Tool-Aufrufe
- Wenn ein Tool keine Ergebnisse liefert, erkläre warum und schlage Alternativen vor"""


class ChatRequest(BaseModel):
    message: str
    api_key: str = ""
    history: list = []  # Konversationshistorie [{role, content}, ...]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # API Key: aus Request, oder aus Environment Variable
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Modus 1: Ohne API Key → Smart Local Engine ──
    if not api_key:
        answer, tools_used = smart_answer(req.message, req.history)
        return {"answer": answer, "tools_used": tools_used, "mode": "local"}

    # ── Modus 2: Mit API Key → Claude mit Konversationshistorie ──
    client = anthropic.Anthropic(api_key=api_key)

    # Konversationshistorie aufbauen (max. letzte 20 Nachrichten)
    messages = []
    for h in (req.history or [])[-20:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    # Aktuelle Nachricht anhängen
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(401, "Ungültiger API-Key")
    except Exception as e:
        raise HTTPException(500, str(e))

    # Tool-Use Loop (max 5 Runden)
    all_tools_used = []
    for _ in range(5):
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                all_tools_used.append(block.name)
                func = TOOL_FUNCTIONS.get(block.name)
                if func:
                    try:
                        result = func(**block.input)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

    # Antwort extrahieren
    answer = ""
    for block in response.content:
        if hasattr(block, "text"):
            answer += block.text

    return {"answer": answer, "tools_used": all_tools_used, "mode": "claude"}


# ── REST Endpoints (direkt) ──────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats():
    return get_graph_stats()

@app.get("/api/bauteile")
def api_bauteile(q: str, hersteller: str = None, baugruppe: str = None):
    return search_bauteile(q, hersteller, baugruppe)

@app.get("/api/cross-references/{teilenummer}")
def api_crossref(teilenummer: str, hersteller: str = None):
    return get_cross_references(teilenummer, hersteller)

@app.get("/api/zulieferer")
def api_zulieferer(kategorie: str = None):
    return get_zulieferer(kategorie)

@app.get("/api/reklamationen")
def api_reklamationen(hersteller: str = None):
    return get_reklamationen(hersteller)

# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Running Gear Knowledge Graph")
    print(f"   Graph: {TOTAL_N} Knoten, {TOTAL_E} Kanten")
    print(f"   URL:   http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
