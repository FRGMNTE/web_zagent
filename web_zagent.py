# web_zagent.py - ZeroAgent (überarbeitet): Persona + strukturierte Ausgabe (ImprovedQuery/Answer/Status),
# Wikipedia-Chunk-Deduplizierung, Programmiersprachensupport, Audit-Logging und Sync zu offline_zagent.
#
# Features:
# - Persona als System-Instruktion
# - LLM-Prompt erwartet JSON-Ausgabe: improved_query, answer, status, reason (opt.), suggestions (opt.)
# - Dedup beim Wikipedia-Import (sha256-Hash)
# - Programmiermodus / Sprache: UI-Option und automatischer Hinweis im Prompt
# - Audit-Log: strukturierte Einträge per save_qa
#
# Optional Abhängigkeiten (nur für bestimmte Features):
# pip install gradio requests wikipedia sentence-transformers chromadb numpy langchain_community

import os
import json
import hashlib
import datetime
import platform
import typing
import requests

import gradio as gr

# Optional modules
try:
    import wikipedia
    _HAS_WIKI = True
except Exception:
    wikipedia = None
    _HAS_WIKI = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENT = True
except Exception:
    SentenceTransformer = None
    _HAS_SENT = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMA = True
except Exception:
    chromadb = None
    _HAS_CHROMA = False

# Local offline agent
try:
    import offline_zagent
except Exception:
    offline_zagent = None

# Ollama wrapper (optional)
try:
    from langchain_community.llms import Ollama
    _HAS_OLLAMA = True
except Exception:
    Ollama = None
    _HAS_OLLAMA = False

# -------- Configuration --------
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
BASE_DIR = os.path.expanduser("~/zero_knowledge")
TOPIC_DIR = os.path.join(BASE_DIR, "topics")
IMPORTS_DIR = os.path.join(BASE_DIR, "imports")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.json")
os.makedirs(TOPIC_DIR, exist_ok=True)
os.makedirs(IMPORTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Persona (Deutsch)
PERSONA_TEXT = (
    "Rolle: Du bist ein unglaublich fähiger KI-Assistent.\n"
    "Verhalten: Du tust dein Bestes, um Anweisungen zu befolgen und löst Probleme eigenständig in Absprache. "
    "Du weigerst dich nur bei großen Risiken.\n"
    "Sicherheitsregeln: Du schützt vor schädlichen Inhalten und vermeidest Beleidigungen. "
    "Gebe keine gefährlichen, illegalen oder beleidigenden Inhalte aus."
)

# LLM init
llm = None
if _HAS_OLLAMA:
    try:
        llm = Ollama(model=DEFAULT_MODEL)
    except Exception as e:
        llm = None
        print("Warnung Ollama:", e)

# sentence-transformers small model (optional)
EMB_MODEL_NAME = os.environ.get("OFFLINE_EMB_MODEL", "all-MiniLM-L6-v2")
_emb_model = None
if _HAS_SENT:
    try:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    except Exception:
        _emb_model = None

# Chromadb client (optional)
_chroma_client = None
if _HAS_CHROMA:
    try:
        _chroma_client = chromadb.Client(ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    except Exception:
        _chroma_client = None

# -------- Helpers --------

def get_zeroagent_info():
    return {
        "model_name": DEFAULT_MODEL,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "file_path": os.path.abspath(__file__),
    }

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def read_json(path: str) -> typing.Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: str, data: typing.Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------- Audit / Q&A Storage (strukturierte Einträge) --------

def get_topic_path(topic: str) -> str:
    safe = "".join(c for c in topic if c.isalnum() or c in ("_", "-")).lower() or "default"
    return os.path.join(TOPIC_DIR, f"{safe}.json")

def save_qa(topic: str, question: str, result: dict) -> None:
    """
    result: strukturierte Antwort-Daten wie:
      {
        "time": iso,
        "improved_query": "...",
        "answer": "...",
        "status": "ok"/"refused",
        "reason": "...",
        "persona": PERSONA_TEXT,
        "source": "offline/chroma/online"
      }
    """
    path = get_topic_path(topic)
    try:
        data = read_json(path) or []
    except Exception:
        data = []
    entry = result.copy()
    entry["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    entry["question"] = question
    data.append(entry)
    try:
        write_json(path, data)
    except Exception as e:
        print("Fehler save_qa:", e)
    # also append to audit log
    try:
        audit = read_json(AUDIT_LOG) or []
        audit_entry = {
            "time": entry["time"],
            "topic": topic,
            "improved_query": entry.get("improved_query",""),
            "status": entry.get("status",""),
            "reason": entry.get("reason",""),
        }
        audit.append(audit_entry)
        write_json(AUDIT_LOG, audit)
    except Exception:
        pass

# -------- Wikipedia Fetch + Dedup + Export (chunks) --------

def fetch_wikipedia_articles(query: str, max_pages: int = 3) -> typing.List[dict]:
    if not _HAS_WIKI:
        return []
    try:
        titles = wikipedia.search(query, results=max_pages)
        results = []
        for t in titles:
            try:
                page = wikipedia.page(t, auto_suggest=False)
                results.append({"title": page.title, "url": page.url, "content": page.content})
            except Exception:
                continue
        return results
    except Exception:
        return []

def chunk_text(text: str, max_chars: int = 1600, overlap: int = 300) -> typing.List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + max_chars, L)
        chunk = text[i:end]
        if end < L:
            last_dot = chunk.rfind(".")
            if last_dot > int(max_chars * 0.4):
                chunk = chunk[:last_dot+1]
                end = i + len(chunk)
        chunks.append(chunk.strip())
        i = max(i + max_chars - overlap, end)
    return [c for c in chunks if c]

def existing_hashes_for_topic(topic: str) -> typing.Set[str]:
    # maintain per-topic index file of chunk hashes
    idx_fn = os.path.join(TOPIC_DIR, f"{topic}_hashes.json")
    existing = read_json(idx_fn) or []
    return set(existing)

def add_hashes_for_topic(topic: str, hashes: typing.List[str]) -> None:
    idx_fn = os.path.join(TOPIC_DIR, f"{topic}_hashes.json")
    existing = read_json(idx_fn) or []
    existing.extend(hashes)
    # deduplicate
    existing = list(dict.fromkeys(existing))
    write_json(idx_fn, existing)

def save_import_file(topic: str, chunks: typing.List[dict]) -> str:
    # perform dedup by hash and skip identical chunks
    kept = []
    hashes = existing_hashes_for_topic(topic)
    new_hashes = []
    for c in chunks:
        h = compute_hash(c.get("text",""))
        if h in hashes:
            continue
        c["_hash"] = h
        kept.append(c)
        new_hashes.append(h)
    if not kept:
        return "Keine neuen Chunks (alle duplikate)"
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    fn = os.path.join(IMPORTS_DIR, f"{topic}_chunks_{ts}.json")
    payload = {"topic": topic, "source": "wikipedia_sync", "created": datetime.datetime.now().isoformat(timespec="seconds"), "chunks": kept}
    write_json(fn, payload)
    # update hash index
    try:
        add_hashes_for_topic(topic, new_hashes)
    except Exception:
        pass
    return fn

def build_chunks_from_wikipedia(query: str, topic: str, max_pages: int = 3) -> typing.List[dict]:
    pages = fetch_wikipedia_articles(query, max_pages=max_pages)
    chunks = []
    for p in pages:
        parts = chunk_text(p["content"])
        for idx, part in enumerate(parts):
            chunks.append({"id": f"{p['title']}_{idx}", "title": p["title"], "source_url": p["url"], "text": part, "created": datetime.datetime.now().isoformat(timespec="seconds")})
    return chunks

def index_chunks_to_chroma(chunks: typing.List[dict], topic: str) -> bool:
    if _chroma_client is None:
        return False
    try:
        col_name = f"zero_{topic}"
        try:
            col = _chroma_client.get_collection(name=col_name)
        except Exception:
            col = _chroma_client.create_collection(name=col_name)
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [{"title": c["title"], "source_url": c["source_url"], "created": c["created"]} for c in chunks]
        if _emb_model is not None:
            emb = _emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
            col.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=emb)
        else:
            col.upsert(ids=ids, metadatas=metadatas, documents=texts)
        _chroma_client.persist()
        return True
    except Exception as e:
        print("Chroma index error:", e)
        return False

def wikipedia_sync_and_export(query: str, topic: str, max_pages: int = 3, index_to_chroma: bool = True, export_for_offline: bool = True) -> typing.Tuple[int, str]:
    chunks = build_chunks_from_wikipedia(query, topic, max_pages=max_pages)
    if not chunks:
        return (0, "Keine Wikipedia-Ergebnisse oder wikipedia-Modul fehlt.")
    import_path = ""
    idx_ok = False
    try:
        # dedup + save import file
        import_path = save_import_file(topic, chunks)
    except Exception as e:
        import_path = f"Fehler beim Schreiben Import-Datei: {e}"
    if index_to_chroma:
        try:
            idx_ok = index_chunks_to_chroma(chunks, topic)
        except Exception:
            idx_ok = False
    # attempt to import directly to offline_zagent (file-based)
    if export_for_offline and offline_zagent is not None and isinstance(import_path, str) and os.path.exists(import_path):
        try:
            offline_zagent.import_chunks(import_path)
        except Exception:
            pass
    msg = f"Chunks gesamt: {len(chunks)}; indexiert: {idx_ok}; export: {import_path}"
    return (len(chunks), msg)

# -------- LLM Call + structured response parsing --------

def call_llm(prompt: str) -> str:
    if llm is None:
        return "(Kein Online-LLM konfiguriert.)"
    try:
        out = llm(prompt)
        if isinstance(out, str):
            return out
        if hasattr(out, "generations"):
            parts = []
            for gen_list in out.generations:
                if isinstance(gen_list, list) and len(gen_list) > 0 and hasattr(gen_list[0], "text"):
                    parts.append(gen_list[0].text)
            if parts:
                return "\n\n".join(parts)
        if isinstance(out, dict) and "text" in out:
            return out["text"]
        return str(out)
    except Exception as e:
        return f"(Fehler beim LLM-Aufruf: {e})"

def parse_structured_response(text: str) -> dict:
    """
    Erwartet JSON-Ausgabe vom LLM. Robust gegen Pre/Post-Text.
    Fallback: wraps raw text into result dict with status 'ok'.
    """
    text = text.strip()
    # attempt to find first { ... } block
    try:
        # find first '{' and last '}' to try to extract JSON
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            json_text = text[first:last+1]
            parsed = json.loads(json_text)
            # ensure keys
            return {
                "improved_query": parsed.get("improved_query") or parsed.get("improved_query_text") or "",
                "answer": parsed.get("answer",""),
                "status": parsed.get("status","ok"),
                "reason": parsed.get("reason",""),
                "suggestions": parsed.get("suggestions", [])
            }
    except Exception:
        pass
    # fallback: plain text -> wrap
    return {"improved_query": "", "answer": text, "status": "ok", "reason": "", "suggestions": []}

# -------- Chat handler (Persona + programming-language support) --------

SUPPORTED_LANGS = ["auto", "python", "javascript", "java", "c", "cpp", "go", "rust", "bash", "sql"]

def build_prompt(message: str, topic: str, previous: str, web_snippets: str, persona: str, prog_lang: str) -> str:
    lang_hint = ""
    if prog_lang and prog_lang.lower() != "auto":
        lang_hint = f"Wenn die Antwort Code enthält, benutze die Programmiersprache: {prog_lang} und formatiere das Code-Beispiel korrekt mit Syntax-Highlighting."
    else:
        lang_hint = "Wenn die Frage Code enthält, erkenne die passende Programmiersprache automatisch und gib das Beispiel korrekt formatiert zurück."

    prompt = f"""
Du bist Zero, ein lokaler Assistent, der bevorzugt über Ollama läuft.
{persona}

System-Infos:
- Modell: {get_zeroagent_info()['model_name']}
- Python: {get_zeroagent_info()['python_version']}
- OS: {get_zeroagent_info()['os']}

THEMA: {topic}

Bisherige Fragen und Antworten zu diesem Thema:
{previous or 'Noch keine Einträge.'}

{web_snippets}

Aufgabe:
1) Formuliere die Nutzerfrage bei Bedarf klarer / präziser (Feld: improved_query).
2) Beantworte die Frage knapp und hilfreich auf Deutsch (Feld: answer).
3) Wenn du die Anfrage ablehnen musst (z.B. gefährlich, illegal), setze status='refused' und gib reason an.
4) Wenn möglich, gib kurze "suggestions" (z.B. sichere Alternativen).
5) Beachte die Sicherheitsregeln und Persona: weigere dich nur bei hohen Risiken, vermeide beleidigende Inhalte.

{lang_hint}

ANTWORTFORMAT: Gib ausschließlich ein JSON-Objekt zurück mit den Feldern:
- improved_query (string)
- answer (string)
- status (string) // 'ok' oder 'refused'
- reason (string) // optional
- suggestions (list of strings) // optional

Beispiel:
{{"improved_query":"Kurz und präzise Version", "answer":"Antwort auf Deutsch...", "status":"ok", "reason":"", "suggestions":[]}}
"""
    # append user's original message at the end (for context)
    prompt += f"\nUSER_MESSAGE:\n{message}\n"
    return prompt

def chat(message: str, history: list, topic: str, use_web: bool = False, use_offline: bool = True, prog_lang: str = "auto", wiki_sync: bool = False):
    topic = (topic or "allgemein").strip() or "allgemein"

    # previous entries
    previous_entries = read_json(get_topic_path(topic)) or []
    previous = "\n".join(f"- [{e.get('time','')}] Q: {e.get('question','')} A: {e.get('answer','')}" for e in previous_entries[-8:]) if previous_entries else "Noch keine Einträge."

    # optional wiki sync
    wiki_msg = ""
    if wiki_sync and _HAS_WIKI:
        _, info = wikipedia_sync_and_export(message, topic, max_pages=2, index_to_chroma=True, export_for_offline=True)
        wiki_msg = f"[Wikipedia-Sync] {info}\n"

    # attempt offline first (fast)
    if use_offline and offline_zagent is not None:
        try:
            offline_answer, conf = offline_zagent.answer_offline(topic, message)
            if offline_answer and conf >= 0.6:
                result = {"improved_query": "", "answer": offline_answer, "status": "ok", "reason": "", "suggestions": []}
                result["persona"] = PERSONA_TEXT
                save_qa(topic, message, result)
                return offline_answer
        except Exception:
            pass

    # optional web search snippets (Bing) - reuse existing function in offline_zagent if available
    web_snippets = ""
    try:
        # if web_search_bing exists in this module (or offline_zagent), call it
        if "web_search_bing" in globals():
            web_snippets = globals()["web_search_bing"](message) or ""
        elif offline_zagent and hasattr(offline_zagent, "web_search_bing"):
            web_snippets = offline_zagent.web_search_bing(message)
        if web_snippets:
            web_snippets = f"Websuche (aktuelles Online-Input):\n{web_snippets}\n"
    except Exception:
        web_snippets = ""

    # build prompt and call LLM
    prompt = build_prompt(message, topic, previous, wiki_msg + web_snippets, PERSONA_TEXT, prog_lang)
    raw = call_llm(prompt)
    parsed = parse_structured_response(raw)
    # attach persona and source info
    parsed["persona"] = PERSONA_TEXT
    parsed["source"] = "online" if llm is not None else "raw"
    # if refused, produce friendly refusal text
    if parsed.get("status") == "refused":
        reply = f"Ich kann dabei nicht helfen. Grund: {parsed.get('reason','Unbekannt')}\n\nAlternativen:\n"
        for s in parsed.get("suggestions", []):
            reply += f"- {s}\n"
        save_qa(topic, message, parsed)
        return reply
    # otherwise return answer (improved format)
    improved = parsed.get("improved_query","").strip()
    answer = parsed.get("answer","").strip()
    # store structured entry
    save_qa(topic, message, parsed)
    # if improved_query present, include it above the answer
    if improved:
        return f"Verbesserte Frage: {improved}\n\nAntwort:\n{answer}"
    return answer or "(Keine Antwort erhalten.)"

# expose helper for Wikipedia sync (UI)
def wikipedia_sync_and_export_ui(query: str, topic: str, pages: int = 2, index_to_chroma: bool = True, export_for_offline: bool = True):
    num, info = wikipedia_sync_and_export(query, topic, max_pages=pages, index_to_chroma=index_to_chroma, export_for_offline=export_for_offline)
    return f"Erzeugte Chunks: {num}\n{info}"

# -------- Gradio App --------

def main():
    prog_dropdown = gr.Dropdown(choices=SUPPORTED_LANGS, label="Programmiersprache (Code-Ausgabe)", value="auto")
    demo = gr.ChatInterface(
        fn=chat,
        title="ZeroAgent – Online/Offline mit Persona & Programmier-Support",
        description=(
            "Themen: z.B. `trading_marktwirtschaft`, `marketing`.\n"
            "Features: Offline-KB, Wikipedia-Sync, strukturierte LLM-Ausgabe (ImprovedQuery+Answer+Status), deduplizierte Imports.\n"
            "Wähle Programmiersprache (oder 'auto') für Code-Antworten."
        ),
        additional_inputs=[
            gr.Textbox(label="Thema (Kategorie)", value="trading_marktwirtschaft"),
            gr.Checkbox(label="Use web search (online)", value=False),
            gr.Checkbox(label="Use offline KB/mini-model (energieeffizient)", value=True),
            gr.Checkbox(label="Wikipedia: Suche & Sync (erweitert Offline-KB)", value=False),
            prog_dropdown,
        ],
    )
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
