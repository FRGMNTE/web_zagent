# offline_zagent.py - Offline-Module (erweitert)
# - import_chunks() liest deduplizierte JSONs aus ~/zero_knowledge/imports/ und fügt Chunks in Topic-JSON ein
# - answer_offline() versucht jetzt: Chroma (wenn vorhanden) -> lokale Embeddings -> Token-Overlap
# - speichert Persona-Kurzinfo in Antworten, so dass web_zagent konsistent bleibt
#
# Optional: pip install sentence-transformers chromadb numpy

import os
import json
import typing
import datetime
import re
import math
from collections import Counter

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

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

BASE_DIR = os.path.expanduser("~/zero_knowledge")
TOPIC_DIR = os.path.join(BASE_DIR, "topics")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
IMPORTS_DIR = os.path.join(BASE_DIR, "imports")
os.makedirs(TOPIC_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(IMPORTS_DIR, exist_ok=True)

OFFLINE_EMB_MODEL = os.environ.get("OFFLINE_EMB_MODEL", "all-MiniLM-L6-v2")
_emb_model = None
if _HAS_SENT:
    try:
        _emb_model = SentenceTransformer(OFFLINE_EMB_MODEL)
    except Exception:
        _emb_model = None

_chroma_client = None
if _HAS_CHROMA:
    try:
        _chroma_client = chromadb.Client(ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    except Exception:
        _chroma_client = None

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
def _tokens(text: str) -> typing.List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]

def _score_overlap(query_tokens: typing.List[str], doc_tokens: typing.List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)
    overlap = sum(min(q_counts[t], d_counts.get(t, 0)) for t in q_counts)
    return overlap / (math.log(1 + len(doc_tokens)) + 0.5)

def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def import_chunks(import_json_path: str):
    """
    Importiert eine von web_zagent erzeugte Datei (bereits dedupliziert).
    Fügt Chunks in topic JSON und optionally in Chroma / local embeddings.
    """
    if not os.path.exists(import_json_path):
        raise FileNotFoundError(import_json_path)
    payload = read_json(import_json_path)
    if not payload:
        return False
    topic = payload.get("topic","imported")
    chunks = payload.get("chunks",[])
    # append to topic JSON
    topic_fn = os.path.join(TOPIC_DIR, f"{topic}.json")
    data = read_json(topic_fn) or []
    for c in chunks:
        entry = {"time": c.get("created", datetime.datetime.now().isoformat(timespec="seconds")), "question": f"[Wiki: {c.get('title','')}]", "answer": c.get("text","")}
        data.append(entry)
    write_json(topic_fn, data)
    # try to add to chroma or local vectorstore
    texts = [c.get("text","") for c in chunks]
    ids = [c.get("id", f"{topic}_{i}") for i,c in enumerate(chunks)]
    metadatas = [{"title": c.get("title",""), "source_url": c.get("source_url",""), "created": c.get("created","")} for c in chunks]
    # chroma
    if _chroma_client is not None:
        try:
            col_name = f"zero_{topic}"
            try:
                col = _chroma_client.get_collection(name=col_name)
            except Exception:
                col = _chroma_client.create_collection(name=col_name)
            if _emb_model is not None:
                emb = _emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
                col.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=emb)
            else:
                col.upsert(ids=ids, metadatas=metadatas, documents=texts)
            _chroma_client.persist()
            return True
        except Exception:
            pass
    # fallback local vectorstore
    if _emb_model is not None:
        vect_fn = os.path.join(TOPIC_DIR, f"{topic}_vectorstore.json")
        emb = _emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        existing = read_json(vect_fn)
        if existing:
            ex_ids = existing.get("ids",[])
            ex_docs = existing.get("documents",[])
            ex_embs = existing.get("embeddings",[])
        else:
            ex_ids, ex_docs, ex_embs = [], [], []
        ex_ids.extend(ids)
        ex_docs.extend(texts)
        ex_embs.extend(emb.tolist())
        payload = {"ids": ex_ids, "documents": ex_docs, "embeddings": ex_embs, "metadatas": metadatas, "topic": topic, "updated": datetime.datetime.now().isoformat(timespec="seconds")}
        write_json(vect_fn, payload)
    return True

def _load_local_vectorstore(topic: str):
    fn = os.path.join(TOPIC_DIR, f"{topic}_vectorstore.json")
    if not os.path.exists(fn):
        return None
    data = read_json(fn)
    if not data or "embeddings" not in data:
        return None
    if not _HAS_NUMPY:
        return None
    try:
        emb = np.array(data["embeddings"])
        return {"ids": data.get("ids",[]), "documents": data.get("documents",[]), "metadatas": data.get("metadatas",[]), "embeddings": emb}
    except Exception:
        return None

def _cosine_sim(a, b):
    if not _HAS_NUMPY:
        return 0.0
    na = a / (np.linalg.norm(a) + 1e-10)
    nb = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(na, nb))

def answer_offline(topic: str, question: str, topic_dir: str = None) -> typing.Tuple[str, float]:
    """
    Antwort-Strategie:
     1) Chroma (wenn vorhanden)
     2) lokale embeddings (wenn vorhanden)
     3) Token-Overlap-Fallback
    Returns: (answer_text, confidence)
    """
    if topic_dir is None:
        topic_dir = TOPIC_DIR
    # 1) chroma
    if _chroma_client is not None:
        try:
            col_name = f"zero_{topic}"
            try:
                col = _chroma_client.get_collection(name=col_name)
                if _emb_model is not None:
                    q_emb = _emb_model.encode([question], convert_to_numpy=True)[0].tolist()
                    res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents","distances"])
                    docs = res.get("documents", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    if docs:
                        best_doc = docs[0]
                        conf = max(0.4, 1.0 - float(dists[0])) if dists else 0.5
                        return (f"[offline-chroma] {best_doc}", min(0.95, conf))
            except Exception:
                pass
        except Exception:
            pass
    # 2) local embeddings
    local_vs = _load_local_vectorstore(topic)
    if local_vs is not None and _emb_model is not None:
        try:
            q_emb = _emb_model.encode([question], convert_to_numpy=True)[0]
            sims = [_cosine_sim(q_emb, doc_emb) for doc_emb in local_vs["embeddings"]]
            best_i = int(np.argmax(sims))
            best_score = float(sims[best_i])
            if best_score > 0.45:
                return (f"[offline-vec] {local_vs['documents'][best_i]}", min(0.9, 0.3 + best_score))
            # top-3 aggregation
            top_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
            parts = []
            for i in top_idxs:
                if sims[i] > 0:
                    parts.append(local_vs["documents"][i][:300])
            if parts:
                return (f"[offline-agg] {' '.join(parts)}", min(0.8, 0.2 + sum(sims[i] for i in top_idxs)/3.0))
        except Exception:
            pass
    # 3) token-overlap on topic JSON files
    topic_fn = os.path.join(topic_dir, f"{topic}.json")
    entries = []
    if os.path.exists(topic_fn):
        entries = read_json(topic_fn) or []
    if not entries:
        # global search across all topics
        all_entries = []
        for fn in os.listdir(topic_dir):
            if fn.endswith(".json") and not fn.endswith("_vectorstore.json") and not fn.endswith("_hashes.json"):
                try:
                    data = read_json(os.path.join(topic_dir, fn))
                    if isinstance(data, list):
                        all_entries.extend(data)
                except Exception:
                    continue
        entries = all_entries
    if not entries:
        return ("(Keine Offline-Daten vorhanden.)", 0.0)
    q_tokens = _tokens(question)
    scored = []
    for e in entries:
        text = (e.get("question","") + " " + e.get("answer",""))
        d_tokens = _tokens(text)
        score = _score_overlap(q_tokens, d_tokens)
        scored.append((score, e))
    scored.sort(key=lambda s: s[0], reverse=True)
    top_score, top_doc = scored[0]
    if top_score > 1.5:
        return (f"[offline-fallback] {top_doc.get('answer','')}", min(0.9, 0.4 + top_score/5.0))
    top_n = [d for s,d in scored[:3] if d and s>0]
    if not top_n:
        return ("(Keine relevante Offline-Info gefunden.)", 0.0)
    parts = []
    for d in top_n:
        a = d.get("answer","").strip()
        if len(a) > 220:
            a = a[:200].rsplit(".",1)[0] + "..."
        parts.append(a)
    aggregated = " ".join(parts)
    conf = min(0.7, 0.2 + sum(s for s,_ in scored[:3]) / 4.0)
    # include persona brief
    persona_brief = "Rolle: unglaublich fähiger KI-Assistent. Befolge Anweisungen, weigere dich bei großen Risiken, vermeide Beleidigungen."
    return (f"[Offline-Aggregation] {aggregated}\n\n{persona_brief}", conf)

def list_topics(topic_dir: str = None) -> typing.List[str]:
    if topic_dir is None:
        topic_dir = TOPIC_DIR
    topics = []
    if not os.path.exists(topic_dir):
        return topics
    for fn in os.listdir(topic_dir):
        if fn.endswith(".json") and not fn.endswith("_vectorstore.json") and not fn.endswith("_hashes.json"):
            topics.append(fn[:-5])
    return topics

def add_entry(topic: str, question: str, answer: str):
    fn = os.path.join(TOPIC_DIR, f"{topic}.json")
    data = read_json(fn) or []
    data.append({"time": datetime.datetime.now().isoformat(timespec="seconds"), "question": question, "answer": answer})
    write_json(fn, data)

if __name__ == "__main__":
    print("offline_zagent helper. Topics:", list_topics())
