# web_zagent.py - Online-Assistant mit Websuche + Ollama

def answer(question: str) -> str:
    # TODO: später: Websuche + Ollama-Aufruf
    return f"Ich habe deine Frage erhalten: {question}"

import os
import json
import datetime
import platform

import gradio as gr
from langchain_community.llms import Ollama


def get_zeroagent_info():
    return {
        "model_name": "qwen2.5:3b",
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "file_path": os.path.abspath(__file__),
    }


# =========  LLM INITIALISIERUNG  =========

llm = Ollama(model="qwen2.5:3b")


# =========  PERSISTENTES THEMEN-WISSEN  =========

BASE_DIR = os.path.expanduser("~/zero_knowledge")
TOPIC_DIR = os.path.join(BASE_DIR, "topics")
os.makedirs(TOPIC_DIR, exist_ok=True)


def get_topic_path(topic: str) -> str:
    safe = "".join(c for c in topic if c.isalnum() or c in ("_", "-")).lower() or "default"
    return os.path.join(TOPIC_DIR, f"{safe}.json")


def load_topic_snippets(topic: str, max_items: int = 10) -> str:
    path = get_topic_path(topic)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    items = data[-max_items:]
    lines = []
    for item in items:
        q = item.get("question", "")
        a = item.get("answer", "")
        ts = item.get("time", "")
        lines.append(f"- [{ts}] Q: {q}\n  A: {a}")
    return "\n".join(lines)

def save_qa(topic: str, question: str, answer: str) -> None:
    path = get_topic_path(topic)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(
        {   
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "question": question,   
            "answer": answer,
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========  CHAT-FUNKTION FÜR CHATINTERFACE  =========

def chat(message: str, history: list, topic: str):
    topic = topic or "allgemein"
    
    previous = load_topic_snippets(topic)
    info = get_zeroagent_info()
    
    prompt = f"""
Du bist Zero, ein lokaler Assistent, der auf meinem Rechner über Ollama läuft.
Du hast KEINEN Zugriff auf das Internet und arbeitest nur mit:
- deinem trainierten Modellwissen
- den bisherigen Fragen und Antworten zu diesem Thema (siehe unten)

System-Infos (nur zur Einordnung, nicht erfinden):
- Sprachmodell: {info['model_name']}
- Python-Version: {info['python_version']}
- Betriebssystem: {info['os']}
        
THEMA: {topic}
            
Bisherige Fragen und Antworten zu diesem Thema:
{previous or "Noch keine Einträge."}
        
Neue Frage des Users:
{message} 
     
Antwort-Regeln:
- Antworte kurz, präzise und auf Deutsch.
- Erfinde keine externen Daten oder Quellen.
- Wenn dir Informationen fehlen, sag ehrlich, dass du es nicht genau weißt.
- Sprich den Nutzer direkt mit "du" an.
- Wenn der Nutzer nach Fehlern oder Verbesserungen fragt, analysiere gezielt diesen Code.
""" 
    
   answer = llm.invoke(prompt)
    save_qa(topic, message, answer)
    
    # ChatInterface erwartet hier nur den neuen Assistant-Text,
    # history wird intern von Gradio verwaltet.
    return answer


# =========  GRADIO-APP  =========

def main():
    demo = gr.ChatInterface(  
        fn=chat,
        title="ZeroAgent – Themen & Offline-Wissen",
        description=(
            "Wähle ein Thema (z.B. `ki`, `finanzen`, `familie`) "
            "und stelle deine Fragen. ZeroAgent merkt sich pro Thema frühere Q&A."
        ),
        additional_inputs=[
            gr.Textbox(label="Thema (Kategorie)", value="ki"),
        ],
    )
    demo.launch(share=False)


if __name__ == "__main__":
    main()
