from pypdf import PdfReader
import streamlit as str
import networkx as nx
import requests
import json
import tempfile
import os
import re
from pyvis.network import Network

OLLURL = "http://localhost:11434/api/generate"
MODEL = "phi3"
VERBOTEN = {
    r"\b[A-Z][a-z]+ University\b",
    r"\b[A-Z][a-z]+ College\b",
    r"\bUniversity of [A-Za-z]+\b",
    r"\bProperty of [A-Za-z]+\b"
}

str.title("Artificial Examinator")
str.write("Upload PDFs to create a 35-question exam and answer sheet.")

def query(prompt):
    response = requests.post(
        OLLURL,
        json={"model": MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def extraction(pdf_files):
    text = ""

    for uploaded_file in pdf_files:
        reader = PdfReader(uploaded_file)

        for page in reader.pages:
            txt = page.extract_text()

            if txt:
                lines = txt.split("\n")
                lines = [l for l in lines if not l.strip().isdigit()]
                lines = [l for l in lines if len(l.split()) > 3]
                text += " ".join(lines) + "\n"

    return text

def clean_question(text):
    cleaned = text.replace("\\", " ")
    for term in VERBOTEN:
        safe = re.escape(term)
        cleaned = re.sub(safe, " ", cleaned)
    return cleaned.strip()

def valid_question(question):
    if not re.match(r"^\d+\.\s+", question):
        return False
    if len(question.split()) < 4:
        return False
    if re.search(r"\bAnswer:|\bAns\b|\bCorrect\b", question, re.IGNORECASE):
        return False
    if re.search(r"\[\d+\]|\(p\.\d+\)", question):
        return False
    if question.endswith(":"):
        return False
    return True

def extract_questions(raw):
    questions = re.split(r'\n\d+\.\s', "\n" + raw)
    questions = [q.strip() for q in questions if q.strip()]
    questions = [f"{i+1}. {q}" for i, q in enumerate(questions)]
    return questions

def knowledge_graph(text):
    graph_prompt = f"""
Extract key concepts and relationships from the following text.
Return JSON ONLY in this format:

{{
    "nodes": ["A", "B"],
    "edges": [["A", "B"], ["B", "C"]]
}}

Text:
{text}
"""
    response = query(graph_prompt)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return {"nodes": [], "edges": []}
        return {"nodes": [], "edges": []}

def display_kg(kg):
    graph = Network(notebook=False, height="650px", width="100%", directed=True)
    graph.barnes_hut()

    for n in kg["nodes"]:
        graph.add_node(n, label=n)

    for s, t in kg["edges"]:
        graph.add_edge(s, t)

    temp = tempfile.gettempdir()
    path = os.path.join(temp, "knowledge_graph.html")
    graph.save_graph(path)

    str.subheader("Knowledge Graph")
    str.components.v1.html(open(path, "r").read(), height=650)

def question_batch(prompt, expected=5, max_retries=3):
    for attempt in range(max_retries):
        response = query(prompt)
        raw = extract_questions(response)
        cleaned = [clean_question(q) for q in raw]
        valid = [q for q in cleaned if valid_question(q)]
        if len(valid) >= expected:
            return valid[:expected]
    needed = expected - len(valid)
    valid += [f"{len(valid)+i+1}. [FAILED TO GENERATE QUESTION]" for i in range(needed)]
    return valid

def question_generator(text, size=5, batch=7):
    questions = []
    for batch_num in range(1, batch + 1):
        prompt = f"""
You are generating batch {batch_num} of {batch}.
Generate EXACTLY {size} multiple-choice questions based on the text below.

CRITICAL RULES — FOLLOW THEM EXACTLY:

1. You MUST output all {size} questions in full.
   - Do NOT skip any.
   - Do NOT summarize.
   - Do NOT write things like:
     • "This pattern continues..."
     • "And so on..."
     • "Repeat the same for the rest."
   If you skip or summarize even once, the output is invalid.

2. DO NOT copy sentences from the text.
   - Paraphrase everything.
   - Reformulate concepts in your own words.

3. REMOVE ANY personal names, institution names, school names,
   company names, or specific locations found in the text.

4. Each question MUST have:
   - A clear question
   - Four answer choices (A–D)
   - EXACTLY one correct answer
   - A difficulty label (Easy, Medium, or Hard)

5. Use this output format EXACTLY:

1. Question text?
   A. Option
   B. Option
   C. Option
   D. Option
Correct: B
Difficulty: Medium

2. Question text?
   A. Option
   B. Option
   C. Option
   D. Option
Correct: A
Difficulty: Easy

(continue this exact pattern until question {size})

TEXT FOR QUESTION GENERATION:
{text}
"""
        batch_questions = question_batch(prompt, expected=size)
        questions.extend(batch_questions)

    # Renumber to 1-35
    final_questions = []
    for i, q in enumerate(questions, start=1):
        q_fixed = re.sub(r"^\d+\.", f"{i}.", q)
        final_questions.append(q_fixed)
    return final_questions[:35]

def answer_key(questions, text):
    prompt = f"""
Provide ONLY the answer key in this format:

1. A
2. C
3. D
...

Questions:
{questions}

Reference text:
{text}
"""
    return query(prompt)

uploaded_files = str.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with str.spinner("Extracting text from PDFs..."):
        full_text = extraction(uploaded_files)

    str.success("PDFs loaded successfully!")

    if str.button("Generate 35 Questions"):
        with str.spinner("Generating questions..."):
            questions = question_generator(full_text)

        str.subheader("35 Questions")
        str.text_area("Questions", questions, height=600)

        str.session_state["questions"] = questions
        str.session_state["text"] = full_text

    if "questions" in str.session_state:

        if str.button("Generate Answer Key"):
            with str.spinner("Generating answer key..."):
                key = answer_key(str.session_state["questions"], str.session_state["text"])

            str.subheader("Answer Key")
            str.text_area("Answer Key", key, height=400)

        if str.button("Generate Knowledge Graph"):
            with str.spinner("Generating knowledge graph..."):
                kg = knowledge_graph(str.session_state["text"])

            display_kg(kg)
