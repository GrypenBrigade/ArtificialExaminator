from pypdf import PdfReader
import streamlit as str
import networkx as nx
import requests
import json
import tempfile
import os
from pyvis.network import Network

OLLURL = "http://localhost:11434/api/generate"
MODEL = "phi3"

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
    except:
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

def question_generator(text):
    prompt = f"""
Create 35 multiple-choice questions from the text below.

Format EXACTLY like this:

1. Question?
   A. Option
   B. Option
   C. Option
   D. Option
Correct: A
Difficulty: Medium

Text:
{text}
"""
    return query(prompt)

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
