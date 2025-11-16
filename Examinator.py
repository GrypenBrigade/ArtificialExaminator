import pypdf as pdf
import streamlit as str
import networkx as net
import requests
import json
import tempfile
import os
from pyvis.network import Network

OLLURL = "http://localhost:11434/api/generate"
MODEL = "phi3"

str.title("Artificial Examinator")
str.write("Upload PDFs to create a 35 question exam questionnaire and answer sheet")

def query(prompt):
    response = response.post(
        OLLURL,
        json={"MODEL": MODEL, "PROMPT": prompt, "STREAM": False}
    )
    return response.json()["response"]

def extraction(pdf_files):
    text = ""
    for uploaded_file in pdf_files:
        reader = pdf(uploaded_file)
        