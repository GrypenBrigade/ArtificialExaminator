from pypdf import PdfReader
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import streamlit as st
import tempfile
import os
import re
import json
import faiss
import numpy

try:
    from SECRETS import API
    api_key = API
except ImportError:
    api_key = ""

MODEL = "gemini-2.5-flash"

VERBOTEN = {
    r"\b[A-Z][a-z]+ University\b",
    r"\b[A-Z][a-z]+ College\b",
    r"\bUniversity of [A-Za-z]+\b",
    r"\bProperty of [A-Za-z]+\b"
}

if api_key:
    CLIENT = genai.Client(api_key=api_key)
else:
    st.warning("API Key missing.")
    CLIENT = None

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed = load_embedder()

st.title("Artificial Examinator")
st.write("Upload PDFs to create a multiple choice question exam, answer key, and knowledge graph.")


def call_genai(model, prompt, temperature=0.7, json_mode=False):
    if not CLIENT:
        return ""
    
    config_args = {
        "temperature": temperature,
        "max_output_tokens": 2000,
        "safety_settings": [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH")
        ]
    }

    if json_mode:
        config_args["response_mime_type"] = "application/json"

    try:
        response = CLIENT.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_args)
        )
        return response.text if response.text else ""
    except Exception as e:
        print(f"GenAI Error: {e}")
        return ""

def extraction(pdf_files):
    text = ""
    for uploaded_file in pdf_files:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    txt = re.sub(r'\s+', ' ', txt).strip()
                    text += txt + "\n\n"
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
    return text

def embed_text(text, chunk_size=30):
    if not text.strip():
        return [], None, None
    
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size * 10: 
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        return [], None, None

    embeddings = embed.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return chunks, embeddings, index

def retrieve_chunks(query_text, chunks, index, top_k=3):
    if not index or not chunks:
        return ""
    query_vec = embed.encode([query_text], convert_to_numpy=True)
    distances, idxs = index.search(query_vec, min(top_k, len(chunks)))
    valid_idxs = [i for i in idxs[0] if 0 <= i < len(chunks)]
    return " ".join([chunks[i] for i in valid_idxs])


# def knowledge_graph(text):
#     if not chunks:
#         st.error("No text chunks available.")
#         return {"nodes": [], "edges": []}


#     num_to_process = 5
#     step = max(1, len(chunks) // num_to_process)
#     selected_indices = range(0, len(chunks), step)[:num_to_process]
    
#     selected_text = ""
#     for i in selected_indices:
#         selected_text += f"--- Section {i+1} ---\n{chunks[i]}\n\n"

 
#     prompt = f"""
#     Analyze the following text sections from a study handout.
#     Identify key concepts (Nodes) and their relationships (Edges).
    
#     RULES:
#     1. Output strictly valid JSON.
#     2. Merge concepts that are similar (e.g., "Tank" and "Tanks" -> "Tank").
#     3. Format: {{ "nodes": ["A", "B"], "edges": [["A", "B"], ["B", "C"]] }}
    
#     Text Sections:
#     {selected_text}
#     """
    
    
#     raw_response = call_genai(MODEL, prompt, temperature=0.1, json_mode=True)

   
#     try:
        
#         cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        
        
#         start = cleaned.find("{")
#         end = cleaned.rfind("}") + 1
#         if start != -1 and end != 0:
#             cleaned = cleaned[start:end]

#         data = json.loads(cleaned)
        
#         if "nodes" not in data: data["nodes"] = []
#         if "edges" not in data: data["edges"] = []
        
#         return data

#     except json.JSONDecodeError:
#         st.warning("Graph generation failed to parse JSON. Retrying with a simplified prompt...")
#         return {"nodes": [], "edges": []}

# def display_kg(kg):
#     if not kg or not kg.get("nodes") or len(kg["nodes"]) == 0:
#         st.warning("No relationships found to graph. (Graph data is empty)")
#         return


#     net = Network(notebook=False, height="600px", width="100%", directed=True, 
#                   bgcolor="#222222", font_color="white")
    
#     for n in kg["nodes"]:
#         net.add_node(n, label=n, color="#4a90e2", title=n)
    
#     for edge in kg["edges"]:
#         if len(edge) >= 2:
#             source, target = edge[0], edge[1]
#             if source in kg["nodes"] and target in kg["nodes"]:
#                 net.add_edge(source, target)

#     net.force_atlas2based()

#     temp_dir = tempfile.gettempdir()
#     path = os.path.join(temp_dir, "knowledge_graph.html")
    
#     try:
#         net.save_graph(path)
#         with open(path, 'r', encoding='utf-8') as f:
#             html = f.read()
#         st.subheader("Knowledge Graph")
#         st.components.v1.html(html, height=620)
#     except Exception as e:
#         st.error(f"Error displaying graph: {e}")

def question_batch(prompt, chunks, index):
    context = retrieve_chunks(prompt, chunks, index, top_k=3)
    if not context and chunks:
        context = chunks[0] 

    full_prompt = f"""
    Generate 5 multiple-choice questions based on the text below.
    Format exactly like this:
    
    1. Question?
    A) Option
    B) Option
    C) Option
    D) Option
    Answer: B (However, hide this in the final output. A seperate function will extract it.)
    

    Critical Rules:
    - Do not reference sample problems provided in the context.
    - Do not say "Based on the text" or similar phrases.
    - Ensure questions are clear and unambiguous.
    - Avoid using any proper nouns or identifiable information.

    Text: {context}
    """
    return call_genai(MODEL, full_prompt, temperature=0.5)

def question_generator(text, chunks, index, ui_placeholder, progress_bar):
    all_questions = []
    total_needed = 35
    questions_per_batch = 5
    batches_needed = (total_needed // questions_per_batch) + 1

    for i in range(batches_needed):

        progress_bar.progress((i) / batches_needed)
        
        start_idx = i * 400 
        if start_idx >= len(text): start_idx = 0 
        query_snippet = text[start_idx : start_idx+400]
        

        raw_response = question_batch(query_snippet, chunks, index)
        
        if not raw_response: continue


        split_qs = re.split(r'\n\d+[\.)]', "\n" + raw_response)
        new_qs_found = 0
        
        for q in split_qs:
            q = q.strip()
            if "?" in q and ("Answer:" in q or "Correct:" in q):
                all_questions.append(q)
                new_qs_found += 1
        
        preview_text = ""
        for idx, q in enumerate(all_questions, 1):
             preview_text += f"{idx}. {q}\n\n"
             
        ui_placeholder.text_area(
            f"Generating... ({len(all_questions)} questions found so far)", 
            preview_text, 
            height=400
        )
        
        if len(all_questions) >= total_needed:
            break

    progress_bar.progress(1.0)

    final_questions = []
    for i, q in enumerate(all_questions[:total_needed], 1):
        q = q.replace("Answer:", "Correct:")
        final_questions.append(f"{i}. {q}")

    return final_questions

def answer_key(questions):
    q_text = "\n".join(questions)
    prompt = f"""
    Extract ONLY the answer key from these questions.
    Format:
    1. A
    2. C
    ...
    
    Questions:
    {q_text}
    """
    return call_genai(MODEL, prompt, temperature=0.1)



uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Extracting text..."):
        full_text = extraction(uploaded_files)

    st.success(f"PDFs loaded ({len(full_text)} chars).")
    chunks, embeddings, index = embed_text(full_text)


    if st.button("Generate Questions"):
        if len(full_text) < 50:
            st.error("Text extraction failed (text too short).")
        else:
            with st.spinner("Generating questions..."):
                prog_bar = st.progress(0)
                live_preview = st.empty()
                questions = question_generator(full_text, chunks, index, total_needed=35)

            if questions:
                st.session_state["questions"] = questions
                st.session_state["text"] = full_text
                st.session_state["chunks"] = chunks # Save chunks for graph
                # Clear the preview container so we can show the final result cleanly
                live_preview.empty() 
                prog_bar.empty()
            else:
                st.error("No questions generated.")

    if "questions" in st.session_state:
        st.subheader("Exam Paper")
        st.text_area("Questions", "\n\n".join(st.session_state["questions"]), height=500)


        if st.button("Generate Answer Key"):
            with st.spinner("Extracting key..."):
                key = answer_key(st.session_state["questions"])
            st.subheader("Answer Key")
            st.text_area("Key", key, height=400)

