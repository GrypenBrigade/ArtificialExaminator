# Artificial Examinator
## An AI exam generator Streamlit app

Import lesson handout PDFs to generate a multiple choice question exam paper.

<p>Model used: gemini-2.5-flash<br>
Embedder model used: all-MiniLM-L6-v2</p>

## How to use:
1. Install the necessary python packages with this command: `pip install streamlit pypdf google google.genai sentence-transformers faiss-cpu`
2. Get an API key from Google AI Studio
3. Store API key and create a SECRETS.py file in your local file directory
4. Run command `streamlit run Examinator.py`
5. Upload lesson handout PDFs and generate questions

Limitation: Can not read images in PDF files, only the text.

## Code snippets

PDF text Extractor:
```python
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

```
Text Embedder and Retriever:

```python
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

```
Question Batch Function:

```python
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

```

Model Caller:
```python
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
```



