import cv2
import pytesseract
import faiss
import pickle
import numpy as np
from fastapi.responses import PlainTextResponse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tesseract Path Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Sentence Transformer Embedder
embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# OCR Function
def extract_text_from_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(image)
    return text

# Text Splitter Function
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

# FAISS Index Creator
def create_faiss_index(chunks):
    embeddings = embedder.embed_documents(chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(index, "index.faiss")

# FAISS Retriever
def retrieve_similar_chunks(query, k=3):
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    query_embedding = embedder.embed_query(query)
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)

    results = [chunks[i] for i in I[0]]
    return results

# Upload Image Endpoint
@app.post("/upload-image/", response_class=PlainTextResponse)
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    extracted_text = extract_text_from_image(content)

    if not extracted_text.strip():
        return PlainTextResponse("**Error:** No text found in image.", status_code=400)

    chunks = split_text_into_chunks(extracted_text)
    create_faiss_index(chunks)

    markdown_response = f"""# Extracted Text from Image 📄

```text
{extracted_text.strip()}
```

**Total Chunks Created:** {len(chunks)}
"""
    return PlainTextResponse(markdown_response)

# # Query Endpoint
# @app.post("/query/", response_class=PlainTextResponse)
# async def query_rag_system(query: str = Form(...)):
#     context_chunks = retrieve_similar_chunks(query)
#     markdown_response = f"""# RAG Query Results 🔍

# **Query:** {query}

# ## Retrieved Context Chunks:
# """
#     for idx, chunk in enumerate(context_chunks, 1):
#         markdown_response += f"\n{idx}. {chunk}"

#     return PlainTextResponse(markdown_response)
