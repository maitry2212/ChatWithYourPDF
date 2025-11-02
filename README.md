---

# 🧠 PDF Question Answering using RAG & Gemini 2.0

This Streamlit web app lets you **upload a PDF**, then **ask questions** about its content using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LangChain**, **FAISS**, **Hugging Face embeddings**, and **Gemini 2.0 Flash**.

---

## 🚀 Features

✅ Upload and parse **any PDF document**
✅ Automatic **text extraction** and **chunking**
✅ **Vector embedding** using `HuggingFaceEmbeddings`
✅ Fast **semantic search** with **FAISS**
✅ Real-time **question answering** with **Gemini 2.0 Flash**
✅ Caching and indexing for improved performance
✅ Easy UI built with **Streamlit**

---

## 🧩 Tech Stack

| Component     | Technology                   |
| ------------- | ---------------------------- |
| Language      | Python                       |
| Framework     | Streamlit                    |
| LLM           | Google Gemini 2.0 Flash      |
| Embeddings    | Hugging Face (MiniLM models) |
| Vector Store  | FAISS                        |
| PDF Parsing   | PyPDF2                       |
| RAG Framework | LangChain                    |

---

## 🗂️ Project Structure

```
📦 pdf-rag-gemini
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── faiss_index/           # Cached FAISS vector stores (auto-created)
├── README.md              # Project documentation
└── .env                   # Store your Google API key here
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/pdf-rag-gemini.git
cd pdf-rag-gemini
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Set your **Google API key** for Gemini:

### Option 1 — In `.env` file

Create a `.env` file in the project root and add:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### Option 2 — In Streamlit Secrets 

If you deploy this on Streamlit Cloud, add this key in:

```
Settings → Secrets → Add new secret
```

### Option 3 — Direct Environment Variable

```bash
export GOOGLE_API_KEY=your_google_api_key_here  # macOS/Linux
set GOOGLE_API_KEY=your_google_api_key_here     # Windows
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Once the app launches, open the local URL (usually `http://localhost:8501`) and:

1. **Upload** your PDF document
2. **Adjust** settings (chunk size, model, retriever top-k, etc.)
3. **Ask** any question about your document!

---

## ⚡ Example Workflow

1. Upload: *"AI Research Paper.pdf"*
2. App extracts and chunks text
3. FAISS index is built and cached
4. Ask:

   ```
   What are the main contributions of this paper?
   ```
5. The model retrieves relevant passages and answers using Gemini Flash.

---

## 🧠 How It Works (Architecture Overview)

**RAG Pipeline**

```
PDF → Text Extraction → Chunking → Embedding → FAISS Vector Store → Retriever → LLM (Gemini)
```

1. **Text Extraction** — Uses `PyPDF2` to extract readable text.
2. **Chunking** — Splits long documents into overlapping text chunks with `RecursiveCharacterTextSplitter`.
3. **Embedding** — Converts chunks into numerical vectors via `HuggingFaceEmbeddings`.
4. **Indexing** — Stores vectors in FAISS for similarity search.
5. **Retrieval + Generation** — Fetches top relevant chunks and sends them with your query to **Gemini 2.0 Flash** for final answer synthesis.

---

## 🧰 Requirements

Your `requirements.txt` should include:

```
streamlit
PyPDF2
langchain
langchain-community
langchain-google-genai
faiss-cpu
sentence-transformers
huggingface-hub
python-dotenv
```

---

## 🧑‍💻 Author

**Maitry Chauhan**
🔗 [GitHub](https://github.com/maitry2212)
💡 Engineering Student | Exploring AI, LangChain & Data Science

---

## 🪪 License

This project is licensed under the **MIT License** — you’re free to modify and use it for personal or academic purposes.

---

