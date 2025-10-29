# 🚀 Project 6 – Groq + LangChain RAG Chatbot (Multiple Files)

## 🧩 Overview

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) chatbot** powered by **Groq’s ultra-fast LLMs** and **LangChain**. The chatbot can **analyze text, PDFs, URLs, and images**, store them in a **vector database (FAISS)**, and answer user questions contextually with accurate references.

It’s built in **modular form** (multiple files) to maintain scalability, reusability, and clean structure — ideal for production-level AI projects.

---

## 🧠 Features

* **Groq LLM Integration:** Uses Groq’s high-speed inference for lightning-fast responses.
* **LangChain Pipeline:** For chaining prompts, managing context, and connecting with retrieval modules.
* **RAG Architecture:** Combines LLM reasoning with factual grounding from stored documents.
* **Document Uploads:** Supports `.txt`, `.pdf`, `.png`, `.jpg`, and URLs.
* **Vector Storage:** Uses **FAISS** for efficient similarity search.
* **Streamlit Interface:** Interactive web-based chatbot UI.

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **LangChain**
* **Groq LLM**
* **FAISS Vector Store**
* **HuggingFace Embeddings**
* **Streamlit**
* **dotenv** (for managing API keys)

---

## 🏗️ Folder Structure

```
📂 groq_langchain_rag_chatbot
 ┣ 📜 app.py                 # Main Streamlit interface
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 .env                   # API key configuration
 ┗ 📜 README.md              # Project documentation
```

---

## 🧰 Installation

1. **Clone the repository**

   ```bash
   git clone 
   cd project-6
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your environment variables**
   Create a `.env` file:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 💡 How It Works

1. Upload documents (text, PDF, images, or links).
2. The system extracts and embeds content using **HuggingFace Embeddings**.
3. FAISS stores embeddings for similarity retrieval.
4. User queries are matched against stored data.
5. **Groq LLM** generates a grounded, context-aware answer.

---

## 🧾 Requirements

Check `requirements.txt` for all dependencies:

```
langchain
langchain_groq
faiss-cpu
streamlit
python-dotenv
sentence-transformers
pillow
```

---

## 🌟 Future Enhancements

* Add chat history memory
* Integrate multi-user sessions
* Expand support for audio and video documents

---

## 👨‍💻 Author

**Ujjwal Kumar**
AI | Building 40 AI Projects in 40 Days

