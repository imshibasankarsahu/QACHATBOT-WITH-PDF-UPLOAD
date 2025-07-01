# 📄 PDF QA Chatbot with Memory (LangChain + Streamlit)

A Streamlit app that allows users to upload PDFs and ask questions about them — with full **chat history memory** and **retrieval-augmented generation (RAG)** using LangChain, Chroma, and HuggingFace embeddings.

---

## 🚀 Features

- ✅ Upload one or more PDF files
- ✅ Extract and chunk text using `RecursiveCharacterTextSplitter`
- ✅ Store chunks in a Chroma vector database
- ✅ Ask natural language questions about the documents
- ✅ Maintains conversation memory per session
- ✅ Powered by **LangChain**, **Streamlit**, **Chroma**, and **ChatGroq (Gemma2-9b-It)**

---

## 🧠 Tech Stack

| Component | Description |
|----------|-------------|
| `Streamlit` | Frontend UI |
| `LangChain` | RAG framework and memory management |
| `Chroma` | Vector store |
| `HuggingFace Embeddings` | For document embeddings |
| `ChatGroq` | LLM backend (Gemma2-9b-It model) |
| `PyPDFLoader` | For PDF document parsing |

---

## 📁 Folder Structure

```
project/
│
├── app.py                  # Main Streamlit app
├── .env                    # API keys (Groq, HuggingFace)
├── requirements.txt        # Required packages
└── README.md               # You're here!
```

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/pdf-chatbot-genai.git
cd pdf-chatbot-genai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token
```

You will also enter your **Groq API key** inside the app UI.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Screenshots

Coming soon...

---

## 🤝 Contributing

Feel free to fork the repo, open issues, or submit PRs. Feedback is welcome!

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Made with ❤️ 
Part of my **30-Day GenAI Real-World Project Challenge**.
