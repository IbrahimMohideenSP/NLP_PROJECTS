# 📘 Research Paper Analyzer

An AI-powered Research Paper Analyzer built using Python, Streamlit, Hugging Face Transformers, and PyMuPDF.

This application allows users to upload research papers in PDF format and automatically generates summaries for important sections including:

- Abstract
- Introduction
- Methodology
- Results
- Conclusion
- Overall Summary

The project uses the BART Transformer Model for high-quality NLP-based summarization.

---

# 🚀 Features

- Upload up to 5 research papers
- Automatic PDF text extraction
- Smart text cleaning and preprocessing
- Section-wise research paper analysis
- AI-based summarization using BART
- Interactive Streamlit user interface
- Fallback model support
- Chunk-based summarization for large papers

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyMuPDF
- BART Large CNN
- Regex

---

# 📁 Project Structure

```bash
Research-Paper-Analyzer/
│
├── app.py
├── app1.py
├── app2.py
├── requirements.txt
└── README.md
```

---

# 📦 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/research-paper-analyzer.git
cd research-paper-analyzer
```

---

## 2️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📄 requirements.txt

Create a file named `requirements.txt` and add:

```txt
streamlit
transformers
torch
pymupdf
sentencepiece
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 🧠 AI Models Used

## Primary Model
`facebook/bart-large-cnn`

## Fallback Model
`sshleifer/distilbart-cnn-12-6`

---

# 🔄 Working Process

1. Upload research papers in PDF format
2. Extract text using PyMuPDF
3. Clean and preprocess text
4. Identify important sections
5. Split long text into chunks
6. Generate summaries using BART
7. Display section-wise analysis

---

# 📸 Application Output

The application generates:

- Abstract Summary
- Introduction Summary
- Methodology Summary
- Results Summary
- Conclusion Summary
- Overall Paper Summary

All results are displayed in an interactive Streamlit interface.

---

# 🎯 Future Enhancements

- Research keyword extraction
- Citation generation
- Research trend analysis
- Multi-language support
- Chat with PDF feature
- GPU acceleration

---

# 👨‍💻 Author

## Ibrahim Mohideen SP

B.Tech Artificial Intelligence and Data Science

Interested in:
- Artificial Intelligence
- Machine Learning
- NLP
- Data Science Projects

---

# 📜 License

This project is licensed under the MIT License.
