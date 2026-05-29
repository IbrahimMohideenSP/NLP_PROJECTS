"""
Research Paper Analyzer (FIXED VERSION)
"""

import re
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PRIMARY_MODEL = "facebook/bart-large-cnn"
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_OUTPUT_TOKENS = 180

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="Research Paper Analyzer", layout="wide")
st.title("📘 Research Paper Analyzer")
st.caption("AI-powered summarization using BART")

# ──────────────────────────────────────────────
# Model Loading (FIXED)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        return pipeline(
            "summarization",   # ✅ FIXED HERE
            model=PRIMARY_MODEL,
            tokenizer=PRIMARY_MODEL,
            device=-1
        )
    except Exception as e:
        st.warning(f"Primary model failed: {e}")
        return pipeline(
            "summarization",
            model=FALLBACK_MODEL,
            tokenizer=FALLBACK_MODEL,
            device=-1
        )

summarizer = load_summarizer()

# ──────────────────────────────────────────────
# PDF Text Extraction
# ──────────────────────────────────────────────
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return text.strip()

# ──────────────────────────────────────────────
# Section Extraction (FIXED)
# ──────────────────────────────────────────────
SECTION_MAP = {
    "abstract": (0.0, 0.08),
    "introduction": (0.08, 0.25),
    "method": (0.25, 0.50),
    "results": (0.50, 0.80),
    "conclusion": (0.80, 1.0),
}

def extract_section(text, key):
    start_ratio, end_ratio = SECTION_MAP[key]
    start = int(len(text) * start_ratio)
    end = int(len(text) * end_ratio)
    return clean_text(text[start:end])[:1200]

# ──────────────────────────────────────────────
# Summarization (IMPROVED)
# ──────────────────────────────────────────────
def summarize(text, min_len=40, max_len=MAX_OUTPUT_TOKENS):
    if not text or len(text.split()) < 20:
        return "Not enough content."

    words = text.split()

    # Chunking for better coverage
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    summaries = []

    for chunk in chunks[:3]:
        try:
            result = summarizer(
                chunk,
                min_length=30,
                max_length=120,
                do_sample=False
            )
            summaries.append(result[0]["summary_text"])
        except:
            continue

    combined = " ".join(summaries)

    try:
        final = summarizer(
            combined,
            min_length=min_len,
            max_length=max_len,
            do_sample=False
        )
        return final[0]["summary_text"]
    except:
        return combined

# ──────────────────────────────────────────────
# Analysis Pipeline
# ──────────────────────────────────────────────
def analyze_paper(raw_text):
    text = clean_text(raw_text)

    abstract = extract_section(text, "abstract")
    intro = extract_section(text, "introduction")
    method = extract_section(text, "method")
    results = extract_section(text, "results")
    conclusion = extract_section(text, "conclusion")

    return {
        "📝 Abstract": abstract[:700],
        "📖 Introduction": summarize(intro),
        "🤖 Model / Method": summarize(method),
        "📊 Results": summarize(results),
        "✅ Conclusion": summarize(conclusion),
        "🔍 Overall Summary": summarize(abstract + " " + intro)
    }

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
files = st.file_uploader(
    "📂 Upload up to 5 PDF papers",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    files = files[:5]

    for i, file in enumerate(files, 1):
        with st.spinner(f"Analyzing {file.name}..."):
            raw = extract_text(file)
            analysis = analyze_paper(raw)

        with st.expander(f"📄 Paper {i}: {file.name}", expanded=True):
            for k, v in analysis.items():
                st.markdown(f"**{k}**")
                st.info(v)

else:
    st.info("Upload research papers to begin.")