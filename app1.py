import re
import streamlit as st
import fitz
from transformers import pipeline

# =========================
# CONFIGURATION
# =========================
PRIMARY_MODEL = "facebook/bart-large-cnn"
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_OUTPUT_TOKENS = 180

# =========================
# STREAMLIT PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="Research Paper Analyzer",
    layout="wide"
)

st.title("📘 Research Paper Analyzer")
st.caption("AI-powered research paper summarization using BART")

# =========================
# LOAD SUMMARIZATION MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_summarizer():
    """
    Load the summarization model.
    Falls back to a lighter model if the primary model fails.
    """

    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=model,
                device=-1
            )

            st.success(f"✅ Loaded model: {model}")
            return summarizer

        except Exception as e:
            st.warning(f"⚠️ Could not load {model}: {e}")

    st.error("❌ Failed to load all models.")
    st.stop()


with st.spinner("Loading summarization model..."):
    summarizer = load_summarizer()

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text(pdf_file) -> str:
    """
    Extract text from uploaded PDF file.
    """

    pdf_bytes = pdf_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in doc:
        pages.append(page.get_text("text"))

    doc.close()

    return "\n".join(pages)

# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    """
    Clean extracted text.
    """

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Fix broken words
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================
# SECTION HEADINGS
# =========================
SECTION_HEADINGS = {
    "abstract": [
        "abstract"
    ],

    "introduction": [
        "introduction",
        r"1[\.\s]+introduction"
    ],

    "method": [
        "method",
        "methodology",
        "approach",
        "proposed method",
        r"2[\.\s]+method",
        r"3[\.\s]+method"
    ],

    "results": [
        "results",
        "experiments",
        "evaluation",
        "experimental results",
        r"4[\.\s]+",
        r"5[\.\s]+"
    ],

    "conclusion": [
        "conclusion",
        "conclusions",
        "summary and conclusion"
    ]
}

# =========================
# SECTION EXTRACTION
# =========================
def extract_section_by_heading(
    text: str,
    key: str,
    max_chars: int = 1500
) -> str:
    """
    Extract section text based on heading patterns.
    """

    keywords = SECTION_HEADINGS.get(key, [])

    pattern = (
        r"(?im)"
        r"(?:^|\n)\s*(?:"
        + "|".join(keywords)
        + r")\s*\n"
        r"(.*?)"
        r"(?=\n\s*(?:[A-Z][^\n]{0,80})\n|\Z)"
    )

    match = re.search(pattern, text, re.DOTALL)

    if match:
        return clean_text(match.group(1))[:max_chars]

    # Fallback section extraction
    fallback_ranges = {
        "abstract": (0.00, 0.08),
        "introduction": (0.08, 0.25),
        "method": (0.25, 0.50),
        "results": (0.50, 0.80),
        "conclusion": (0.80, 1.00)
    }

    lo, hi = fallback_ranges.get(key, (0.0, 1.0))

    start = int(len(text) * lo)
    end = int(len(text) * hi)

    return clean_text(text[start:end])[:max_chars]

# =========================
# CHUNK SUMMARIZATION
# =========================
def summarize_chunk(chunk: str) -> str | None:
    """
    Summarize individual text chunk.
    """

    words = chunk.split()
    word_count = len(words)

    if word_count < 30:
        return None

    safe_max = min(120, max(40, word_count // 2))
    safe_min = min(20, safe_max - 10)

    try:
        result = summarizer(
            chunk,
            min_length=safe_min,
            max_length=safe_max,
            do_sample=False
        )

        return result[0]["summary_text"]

    except Exception as e:
        st.warning(f"Chunk summarization failed: {e}")
        return None

# =========================
# FINAL SUMMARIZATION
# =========================
def summarize(
    text: str,
    min_len: int = 40,
    max_len: int = MAX_OUTPUT_TOKENS
) -> str:
    """
    Summarize complete section text.
    """

    if not text or len(text.split()) < 20:
        return "_Not enough content in this section._"

    words = text.split()

    # Split into chunks
    chunks = [
        " ".join(words[i:i + 300])
        for i in range(0, len(words), 300)
    ]

    partial_summaries = []

    # Summarize up to first 4 chunks
    for chunk in chunks[:4]:

        summary = summarize_chunk(chunk)

        if summary:
            partial_summaries.append(summary)

    if not partial_summaries:
        return "_Could not generate a summary for this section._"

    combined = " ".join(partial_summaries)

    combined_words = combined.split()

    # Return directly if already short
    if len(combined_words) < 30:
        return combined

    safe_max = min(
        max_len,
        max(40, len(combined_words) // 2)
    )

    safe_min = min(
        min_len,
        safe_max - 10
    )

    try:
        final_summary = summarizer(
            combined,
            min_length=safe_min,
            max_length=safe_max,
            do_sample=False
        )

        return final_summary[0]["summary_text"]

    except Exception as e:
        st.warning(f"Final summarization failed: {e}")
        return combined

# =========================
# PAPER ANALYSIS
# =========================
def analyze_paper(raw_text: str) -> dict:
    """
    Analyze research paper sections.
    """

    text = clean_text(raw_text)

    abstract = extract_section_by_heading(text, "abstract")
    introduction = extract_section_by_heading(text, "introduction")
    method = extract_section_by_heading(text, "method")
    results = extract_section_by_heading(text, "results")
    conclusion = extract_section_by_heading(text, "conclusion")

    return {
        "📝 Abstract":
            abstract[:700] or "_No abstract found._",

        "📖 Introduction":
            summarize(introduction),

        "🤖 Model / Method":
            summarize(method),

        "📊 Results":
            summarize(results),

        "✅ Conclusion":
            summarize(conclusion),

        "🔍 Overall Summary":
            summarize(
                (abstract + " " + introduction)[:1200]
            )
    }

# =========================
# FILE UPLOAD UI
# =========================
st.divider()

files = st.file_uploader(
    "Upload up to 5 PDF research papers",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# MAIN PROCESSING
# =========================
if files:

    files = files[:5]

    progress_bar = st.progress(
        0,
        text="Starting analysis..."
    )

    for i, file in enumerate(files, start=1):

        progress_bar.progress(
            i / len(files),
            text=f"Analyzing: {file.name}"
        )

        with st.spinner(
            f"Processing {file.name} ({i}/{len(files)})..."
        ):

            try:
                raw_text = extract_text(file)

                if not raw_text.strip():
                    st.error(
                        f"❌ Could not extract text from {file.name}"
                    )
                    continue

                analysis = analyze_paper(raw_text)

            except Exception as e:
                st.error(
                    f"❌ Failed to process {file.name}: {e}"
                )
                continue

        # =========================
        # DISPLAY RESULTS
        # =========================
        with st.expander(
            f"📄 Paper {i}: {file.name}",
            expanded=True
        ):

            cols = st.columns(2)

            items = list(analysis.items())

            for j, (label, content) in enumerate(items):

                with cols[j % 2]:

                    st.markdown(f"### {label}")

                    st.info(content)

    progress_bar.progress(
        1.0,
        text="✅ All papers analyzed successfully!"
    )

else:
    st.info(
        "📂 Upload one or more research papers (PDF) to begin analysis."
    )