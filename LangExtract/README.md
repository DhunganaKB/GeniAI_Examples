<div align="center">

# 🔍 LangExtract Studio

**A Streamlit-powered UI for Google's [LangExtract](https://github.com/google/langextract) library**

*Extract structured data from unstructured text with LLM-powered source grounding — no coding required.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangExtract](https://img.shields.io/badge/LangExtract-1.1+-4285f4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/langextract)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)

[Getting Started](#-getting-started) · [Features](#-features) · [Medical Pipeline](#-medical-extraction-pipeline) · [How It Works](#-how-it-works) · [Presets](#-built-in-presets) · [Resources](#-learn-more) · [Article](#-read-the-article)

</div>

---

## What is This?

**LangExtract Studio** is a web application that wraps Google's [LangExtract](https://github.com/google/langextract) library in a guided, step-by-step Streamlit interface. It lets you:

- Extract structured entities from **any text** (clinical notes, legal contracts, literature, customer reviews)
- Get **source-grounded** results — every extraction is mapped to its exact character position in the original document
- Use **pre-built templates** or define your own custom extraction schema
- **Visualize** results with LangExtract's interactive HTML highlighting
- **Export** results as JSONL or JSON

All with **zero Python coding** — just select a preset, paste your text, and click run.

It also includes a **standalone medical extraction pipeline** (`medical_extraction.py`) that demonstrates clinical NER — extracting medications, diagnoses, adverse reactions, vital signs, lab results, and procedures from clinical notes.

<div align="center">
<img src="images/architecture_diagram.svg" alt="Architecture Diagram" width="90%">
<br>
<em>How LangExtract works: from raw input to source-grounded, verifiable output</em>
</div>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Gemini API key ([get one free here](https://aistudio.google.com/apikey)), OR an OpenAI API key, OR a local Ollama installation

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/langextract-studio.git
cd langextract-studio

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your key:
#   LANGEXTRACT_API_KEY=your-gemini-api-key-here

# Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Docker (Alternative)

```bash
docker build -t langextract-studio .
docker run -p 8501:8501 -e LANGEXTRACT_API_KEY="your-key" langextract-studio
```

---

## ✨ Features

### 🎯 One-Click Presets

Select from 5 built-in extraction templates. One click auto-fills the prompt, entity classes, and few-shot examples.

### 🔗 Source Grounding

Every extracted entity links back to its **exact character offset** in the source document. No hallucinations — if the model extracts it, you can verify it exists in the text.

### 🤖 Multi-Provider Support

| Provider | Models | Setup |
|---|---|---|
| **Gemini** (recommended) | gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash | `LANGEXTRACT_API_KEY` env var |
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` env var |
| **Ollama** (local) | gemma2:2b, llama3.1:8b, mistral:7b | Ollama running locally |

Provider-specific settings (`fence_output`, `use_schema_constraints`, `model_url`) are handled automatically.

### 📊 Advanced Extraction Settings

| Setting | What It Does |
|---|---|
| **Extraction Passes** (1-5) | Multiple passes with first-pass-wins merge for better recall |
| **Parallel Workers** (1-20) | Concurrent threads for processing document chunks |
| **Chunk Size** | Paragraph-aware document splitting for long texts |

### 📥 Multiple Input & Output Formats

**Input:** Paste text, URL (e.g. Project Gutenberg), or file upload (.txt, .md, .csv)

**Output:** Metrics dashboard, filterable results, interactive HTML visualization, JSONL export, JSON export

---

## 🔄 How It Works

<div align="center">
<img src="images/streamlit_workflow.svg" alt="4-Step Workflow" width="90%">
</div>

### Step 1: Define Your Extraction Task

Select a preset or manually define:
- **Extraction prompt** — natural language description of what to extract
- **Entity classes** — categories like `medication`, `character`, `party`

### Step 2: Few-Shot Examples

Teach the model with 1-2 examples. Each example has:
- A **text passage**
- **Expected extractions** — with class, verbatim text (must be exact copy from passage), and optional attributes

> **Why verbatim?** This is what powers source grounding. The library matches extracted text back to exact positions in the source document via string matching.

### Step 3: Input Your Document

Feed the text you want to extract from:
- **Paste** — for quick experiments
- **URL** — point at any plain-text URL (entire novels from Project Gutenberg, etc.)
- **Upload** — local .txt, .md, or .csv files

### Step 4: Run & Explore

A pre-flight checklist confirms everything is ready. Hit run and get:
- **Metrics** — total extractions, per-class counts
- **Results** — filterable, expandable cards with class, text, and attributes
- **Visualization** — LangExtract's interactive HTML with source-grounded highlighting
- **Export** — JSONL (native format) or JSON

---

## 📋 Built-in Presets

| Preset | Classes | Use Case |
|---|---|---|
| 🎭 **Character Analysis** | `character` | Extract characters, emotions, relationships from literary texts |
| 💊 **Medical / Clinical NER** | `medication`, `adverse_reaction` | Extract medications, dosages, adverse reactions from clinical notes |
| ⚖️ **Legal Entity Extraction** | `party`, `date`, `obligation`, `amount` | Extract parties, deadlines, amounts from contracts |
| 📝 **Customer Feedback** | `feature`, `issue` | Extract features (with sentiment) and issues (with severity) from reviews |
| 🔗 **Knowledge Graph** | `entity`, `relationship` | Extract entities and relationships for graph databases |

<div align="center">
<img src="images/use_cases.svg" alt="Use Cases" width="90%">
</div>

---

## 🆚 Why LangExtract?

<div align="center">
<img src="images/comparison_table.svg" alt="Comparison" width="90%">
</div>

**vs. Regex/spaCy:** No weeks of rules or retraining — just 1-2 examples. Supports any custom entity type.

**vs. Raw LLM prompting:** Source grounding proves extractions are real (not hallucinated). Handles long documents via auto-chunking. Schema-constrained output format.

**Best of both worlds:** LLM flexibility + traditional NLP reliability.

---

## 💊 Medical Extraction Pipeline

The repo includes a **standalone medical NER module** (`medical_extraction.py`) that extracts 6 entity types from clinical text — no Streamlit required:

| Entity Class | What It Extracts | Example Attributes |
|---|---|---|
| `medication` | Drug names with context | route, frequency, indication |
| `adverse_reaction` | Side effects | severity, timing, related medication |
| `diagnosis` | Medical conditions | status (active/resolved/suspected) |
| `vital_sign` | Vitals measurements | measurement, value, status |
| `lab_result` | Lab test results | test, value, reference range, status |
| `procedure` | Procedures performed | timing |

### Run It

```bash
export LANGEXTRACT_API_KEY="your-gemini-api-key"
python medical_extraction.py
```

### What It Produces

Tested on 3 sample clinical notes (ED visit, pediatric asthma, ICU sepsis):

```
Total clinical notes processed: 3
Total entities extracted:       47

Entities by class:
    diagnosis               15
    medication              12
    vital_sign              11
    lab_result               6
    procedure                2
    adverse_reaction         1
```

**Output files:**

```
output/
├── medical_extractions.json        ← Combined structured results (all notes)
├── note_001_extractions.jsonl      ← LangExtract native format (per note)
```

### Sample Structured Output

```json
{
  "class": "medication",
  "text": "Meropenem 1g",
  "attributes": {
    "route": "IV",
    "frequency": "every 8 hours",
    "indication": "sepsis"
  }
}
```

> **Why healthcare?** Source grounding is critical in clinical settings — every extracted medication or diagnosis must be traceable back to the exact text in the clinical note. LangExtract provides this by mapping extractions to character offsets.

See Google's own healthcare demo: [**RadExtract** on HuggingFace](https://huggingface.co/spaces/google/radextract)

---

## 📁 Project Structure

```
langextract-studio/
├── app.py                    # Streamlit web UI (~490 lines)
├── medical_extraction.py     # Standalone medical NER pipeline
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── output/                   # Generated extraction results
│   ├── medical_extractions.json
│   ├── note_*_extractions.jsonl
│   └── note_*_visualization.html

```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LANGEXTRACT_API_KEY` | Yes (for Gemini) | Google Gemini API key |
| `OPENAI_API_KEY` | Yes (for OpenAI) | OpenAI API key |

### Sidebar Settings

All model and extraction settings are configurable via the sidebar:

- **Model provider** — Gemini, OpenAI, or Ollama
- **Model** — specific model within the provider
- **Extraction passes** — 1 to 5 (more passes = better recall)
- **Parallel workers** — 1 to 20 (for long document throughput)
- **Chunk size** — 0 (no chunking) to 100,000 characters

---

## 📝 Quick Example: Extract from Romeo & Juliet

1. Launch the app: `streamlit run app.py`
2. Select **🎭 Character Analysis (Literature)** from the preset dropdown
3. In Step 3, select **🌐 URL** and paste:
   ```
   https://www.gutenberg.org/files/1513/1513-0.txt
   ```
4. In the sidebar, set:
   - **Extraction passes:** 3
   - **Parallel workers:** 10
   - **Chunk size:** 1000
5. Click **🚀 Run Extraction**
6. Explore the results — every character extracted with emotions and relationships, source-grounded to exact positions in the text

---

## 🐍 Using LangExtract Directly (Python API)

If you prefer the Python API over the Streamlit UI:

```python
import langextract as lx

# Define what to extract with a few-shot example
examples = [
    lx.data.ExampleData(
        text="Patient was prescribed Metformin 500mg orally twice daily.",
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Metformin 500mg",
                attributes={"route": "oral", "frequency": "twice daily"}
            ),
        ]
    )
]

# Run extraction
result = lx.extract(
    text_or_documents="Your clinical notes here...",
    prompt_description="Extract medications and dosages.",
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Access results
for ext in result.extractions:
    print(f"[{ext.extraction_class}] {ext.extraction_text} — {ext.attributes}")

# Save and visualize
lx.io.save_annotated_documents([result], output_dir="./output", output_name="results.jsonl")
html = lx.visualize("./output/results.jsonl")
```

---

## 📰 Read the Article

For a deep-dive into how LangExtract works, why source grounding matters, and lessons learned building this app:

- **Markdown:** [`ARTICLE.md`](ARTICLE.md)
- **Styled HTML:** [`article.html`](article.html) — open in your browser for the full experience with images

---

## 📚 Learn More

### Official Resources
- [LangExtract GitHub](https://github.com/google/langextract) — source code, docs, examples
- [Google Developers Blog — Introducing LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/libraries/langextract)
- [PyPI Package](https://pypi.org/project/langextract/)
- [Zenodo DOI: 10.5281/zenodo.17015089](https://zenodo.org/doi/10.5281/zenodo.17015089)

### Live Demos
- [RadExtract on HuggingFace](https://huggingface.co/spaces/google/radextract) — radiology report structuring
- [RadExtract Direct Link](https://google-radextract.hf.space/)

### Tutorials & Coverage
- [DataCamp Tutorial](https://www.datacamp.com/tutorial/langextract)
- [InfoQ Coverage](https://www.infoq.com/news/2025/08/google-langextract-python/)
- [MarkTechPost Coverage](https://www.marktechpost.com/2025/08/04/google-ai-releases-langextract-an-open-source-python-library-that-extracts-structured-data-from-unstructured-text-documents/)

### Example Docs from Google
- [Medication Extraction Examples](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)
- [Longer Text Example](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)
- [Batch API Example](https://github.com/google/langextract/blob/main/docs/examples/batch_api_example.md)
- [Japanese Extraction](https://github.com/google/langextract/blob/main/docs/examples/japanese_extraction.md)
- [Romeo & Juliet Notebook](https://github.com/google/langextract/blob/main/examples/notebooks/romeo_juliet_extraction.ipynb)
- [Custom Provider Plugin](https://github.com/google/langextract/tree/main/examples/custom_provider_plugin)
- [Ollama Integration](https://github.com/google/langextract/tree/main/examples/ollama)

---

## 🤝 Contributing

Contributions are welcome! Some ideas:

- [ ] Add more preset templates (e.g., resume parsing, scientific paper extraction)
- [ ] Batch processing mode for multiple documents
- [ ] Save/load custom presets to file
- [ ] Side-by-side comparison of different model outputs
- [ ] Integration with Vertex AI batch processing

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

LangExtract is an open-source library by Google, also under Apache 2.0. This project is **not** an official Google product.

---

## 🙏 Acknowledgments

- [Google LangExtract](https://github.com/google/langextract) — the core extraction library
- [Streamlit](https://streamlit.io) — the web framework
- [Project Gutenberg](https://www.gutenberg.org) — free literary texts for testing

---

<div align="center">

**Built with LangExtract + Streamlit**

⭐ Star this repo if you found it useful!

</div>
