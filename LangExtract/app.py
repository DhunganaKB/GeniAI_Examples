import streamlit as st
import langextract as lx
import json
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangExtract Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "extraction_result" not in st.session_state:
    st.session_state["extraction_result"] = None
if "visualization_html" not in st.session_state:
    st.session_state["visualization_html"] = None
if "examples" not in st.session_state:
    st.session_state["examples"] = []
if "results_path" not in st.session_state:
    st.session_state["results_path"] = None
if "loaded_preset" not in st.session_state:
    st.session_state["loaded_preset"] = None

# ---------------------------------------------------------------------------
# PRESET DEFINITIONS
# ---------------------------------------------------------------------------
PRESET_NAMES = [
    "-- Select a preset --",
    "🎭 Character Analysis (Literature)",
    "💊 Medical / Clinical NER",
    "⚖️ Legal Entity Extraction",
    "📝 Customer Feedback Analysis",
    "🔗 Knowledge Graph Entities",
]

PRESETS = {
    "🎭 Character Analysis (Literature)": {
        "prompt": "Extract characters, their emotional states, and relationships in order of appearance.",
        "classes": "character",
        "examples": [
            {
                "text": "ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
                "extractions": [
                    {"extraction_class": "character", "extraction_text": "ROMEO", "attributes": {"emotional_state": "wonder", "role": "protagonist"}},
                    {"extraction_class": "character", "extraction_text": "Juliet", "attributes": {"emotional_state": "referenced", "role": "love interest"}},
                ],
            }
        ],
    },
    "💊 Medical / Clinical NER": {
        "prompt": "Extract medications, dosages, routes of administration, and any adverse reactions mentioned.",
        "classes": "medication, adverse_reaction",
        "examples": [
            {
                "text": "Patient was prescribed Metformin 500mg orally twice daily. Reported mild nausea after first dose.",
                "extractions": [
                    {"extraction_class": "medication", "extraction_text": "Metformin 500mg", "attributes": {"route": "oral", "frequency": "twice daily"}},
                    {"extraction_class": "adverse_reaction", "extraction_text": "mild nausea", "attributes": {"severity": "mild", "timing": "after first dose"}},
                ],
            }
        ],
    },
    "⚖️ Legal Entity Extraction": {
        "prompt": "Extract parties, dates, obligations, and monetary amounts from legal text.",
        "classes": "party, date, obligation, amount",
        "examples": [
            {
                "text": "ACME Corp agrees to pay Widget Inc $50,000 by December 31, 2025 for consulting services.",
                "extractions": [
                    {"extraction_class": "party", "extraction_text": "ACME Corp", "attributes": {"role": "payer"}},
                    {"extraction_class": "party", "extraction_text": "Widget Inc", "attributes": {"role": "payee"}},
                    {"extraction_class": "amount", "extraction_text": "$50,000", "attributes": {"currency": "USD"}},
                    {"extraction_class": "date", "extraction_text": "December 31, 2025", "attributes": {"type": "deadline"}},
                ],
            }
        ],
    },
    "📝 Customer Feedback Analysis": {
        "prompt": "Extract sentiment, product features mentioned, and issues reported from customer reviews.",
        "classes": "feature, issue",
        "examples": [
            {
                "text": "Love the battery life on this phone! But the camera quality in low light is terrible and the app crashes frequently.",
                "extractions": [
                    {"extraction_class": "feature", "extraction_text": "battery life", "attributes": {"sentiment": "positive"}},
                    {"extraction_class": "issue", "extraction_text": "camera quality in low light is terrible", "attributes": {"component": "camera", "severity": "major"}},
                    {"extraction_class": "issue", "extraction_text": "app crashes frequently", "attributes": {"component": "software", "severity": "major"}},
                ],
            }
        ],
    },
    "🔗 Knowledge Graph Entities": {
        "prompt": "Extract entities and their relationships to populate a knowledge graph. Include entity types and relationship labels.",
        "classes": "entity, relationship",
        "examples": [
            {
                "text": "Albert Einstein developed the theory of relativity while working at the Swiss Patent Office in Bern.",
                "extractions": [
                    {"extraction_class": "entity", "extraction_text": "Albert Einstein", "attributes": {"type": "person", "role": "scientist"}},
                    {"extraction_class": "entity", "extraction_text": "theory of relativity", "attributes": {"type": "scientific_theory"}},
                    {"extraction_class": "entity", "extraction_text": "Swiss Patent Office", "attributes": {"type": "organization", "location": "Bern"}},
                    {"extraction_class": "relationship", "extraction_text": "developed", "attributes": {"subject": "Albert Einstein", "object": "theory of relativity", "type": "created"}},
                ],
            }
        ],
    },
}


def load_preset_callback():
    """Called when preset selectbox changes — loads data into session state."""
    choice = st.session_state["preset_selector"]
    if choice != "-- Select a preset --" and choice in PRESETS:
        preset = PRESETS[choice]
        st.session_state["prompt_text"] = preset["prompt"]
        st.session_state["classes_text"] = preset["classes"]
        st.session_state["examples"] = preset["examples"]
        st.session_state["loaded_preset"] = choice


# ---------------------------------------------------------------------------
# Sidebar – global settings
# ---------------------------------------------------------------------------
st.sidebar.title("🔍 LangExtract Studio")
st.sidebar.caption("Structured extraction powered by LLMs with source grounding")

st.sidebar.divider()
st.sidebar.subheader("🔑 API Configuration")

api_provider = st.sidebar.selectbox(
    "Model provider",
    ["Gemini (Google)", "OpenAI", "Ollama (Local)"],
)

if api_provider == "Gemini (Google)":
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=os.getenv("LANGEXTRACT_API_KEY", ""),
        type="password",
        help="Set LANGEXTRACT_API_KEY env var or paste here.",
    )
    model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    fence_output = True
    use_schema = True
    model_url = None
elif api_provider == "OpenAI":
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    model_options = ["gpt-4o", "gpt-4o-mini"]
    fence_output = True
    use_schema = False
    model_url = None
else:  # Ollama
    api_key = None
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    model_options = ["gemma2:2b", "llama3.1:8b", "mistral:7b"]
    fence_output = False
    use_schema = False
    model_url = ollama_url

model_id = st.sidebar.selectbox("Model", model_options, index=0)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Advanced Settings")

extraction_passes = st.sidebar.slider(
    "Extraction passes",
    min_value=1, max_value=5, value=1,
    help="Multiple passes improve recall on dense documents (first-pass-wins merge).",
)
max_workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1, max_value=20, value=1,
    help="Threads for parallel chunk processing. Increase for long documents.",
)
max_char_buffer = st.sidebar.number_input(
    "Chunk size (characters)",
    min_value=0, max_value=100000, value=0, step=500,
    help="0 = no chunking. Set >0 to split long docs into chunks.",
)

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
st.title("🔍 LangExtract Studio")
st.markdown("Extract structured data from unstructured text with LLM-powered source grounding.")

# ===========================================================================
# STEP 1: Quick-start preset OR manual config
# ===========================================================================
st.header("Step 1: Define Your Extraction Task")

st.selectbox(
    "⚡ Quick-start: select a preset to auto-fill everything",
    PRESET_NAMES,
    key="preset_selector",
    on_change=load_preset_callback,
)

# Show confirmation when preset is loaded
if st.session_state.get("loaded_preset"):
    st.success(f"✅ Preset loaded: **{st.session_state['loaded_preset']}** — prompt, classes, and examples are all filled in!")

# Initialize widget keys with defaults if they don't exist
if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = ""
if "classes_text" not in st.session_state:
    st.session_state["classes_text"] = "entity"

# Prompt — uses key so session_state["prompt_text"] drives the value
prompt_value = st.text_area(
    "Extraction prompt — describe what to extract",
    key="prompt_text",
    height=80,
    placeholder="e.g. Extract characters, their emotional states, and relationships in order of appearance.",
)

# Classes
classes_value = st.text_input(
    "Extraction classes (comma-separated)",
    key="classes_text",
    help="e.g. character, medication, adverse_reaction",
)

# Parse classes for use in the example builder
available_classes = [c.strip() for c in classes_value.split(",") if c.strip()] if classes_value else ["entity"]

# ===========================================================================
# STEP 2: Few-Shot Examples
# ===========================================================================
st.header("Step 2: Few-Shot Examples")

if st.session_state["examples"]:
    st.markdown(f"**{len(st.session_state['examples'])} example(s) loaded.**")
    for i, ex in enumerate(st.session_state["examples"]):
        with st.expander(f"📋 Example {i + 1}: _{ex['text'][:80]}..._", expanded=True):
            st.markdown("**Example text:**")
            st.code(ex["text"], language=None)
            st.markdown("**Expected extractions:**")
            for j, ext in enumerate(ex["extractions"]):
                attr_str = ""
                if ext.get("attributes"):
                    attr_str = "  →  " + ", ".join(f"`{k}={v}`" for k, v in ext["attributes"].items())
                st.markdown(f"  {j+1}. **[{ext['extraction_class']}]** `{ext['extraction_text']}`{attr_str}")
            if st.button(f"🗑️ Remove", key=f"rm_{i}"):
                st.session_state["examples"].pop(i)
                st.rerun()

    if st.button("🗑️ Clear all examples"):
        st.session_state["examples"] = []
        st.session_state["loaded_preset"] = None
        st.rerun()
else:
    st.warning("⚠️ No examples loaded yet. **Select a preset above** (easiest) or add one manually below.")

# Manual example builder
with st.expander("➕ Add a new example manually", expanded=not st.session_state["examples"]):
    st.markdown(
        """
        **Instructions:**
        1. Paste a short text passage below
        2. For each entity, fill in the class, the **exact verbatim text** from the passage, and optional attributes
        3. Click "Add this example"
        """
    )

    ex_text = st.text_area(
        "Example text passage",
        height=100,
        key="new_ex_text",
        placeholder='e.g. ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.',
    )

    st.markdown("**Define extractions from this text:**")
    num_extractions = st.number_input("Number of extractions", min_value=1, max_value=20, value=2, key="num_ext")

    new_extractions = []
    for idx in range(int(num_extractions)):
        st.markdown("---")
        cols = st.columns([1, 2, 2])
        with cols[0]:
            ext_class = st.selectbox(f"Class #{idx+1}", available_classes, key=f"ext_class_{idx}")
        with cols[1]:
            ext_text = st.text_input(f"Verbatim text #{idx+1}", key=f"ext_text_{idx}", placeholder="Copy exact text from passage")
        with cols[2]:
            ext_attrs = st.text_input(f"Attributes #{idx+1}", key=f"ext_attrs_{idx}", placeholder="key1=val1, key2=val2")

        if ext_text.strip():
            attrs = {}
            if ext_attrs.strip():
                for pair in ext_attrs.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        attrs[k.strip()] = v.strip()
            new_extractions.append({
                "extraction_class": ext_class,
                "extraction_text": ext_text.strip(),
                "attributes": attrs,
            })

    if st.button("➕ Add this example", type="secondary"):
        if not ex_text or not ex_text.strip():
            st.error("Please enter an example text passage.")
        elif not new_extractions:
            st.error("Please fill in at least one extraction with verbatim text.")
        else:
            bad = [e for e in new_extractions if e["extraction_text"] not in ex_text]
            if bad:
                for e in bad:
                    st.warning(f"⚠️ `{e['extraction_text']}` not found verbatim in the example text!")
                st.error("Fix the above — extraction text must be an **exact copy** from the passage.")
            else:
                st.session_state["examples"].append({"text": ex_text.strip(), "extractions": new_extractions})
                st.toast("Example added!")
                st.rerun()


# ===========================================================================
# STEP 3: Input document
# ===========================================================================
st.header("Step 3: Input Document to Extract From")

input_method = st.radio("Input source", ["📝 Paste text", "🌐 URL", "📁 Upload file"], horizontal=True)

input_text = None
if input_method == "📝 Paste text":
    input_text = st.text_area("Paste your document text", height=250, placeholder="Paste the text you want to extract from...")
elif input_method == "🌐 URL":
    input_text = st.text_input(
        "Document URL",
        placeholder="https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
        help="Supports any plain-text URL. LangExtract will download and process it.",
    )
else:
    uploaded = st.file_uploader("Upload a text file", type=["txt", "md", "csv"])
    if uploaded is not None:
        input_text = uploaded.read().decode("utf-8", errors="replace")
        st.text_area("Preview (first 2000 chars)", input_text[:2000], height=200, disabled=True)


# ===========================================================================
# STEP 4: Run
# ===========================================================================
st.header("Step 4: Run Extraction")

prompt_filled = bool(prompt_value and prompt_value.strip())
has_examples = bool(st.session_state["examples"])
has_input = bool(input_text and input_text.strip())
has_key = bool(api_key) if api_provider != "Ollama (Local)" else True

checks = [
    ("API key configured", has_key),
    ("Extraction prompt filled", prompt_filled),
    ("At least one few-shot example", has_examples),
    ("Input document provided", has_input),
]

for label, ok in checks:
    st.markdown(f"{'✅' if ok else '❌'} {label}")

all_ready = all(ok for _, ok in checks)

if not all_ready:
    st.warning("Complete all the steps above before running extraction.")


def build_lx_examples():
    lx_examples = []
    for ex in st.session_state["examples"]:
        extractions = [
            lx.data.Extraction(
                extraction_class=e["extraction_class"],
                extraction_text=e["extraction_text"],
                attributes=e.get("attributes", {}),
            )
            for e in ex["extractions"]
        ]
        lx_examples.append(lx.data.ExampleData(text=ex["text"], extractions=extractions))
    return lx_examples


if st.button("🚀 Run Extraction", type="primary", disabled=not all_ready, use_container_width=True):
    lx_examples = build_lx_examples()

    extract_kwargs = {
        "text_or_documents": input_text,
        "prompt_description": prompt_value,
        "examples": lx_examples,
        "model_id": model_id,
        "extraction_passes": extraction_passes,
        "max_workers": max_workers,
        "fence_output": fence_output,
        "use_schema_constraints": use_schema,
    }

    if api_key:
        extract_kwargs["api_key"] = api_key
    if model_url:
        extract_kwargs["model_url"] = model_url
    if max_char_buffer > 0:
        extract_kwargs["max_char_buffer"] = max_char_buffer

    with st.spinner("🔄 Running extraction... this may take a moment."):
        try:
            result = lx.extract(**extract_kwargs)
            st.session_state["extraction_result"] = result

            tmpdir = tempfile.mkdtemp()
            output_path = os.path.join(tmpdir, "results.jsonl")
            lx.io.save_annotated_documents([result], output_dir=tmpdir, output_name="results.jsonl")
            st.session_state["results_path"] = output_path

            try:
                html = lx.visualize(output_path)
                st.session_state["visualization_html"] = html
            except Exception:
                st.session_state["visualization_html"] = None

            st.success("🎉 Extraction complete! Scroll down to see results.")
            st.rerun()
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.exception(e)


# ===========================================================================
# RESULTS
# ===========================================================================
result = st.session_state.get("extraction_result")
if result is not None:
    st.divider()
    st.header("📊 Extraction Results")

    extractions = list(result.extractions)

    class_counts = {}
    for ext in extractions:
        class_counts[ext.extraction_class] = class_counts.get(ext.extraction_class, 0) + 1

    metric_cols = st.columns(min(len(class_counts) + 1, 5))
    metric_cols[0].metric("Total Extractions", len(extractions))
    for i, (cls, count) in enumerate(class_counts.items()):
        metric_cols[(i + 1) % len(metric_cols)].metric(cls, count)

    all_classes = sorted(class_counts.keys())
    selected_classes = st.multiselect("Filter by class", all_classes, default=all_classes)

    for ext in extractions:
        if ext.extraction_class not in selected_classes:
            continue
        with st.expander(f"**[{ext.extraction_class}]** {ext.extraction_text[:120]}"):
            st.markdown(f"**Class:** `{ext.extraction_class}`")
            st.markdown(f"**Extracted Text:** {ext.extraction_text}")
            if ext.attributes:
                st.markdown("**Attributes:**")
                st.json(ext.attributes)

    st.subheader("📥 Download Results")
    dl_cols = st.columns(3)

    results_path = st.session_state.get("results_path")
    if results_path and os.path.exists(results_path):
        with open(results_path, "r") as f:
            jsonl_data = f.read()
        dl_cols[0].download_button("Download JSONL", data=jsonl_data, file_name="langextract_results.jsonl", mime="application/jsonl")

    json_data = [{"class": e.extraction_class, "text": e.extraction_text, "attributes": e.attributes} for e in extractions]
    dl_cols[1].download_button("Download JSON", data=json.dumps(json_data, indent=2), file_name="langextract_results.json", mime="application/json")

    # Visualization
    st.divider()
    st.header("🎨 Interactive Visualization")
    st.markdown("Entities highlighted in their original text with source grounding (character offsets).")

    html_content = st.session_state.get("visualization_html")
    if html_content:
        st.components.v1.html(html_content, height=800, scrolling=True)
        dl_cols[2].download_button("Download HTML Visualization", data=html_content, file_name="langextract_visualization.html", mime="text/html")
    else:
        st.info("Visualization could not be generated for this result.")
