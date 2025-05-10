# app.py
"""
End‑to‑end PDF → text → LLM‑structured JSON → "textified" JSON pipeline.

Folders created on first run:
    data/
        text_outputs/
        json_outputs/
        textified_chunks/
    prompts/               # put your prompt .txt files here
"""

import os
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import openai


# ────────────────────────────────────────────────────────────
# 0.  Environment + OpenAI wrapper
# ────────────────────────────────────────────────────────────
load_dotenv()                              # looks for .env in project root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)    # Initialize client


def call_gpt(system_prompt: str, user_prompt: str, model: str = "gpt-4o-2024-08-06") -> list[dict]:
    """
    Wrapper around OpenAI ChatCompletion for any GPT model.
    Expects the model to return valid JSON (list of objects).
    
    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        model: The model to use (defaults to gpt-4o-mini)
    
    Returns:
        list[dict]: The parsed JSON response
        
    Raises:
        ValueError: If the model is invalid or not available
        json.JSONDecodeError: If the response is not valid JSON
        openai.APIError: For other API-related errors
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print("\n=== Raw Model Response ===")
            print(content)
            print("========================\n")
            raise ValueError(f"Model response was not valid JSON: {str(e)}\nRaw response printed to console.")
    except openai.NotFoundError:
        raise ValueError(f"Model '{model}' not found or not available")
    except openai.APIError as e:
        raise ValueError(f"OpenAI API error: {str(e)}")


# ────────────────────────────────────────────────────────────
# 1.  PDF‑to‑Text helper
# ────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")
TEXT_DIR          = DATA_DIR / "text_outputs"
JSON_DIR          = DATA_DIR / "json_outputs"
TEXTIFIED_DIR     = DATA_DIR / "textified_chunks"
PROMPT_DIR        = Path("prompts")

for d in (TEXT_DIR, JSON_DIR, TEXTIFIED_DIR, PROMPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def pdf_to_clean_text(pdf_path: Path) -> str:
    """Extracts raw text from PDF, does light cleanup."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    raw = "\n".join(pages)

    # Very light cleanup; extend if you like
    cleaned = "\n".join(line.strip() for line in raw.splitlines() if line.strip())

    return cleaned


def save_overwriting(folder: Path, base_name: str, ext: str, data: str | bytes) -> Path:
    """
    Saves <base_name>.<ext> inside <folder>, overwriting any previous file
    that starts with <base_name>.
    """
    # Remove old versions
    for old in folder.glob(f"{base_name}*{ext}"):
        old.unlink()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    new_path = folder / f"{base_name}_{timestamp}{ext}"
    mode = "w" if isinstance(data, str) else "wb"
    with open(new_path, mode) as f:
        f.write(data)
    return new_path


# ────────────────────────────────────────────────────────────
# 2.  JSON → "textified" JSON helper
# ────────────────────────────────────────────────────────────
def textify_structured_json(sections: list[dict]) -> list[str]:
    """
    Turns structured transcript JSON into a list of human‑readable strings
    (one per section) with metadata tags at the top.
    """
    chunks = []
    for sec in sections:
        header_lines = [
            f"company: {sec.get('company', '')}",
            f"quarter: {sec.get('quarter', '')}",
            f"type: {sec.get('type', '')}",
            f"section_type: {sec.get('section_type', '')}",
            f"speakers: {sec.get('speakers', '')}",
            "",  # blank line before content
        ]
        chunk = "\n".join(header_lines) + sec.get("content", "")
        chunks.append(chunk)
    return chunks


# ────────────────────────────────────────────────────────────
# 3.  Streamlit UI
# ────────────────────────────────────────────────────────────
st.title("PDF → GPT → Structured JSON → Textified JSON")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    pdf_name = Path(uploaded_pdf.name).stem
    pdf_temp_path = DATA_DIR / uploaded_pdf.name
    pdf_temp_path.write_bytes(uploaded_pdf.getbuffer())

    st.success(f"📄 Received **{uploaded_pdf.name}**")

    # 3.1  Extract text & save
    if st.button("Extract text"):
        text = pdf_to_clean_text(pdf_temp_path)
        text_path = save_overwriting(TEXT_DIR, pdf_name, ".txt", text)
        st.info(f"✅ Text saved to **{text_path.name}**")
        st.text_area("Preview (first 2000 chars)", text[:2000], height=250)

    # 3.2  LLM processing
    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    if prompt_files:
        prompt_choice = st.selectbox(
            "Prompt file to use",
            options=[p.name for p in prompt_files],
            index=0,
        )
        if st.button("Run GPT"):
            text_file_candidates = list(TEXT_DIR.glob(f"{pdf_name}*.txt"))
            if not text_file_candidates:
                st.error("🛑 No extracted text file found. Run 'Extract text' first.")
            else:
                # Use the most recent text file
                text_file = max(text_file_candidates, key=os.path.getmtime)
                raw_text = text_file.read_text()
                system_prompt = (PROMPT_DIR / prompt_choice).read_text()
                st.write("🔄 Calling GPT… (this may take a bit)")

                try:
                    structured = call_gpt(system_prompt, raw_text)
                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                else:
                    json_path = save_overwriting(JSON_DIR, pdf_name, ".json",
                                                 json.dumps(structured, indent=2))
                    st.success(f"✅ Structured JSON saved to **{json_path.name}**")
                    st.json(structured[:2])  # show first two sections as sanity check

    # 3.3  Textify JSON
    if st.button("Textify latest structured JSON"):
        json_candidates = list(JSON_DIR.glob(f"{pdf_name}*.json"))
        if not json_candidates:
            st.error("🛑 No structured JSON found. Run GPT step first.")
        else:
            json_file = max(json_candidates, key=os.path.getmtime)
            sections = json.loads(json_file.read_text())
            chunks = textify_structured_json(sections)
            textified_path = save_overwriting(TEXTIFIED_DIR, pdf_name, ".json",
                                              json.dumps(chunks, indent=2))
            st.success(f"✅ Textified JSON saved to **{textified_path.name}**")
            st.write("First chunk preview:")
            st.code(chunks[0][:1000] + ("…" if len(chunks[0]) > 1000 else ""))

st.markdown("---")
st.caption("Basic prototype • overwrites existing files with same base name.")