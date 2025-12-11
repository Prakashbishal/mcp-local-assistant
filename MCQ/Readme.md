
# 📘 MCQ Answering MCP Server

*A FastMCP-based server for automated MCQ extraction & answer generation from university exam PDFs.*

---

## 🔧 Overview

This MCP server provides a fully local workflow to:

1. **Import exam PDFs/PPTX** into the server
2. **Extract readable text** into `.txt` format
3. **Feed the text to a local LLM via Ollama**
4. **Automatically generate MCQ answers**
5. **Store results in `data/answers/`**

This server is intentionally **stand-alone**, separate from your Study Assistant MCP.
Its only purpose is **MCQ → Answer prediction**.

---

## 📂 Directory Structure

```
project-root/
│
├── mcq_answer_mcp_server.py     # FastMCP server implementation
│
└── data/
    ├── pyq_raw/                  # PDFs / PPTX input files
    ├── pyq_txt/                  # Extracted text files
    └── answers/                  # Final answer outputs
```

---

## 🚀 Features

### ✔ 1. Safe File Handling

All user-supplied filenames are sanitized using internal `_safe_*_file()` helpers.

### ✔ 2. PDF → TXT Extraction

Uses `pypdf` to extract text page-by-page.

### ✔ 3. PPTX → TXT Extraction

Uses `python-pptx` to extract slide text.

### ✔ 4. Smart Section A Extraction

Extracts only the *real* MCQ section using a robust `rfind("Section A")` rule.

### ✔ 5. MCQ Answer Generation

* Sends **only Section A** to the model
* Enforces **20 answers in strict `1: A` format**
* Cleans output automatically
* Rejects trivial patterns
* Uses deterministic model settings (`temperature = 0`)

### ✔ 6. Local LLM via Ollama

Compatible with models like:

```
llama3.1:8b
llama3.1:70b
mistral-nemo
```

---

## 🛠️ Requirements

Install core dependencies:

```bash
pip install fastmcp ollama pypdf python-pptx
```

Ensure **Ollama** is installed and running:

```bash
ollama pull llama3.1:8b
```

---

## ▶️ Running the MCP Server

Start the server:

```bash
python mcq_answer_mcp_server.py
```

Your MCP client (Claude Desktop, Cursor, custom agent, etc.) will use `stdio` transport to communicate.

---

## 📌 Available Tools

### 1. `list_pyq_raw_files()`

List files in `data/pyq_raw/`.

### 2. `extract_pyq_pdf_to_txt(filename, output_name)`

Convert a PDF into a `.txt` file under `data/pyq_txt/`.

### 3. `extract_pyq_pptx_to_txt(filename, output_name)`

Convert a PowerPoint file into `.txt`.

### 4. `list_pyq_txt_files()`

List extracted txt files.

### 5. `answer_mcq_file(pyq_txt_filename, model="llama3.1:8b", output_filename=None)`

Reads the TXT file → Extracts Section A → Sends it to LLM → Produces answers.

### 6. `debug_preview_file(pyq_txt_filename, max_chars=3000)`

Preview part of a `.txt` file inside the MCP environment.

---

## 📘 Example Usage

### 1. Extract from PDF

```json
{
  "tool": "extract_pyq_pdf_to_txt",
  "args": {
    "filename": "COMP6246.pdf",
    "output_name": "ml_question"
  }
}
```

### 2. Preview TXT (debug)

```json
{
  "tool": "debug_preview_file",
  "args": { "pyq_txt_filename": "ml_question.txt" }
}
```

### 3. Generate MCQ Answers

```json
{
  "tool": "answer_mcq_file",
  "args": {
    "pyq_txt_filename": "ml_question.txt",
    "model": "llama3.1:8b"
  }
}
```

### Output Example

```
Answers written to answers_for_ml_question.txt.

1: B
2: C
3: D
4: D
5: A
6: A
7: B
8: D
9: A
10: D
11: B
12: D
13: C
14: D
15: C
16: C
17: A
18: B
19: D
20: A
```

---

## 🔒 Safety Notes

* All file operations use **strict path sanitization** to prevent traversal.
* The model receives **no study materials**, only the question text.
* Output is deterministic, clean, and post-processed to avoid model drift.

---

## 🧩 Future Enhancements (optional)

* Per-question LLM calls (higher accuracy)
* Automatic detection of MCQ counts
* Integration with a test suite to validate OCR correctness
* Support for more exam formats

