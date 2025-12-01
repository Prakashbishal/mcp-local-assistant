# MCP Local Assistant

A small Model Context Protocol (MCP) server that exposes useful tools
for working with local files and datasets.

It is designed as a lightweight **AI-powered local assistant**:
the tools can be called from MCP clients like the MCP Inspector,
ChatGPT Desktop, or Claude Desktop.

---

## Features

- File utilities:
  - `write_note(filename, text)` – append text to a file in `data/`
  - `list_data_files()` – list files in `data/`
  - `read_text_file(filename)` – read file content
  - `delete_file(filename)` – delete a file in `data/`
  - `search_in_file(filename, query)` – search within one file
  - `search_in_all_files(query)` – search across all text files
  - `summarize_text_file(filename, max_chars)` – simple text summary

- CSV helpers (no external dependencies):
  - `describe_dataset(filename)` – rows, columns, missing values
  - `preview_dataset(filename, n)` – first `n` rows as JSON-friendly dicts

- Utility tools:
  - `greet(name)` – simple greeting
  - `current_time()` – current local time

All tools are typed and exposed via MCP so that an LLM can discover and call them.

---

## Requirements

- Python 3.10+
- `pip` for installing dependencies

Install Python dependencies with:

```bash
pip install -r requirements.txt
