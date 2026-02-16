# Gemini Web Research Agent

A Streamlit chat app that uses **Gemini 2.5 Flash** with tools for web research and short-term memory.

## What this repo does

This project provides a chat interface where the assistant can:
- Search the web with DuckDuckGo (`web_search`)
- Read and clean article text from URLs (`read_url`)
- Query Wikipedia for background context
- Ground answers in current date/time (`today`, `now`)
- Keep per-session conversation memory in process memory

The UI is in `app.py` and the agent/tool logic is in `agent.py`.

## Tech stack

- Python
- Streamlit
- LangChain + Google GenAI integration
- Requests + BeautifulSoup for URL extraction
- DuckDuckGo + Wikipedia tools

## Requirements

- Python 3.10+
- A Google Gemini API key (`GOOGLE_API_KEY`)

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## How to use

1. Paste your Gemini API key in the sidebar (`GOOGLE_API_KEY` field).
2. Optionally set:
- `session_id` to separate memory between conversations
- Number of search results
- Verbose traces
3. Click **(Re)crear agente** to rebuild the agent with current settings.
4. Ask questions in the chat box.
5. Use **Reset chat** to clear both visible messages and in-memory history for that `session_id`.

## Project structure

- `app.py`: Streamlit UI, session state, chat loop
- `agent.py`: tool definitions, prompt, memory store, agent builder
- `requirements.txt`: Python dependencies

## Notes and behavior

- Memory is in-process only (`_store` dict in `agent.py`): restarting the app clears it.
- `read_url` strips scripts/styles/nav and returns up to ~12,000 chars of cleaned text.
- `today`/`now` default to `Europe/Madrid` if no timezone is provided.

## Troubleshooting

- If the assistant does not respond, verify API key is valid and the agent is created.
- If web search fails, DuckDuckGo can be rate-limited; retry or provide a direct URL for `read_url`.
- If dependencies fail to resolve, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

## Security

- API key is stored only in Streamlit session state for the browser session.
- Do not commit secrets to git.
