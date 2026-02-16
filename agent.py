# agent.py
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


# ---------------- OUTPUT NORMALIZER ----------------
def normalize_to_text(x) -> str:
    """Convierte salidas tipo LangChain/blocks a texto plano."""
    if isinstance(x, str):
        return x

    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(x, dict):
        if "output" in x:
            return normalize_to_text(x["output"])
        if "text" in x and isinstance(x["text"], str):
            return x["text"]

    return str(x)


# ---------------- TOOLS ----------------
@tool
def today(tz: str = "Europe/Madrid") -> str:
    """Devuelve la fecha de hoy (YYYY-MM-DD) en la zona horaria indicada."""
    return datetime.now(ZoneInfo(tz)).date().isoformat()


@tool
def now(tz: str = "Europe/Madrid") -> str:
    """Devuelve fecha y hora actual (ISO 8601) en la zona horaria indicada."""
    return datetime.now(ZoneInfo(tz)).isoformat()


@tool
def read_url(url: str) -> str:
    """Descarga una URL y devuelve el texto principal (limpio) para poder resumir/citar."""
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text(" ").split())
    return text[:12000]


# ✅ DuckDuckGo "robusto": backend lite + captura de JSONDecodeError
_ddg = DuckDuckGoSearchResults(num_results=8, backend="lite")

@tool
def web_search(query: str) -> str:
    """Busca en la web con DuckDuckGo. Devuelve resultados (texto) aunque DDG falle."""
    try:
        # DuckDuckGoSearchResults en LC suele devolver str con resultados formateados
        return _ddg.invoke(query)
    except json.JSONDecodeError:
        # típico cuando DDG devuelve vacío / rate-limit / bloqueo
        return (
            "DuckDuckGo devolvió una respuesta vacía/no-JSON (posible bloqueo o rate limit). "
            "Prueba otra consulta, espera un poco, o usa Wikipedia/read_url con una URL directa."
        )
    except Exception as e:
        return f"Error en DuckDuckGo: {type(e).__name__}: {e}"


# ---------------- MEMORY STORE ----------------
_store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# ---------------- BUILD AGENT ----------------
def build_agent_with_memory(
    api_key: str,
    num_results: int = 8,
    verbose: bool = False,
    model: str = "gemini-2.5-flash",
):
    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=0.2,
    )

    # Wikipedia tool
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=4000)
    )

    # Ajusta el num_results también en el ddg global (sin recrear el tool de LC)
    global _ddg
    _ddg = DuckDuckGoSearchResults(num_results=num_results, backend="lite")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant with access to tools.

Available tools:
- web_search: find relevant links on the web (DuckDuckGo).
- read_url: open a URL and extract cleaned text for summarizing/citing.
- Wikipedia: stable, definitional or historical background.
- today / now: get the current date/time (useful for time-sensitive questions).

When to use today/now:
- If the user says “today”, “yesterday”, “tomorrow”, “this week”, “latest”, “recent”, “currently”, or asks about deadlines/schedules,
  call today() (or now()) first to ground your answer in the correct date.

Process:
1) If time-sensitive: call today() (or now()) first and mention the date you’re using.
2) Search with web_search when needed, collect several candidate links.
3) Open 1–2 best links with read_url if the snippet isn’t enough (prefer reputable and/or recent sources).
4) Use Wikipedia for definitions/background when appropriate.
5) Answer clearly and include a short 'Sources:' list with the URLs you actually used/opened."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # ✅ usamos web_search en vez de DuckDuckGoSearchResults directo
    tools = [web_search, wikipedia, read_url, today, now]

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory
