# agent.py
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

    # ✅ conservar espacios
    text = " ".join(soup.get_text(" ").split())
    return text[:12000]


# ---------------- MEMORY STORE ----------------
_store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# ---------------- BUILD AGENT ----------------
def build_agent_with_memory(api_key: str, num_results: int = 8, verbose: bool = False):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.2,
    )

    search = DuckDuckGoSearchResults(num_results=num_results)
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=4000)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant with access to tools.

Available tools:
- DuckDuckGoSearchResults: find relevant links on the web (returns multiple results).
- read_url: open a URL and extract cleaned text for summarizing/citing.
- Wikipedia: stable, definitional or historical background.
- today / now: get the current date/time (useful for time-sensitive questions).

When to use today/now:
- If the user says “today”, “yesterday”, “tomorrow”, “this week”, “latest”, “recent”, “currently”, or asks about deadlines/schedules,
  call today() (or now()) first to ground your answer in the correct date.

Process:
1) If time-sensitive: call today() (or now()) first and mention the date you’re using.
2) Search with DuckDuckGoSearchResults when needed, collect several candidate links.
3) Open 1–2 best links with read_url if the snippet isn’t enough (prefer reputable and/or recent sources).
4) Use Wikipedia for definitions/background when appropriate.
5) Answer clearly and include a short 'Sources:' list with the URLs you actually used/opened."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [search, wikipedia, read_url, today, now]

    # ✅ firma habitual: (llm, tools, prompt)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory
