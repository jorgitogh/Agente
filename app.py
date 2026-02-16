# app.py
import streamlit as st
from agent import build_agent
from agent import build_agent, normalize_to_text


st.set_page_config(page_title="Gemini Web Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Gemini 2.5 Flash — Web Research Agent")
st.caption("DuckDuckGo + Wikipedia + read_url + today/now")

# ---------------- Session state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔐 API Key (Gemini)")
    api_key = st.text_input(
        "Pega tu GOOGLE_API_KEY",
        value=st.session_state.api_key,
        type="password",
        help="Se guarda solo en la sesión del navegador (st.session_state).",
    )
    st.session_state.api_key = api_key

    st.divider()
    st.header("⚙️ Ajustes")
    st.session_state.session_id = st.text_input("session_id (memoria)", st.session_state.session_id)
    num_results = st.slider("Resultados búsqueda", 3, 12, 8)
    verbose = st.toggle("Verbose (trazas)", value=False)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 (Re)crear agente", use_container_width=True, disabled=not api_key):
            st.session_state.agent = build_agent(
                api_key=api_key,
                num_results=num_results,
                verbose=verbose,
                model="gemini-2.5-flash",
            )
            st.success("Agente creado.")
    with col2:
        if st.button("🧹 Reset chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Autocrear agente si hay key y aún no hay agente
if st.session_state.agent is None and st.session_state.api_key:
    st.session_state.agent = build_agent(
        api_key=st.session_state.api_key,
        num_results=8,
        verbose=False,
        model="gemini-2.5-flash",
    )

# ---------------- Render chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ... arriba mantienes el render del historial tal cual ...

disabled_chat = (not st.session_state.api_key) or (st.session_state.agent is None)

if disabled_chat:
    st.info("Introduce tu GOOGLE_API_KEY en la barra lateral para empezar.")
else:
    user_text = st.chat_input("Pregunta algo…")

    if user_text:
        # 1) guarda mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": user_text})

        # 2) calcula respuesta
        try:
            res = st.session_state.agent.invoke(
                {"input": user_text},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            from agent import normalize_to_text  # o impórtalo arriba del todo
            answer = normalize_to_text(res.get("output", res))

        except Exception as e:
            answer = f"⚠️ Error: {type(e).__name__}: {e}"

        # 3) guarda respuesta del assistant
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 4) re-render en limpio (evita duplicados)
        st.rerun()
