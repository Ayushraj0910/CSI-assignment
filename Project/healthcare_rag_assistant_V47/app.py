import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import DATASET_PATH, GROQ_API_KEY
from src.database import PatientDatabase
from src.llm import GroqLLM
from src.router import Orchestrator
from src.sql_agent import SQLAgent
from src.rag_agent import RAGAgent

st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Dashboard styling
# -----------------------------
st.markdown(
    """
    <style>
        :root {
            --green-950: #0d4f3b;
            --green-900: #126247;
            --green-800: #167653;
            --green-700: #229361;
            --green-600: #36aa70;
            --green-200: #cfe9d9;
            --green-100: #e8f5ed;
            --green-50: #f4faf6;
            --ink: #18382c;
            --muted: #63796e;
            --line: #d8e9df;
        }

        .stApp {
            background:
                radial-gradient(circle at 91% 11%, rgba(177, 224, 194, .38), transparent 20%),
                radial-gradient(circle at 70% 86%, rgba(213, 240, 222, .30), transparent 24%),
                linear-gradient(135deg, #f8fcf9 0%, #eff8f2 52%, #f8fcfa 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] { background: rgba(248,252,249,.82); }
        [data-testid="stToolbar"] { color: var(--green-800); }

        [data-testid="stSidebar"] {
            width: 255px !important;
            background: linear-gradient(180deg, #f5fbf7 0%, #edf7f1 100%);
            border-right: 1px solid #d7e9de;
        }
        [data-testid="stSidebar"] * { color: var(--ink); }
        [data-testid="stSidebar"] hr { margin: 1rem 0; border-color: #dbe9df !important; }

        .sidebar-brand {
            display:flex; align-items:center; gap:10px; margin:4px 0 12px;
        }
        .brand-heart {
            width:42px; height:42px; border-radius:14px;
            display:flex; align-items:center; justify-content:center;
            background:linear-gradient(145deg,#d8f0df,#bfe5ce);
            color:var(--green-800); font-size:23px;
            box-shadow:0 5px 14px rgba(40,126,82,.10);
        }
        .brand-title { color:var(--green-950); font-size:1.12rem; font-weight:850; line-height:1.08; }
        .brand-subtitle { color:#6b8176; font-size:.76rem; margin-top:5px; }

        .api-status {
            display:inline-flex; align-items:center; gap:6px;
            padding:5px 10px; border-radius:999px;
            background:#dff2e6; color:#167653 !important;
            font-size:.76rem; font-weight:750; margin-top:4px;
        }
        .api-dot { width:7px; height:7px; border-radius:50%; background:#2aa866; display:inline-block; }

        .tip-card {
            background:rgba(255,255,255,.82);
            border:1px solid #d6e9dd; border-radius:17px;
            padding:1rem; margin-top:.45rem;
            box-shadow:0 7px 20px rgba(45,116,77,.055);
        }
        .tip-head { display:flex; justify-content:space-between; align-items:center; }
        .tip-label { color:var(--green-800); font-size:.78rem; font-weight:850; text-transform:uppercase; letter-spacing:.055em; }
        .tip-icon { color:var(--green-700); font-size:1rem; }
        .tip-text { color:#36594b; font-size:.91rem; line-height:1.52; margin-top:.55rem; }
        .tip-dots { text-align:center; margin-top:.8rem; color:#b9d7c5; letter-spacing:5px; font-size:.75rem; }
        .tip-dots span { color:#2da266; }
        .tip-footer { color:#71877c; font-size:.72rem; margin-top:.65rem; line-height:1.45; }

        .main-shell { max-width: 1000px; margin: 0 auto; padding: 8px 18px 110px; }
        .title-row { display:flex; align-items:center; gap:14px; margin-top:4px; }
        .title-icon {
            width:48px; height:48px; border-radius:50%; background:#e2f2e7;
            display:flex; align-items:center; justify-content:center;
            color:var(--green-800); font-size:25px;
        }
        .main-title { font-size:2.55rem; font-weight:850; color:var(--green-950); margin:0; letter-spacing:-.045em; }
        .main-subtitle { color:#668174; font-size:1.02rem; margin:2px 0 1.15rem 62px; }

        .welcome-card {
            background:rgba(255,255,255,.90); border:1px solid #dcebe2; border-radius:17px;
            padding:1rem 1.1rem; margin:0 0 1.15rem;
            box-shadow:0 7px 22px rgba(48,104,75,.055);
        }
        .welcome-inner { display:flex; align-items:flex-start; gap:12px; }
        .welcome-icon { width:39px; height:39px; border-radius:50%; background:#e3f3e8; display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
        .welcome-card-title { color:var(--green-950); font-size:1rem; font-weight:800; margin-bottom:.22rem; }
        .welcome-card-text { color:#587064; margin:0; font-size:.84rem; line-height:1.45; }

        [data-testid="stChatMessage"] {
            border:1px solid #dcebe2 !important; border-radius:16px !important;
            background:rgba(255,255,255,.92) !important;
            box-shadow:0 5px 18px rgba(45,100,68,.045) !important;
            margin-bottom:.72rem !important; padding:.15rem .75rem !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li { color:#213f33 !important; font-size:.9rem; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background:#e5f4e9 !important; border-color:#cde5d5 !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            background:#e2f3e8 !important; color:var(--green-800) !important;
        }

        .assistant-source {
            display:inline-block; background:#e7f5eb; border:1px solid #d1e8d9;
            color:#317153; border-radius:999px; padding:4px 9px; font-size:.7rem; font-weight:700; margin-top:.5rem;
        }
        .assistant-time { color:#7a8f85; font-size:.7rem; float:right; margin-top:.55rem; }

        .recovery-card {
            background:linear-gradient(135deg,#f0f9f3,#e7f5eb);
            border:1px solid #d1e7d8; border-radius:13px; padding:.75rem .9rem; margin:.7rem 0 .45rem;
        }
        .recovery-title { color:var(--green-800); font-size:.82rem; font-weight:800; margin-bottom:.5rem; }
        .recovery-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; }
        .recovery-item { display:flex; gap:7px; align-items:flex-start; color:#365649; font-size:.72rem; line-height:1.4; }
        .recovery-item-icon { width:34px; height:34px; flex:0 0 34px; border-radius:50%; background:#ddf0e3; display:flex; align-items:center; justify-content:center; }

        [data-testid="stChatInput"], section[data-testid="stChatInput"] {
            background:#ffffff !important; border:1.5px solid #3aa56d !important; border-radius:13px !important;
            box-shadow:0 6px 18px rgba(42,105,69,.09) !important; color-scheme:light !important;
        }
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInput"] [data-baseweb="textarea"] { background:#fff !important; color-scheme:light !important; }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:hover {
            background:#fff !important; color:#163b2c !important; -webkit-text-fill-color:#163b2c !important;
            caret-color:#2c8a62 !important; opacity:1 !important; font-size:.9rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color:#7a9085 !important; -webkit-text-fill-color:#7a9085 !important; opacity:1 !important; }
        [data-testid="stChatInput"] button { background:#2fa568 !important; border:0 !important; color:#fff !important; border-radius:10px !important; }
        [data-testid="stChatInput"] button:hover { background:#238a58 !important; }
        [data-testid="stBottom"] { background:rgba(241,249,244,.96) !important; border-top:1px solid #d5e8dc !important; color-scheme:light !important; }
        [data-testid="stBottom"] textarea { background:#fff !important; color:#163b2c !important; -webkit-text-fill-color:#163b2c !important; }
        [data-testid="stBottom"] textarea::placeholder { color:#7a9085 !important; -webkit-text-fill-color:#7a9085 !important; }
        [data-testid="stBottom"] button { background:#2fa568 !important; color:#fff !important; border:0 !important; }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color:#14563f !important; -webkit-text-fill-color:#14563f !important; font-weight:750 !important; opacity:1 !important; }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] input:focus,
        [data-testid="stSidebar"] input:hover {
            background:#fff !important; color:#17352a !important; -webkit-text-fill-color:#17352a !important;
            caret-color:#2c8a62 !important; border:1px solid #c7dfd0 !important; border-radius:11px !important;
            box-shadow:0 3px 10px rgba(42,105,69,.06) !important; opacity:1 !important; color-scheme:light !important;
        }
        [data-testid="stSidebar"] input::placeholder { color:#81968b !important; -webkit-text-fill-color:#81968b !important; opacity:1 !important; }
        .stCaption, [data-testid="stCaptionContainer"] { color:#71877c !important; }
        footer { background:#f1f9f4 !important; }
        hr { border-color:#dbe9df !important; }

        @media (max-width: 850px) {
            .main-title { font-size:2.1rem; }
            .main-subtitle { margin-left:0; }
            .recovery-grid { grid-template-columns:repeat(2,1fr); }
        }

        /* Final dark healthcare dashboard theme */
        .stApp {
            background:
                radial-gradient(circle at 88% 8%, rgba(38,120,82,.28), transparent 22%),
                radial-gradient(circle at 65% 85%, rgba(21,93,63,.22), transparent 28%),
                linear-gradient(135deg, #07110e 0%, #0b1713 48%, #08120f 100%) !important;
            color: #e6f5ec !important;
        }
        [data-testid="stHeader"] { background: rgba(7,17,14,.88) !important; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#0b1713 0%,#0d2119 100%) !important;
            border-right: 1px solid #193c2d !important;
        }
        [data-testid="stSidebar"] * { color: #dcefe5 !important; }
        [data-testid="stSidebar"] hr { border-color:#214536 !important; }
        .brand-heart { background:linear-gradient(145deg,#173d2d,#1b5a3e) !important; color:#9ee6bd !important; }
        .brand-title { color:#dff8e9 !important; }
        .brand-subtitle { color:#8eaa9d !important; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color:#baf0d0 !important; -webkit-text-fill-color:#baf0d0 !important; }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] input:focus,
        [data-testid="stSidebar"] input:hover {
            background:#101d19 !important; color:#ecfff4 !important; -webkit-text-fill-color:#ecfff4 !important;
            caret-color:#72d89f !important; border:1px solid #2c6b50 !important;
            box-shadow:inset 0 0 0 1px rgba(77,170,120,.08),0 4px 12px rgba(0,0,0,.22) !important;
            color-scheme:dark !important;
        }
        [data-testid="stSidebar"] input::placeholder { color:#718f82 !important; -webkit-text-fill-color:#718f82 !important; }
        .api-status { background:#153d2d !important; color:#8fe1b2 !important; }
        .tip-card { background:linear-gradient(145deg,#10231b,#0d1c16) !important; border-color:#244c39 !important; box-shadow:0 8px 22px rgba(0,0,0,.2) !important; }
        .tip-label { color:#86d9aa !important; }
        .tip-text { color:#c7ddd2 !important; }
        .tip-dots { color:#385b4b !important; }
        .tip-dots span { color:#45b979 !important; }
        .tip-icon { color:#72d89f !important; }
        .main-shell { color:#e6f5ec !important; }
        .title-icon { background:#153d2d !important; color:#8fe1b2 !important; }
        .main-title { color:#e5f8ed !important; }
        .main-subtitle { color:#9ab5a8 !important; }
        .welcome-card { background:rgba(13,31,24,.88) !important; border-color:#234a39 !important; box-shadow:0 8px 24px rgba(0,0,0,.2) !important; }
        .welcome-icon { background:#163d2d !important; }
        .welcome-card-title { color:#dff8e9 !important; }
        .welcome-card-text { color:#9eb8ab !important; }
        [data-testid="stChatMessage"] { background:#101e18 !important; border-color:#254a3a !important; box-shadow:0 6px 20px rgba(0,0,0,.22) !important; }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color:#d8ebe1 !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#153326 !important; border-color:#2b6048 !important; }
        [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { background:#173d2d !important; color:#8fe1b2 !important; }
        .assistant-source { background:#163c2d !important; border-color:#2b6048 !important; color:#9be1b8 !important; }
        .assistant-time { color:#718d80 !important; }
        .recovery-card { background:linear-gradient(135deg,#10271d,#102119) !important; border-color:#28523d !important; }
        .recovery-title { color:#8fe1b2 !important; }
        .recovery-item { color:#c3d9ce !important; }
        .recovery-item-icon { background:#173d2d !important; }
        [data-testid="stChatInput"], section[data-testid="stChatInput"] {
            background:#101d19 !important; border:1.5px solid #347956 !important; box-shadow:0 8px 24px rgba(0,0,0,.25) !important; color-scheme:dark !important;
        }
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInput"] [data-baseweb="textarea"] { background:#101d19 !important; color-scheme:dark !important; }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:hover {
            background:#101d19 !important; color:#effff5 !important; -webkit-text-fill-color:#effff5 !important;
            caret-color:#72d89f !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color:#78968a !important; -webkit-text-fill-color:#78968a !important; }
        [data-testid="stChatInput"] button { background:#2fa568 !important; color:#fff !important; }
        [data-testid="stBottom"] { background:rgba(7,17,14,.96) !important; border-top:1px solid #1d3e30 !important; color-scheme:dark !important; }
        [data-testid="stBottom"] textarea { background:#101d19 !important; color:#effff5 !important; -webkit-text-fill-color:#effff5 !important; }
        [data-testid="stBottom"] textarea::placeholder { color:#78968a !important; -webkit-text-fill-color:#78968a !important; }
        [data-testid="stBottom"] button { background:#2fa568 !important; color:#fff !important; }
        .stCaption, [data-testid="stCaptionContainer"] { color:#819d90 !important; }
        hr { border-color:#214536 !important; }
        footer { background:#07110e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# System loading
# -----------------------------
@st.cache_resource
def load_system(key):
    db = PatientDatabase(DATASET_PATH, ROOT / "data/healthcare.db")
    llm = GroqLLM(key)
    return db, llm, Orchestrator(llm), SQLAgent(db, llm), RAGAgent(llm)


# -----------------------------
# Sidebar: API + rotating tips
# -----------------------------
HEALTH_TIPS = [
    "Stay hydrated throughout the day, especially when recovering from illness.",
    "Adequate sleep supports the body's normal recovery and immune function.",
    "Wash your hands regularly and avoid touching your face to reduce infection spread.",
    "Choose balanced meals with vegetables, fruit, protein and whole grains when possible.",
    "Gentle movement can be helpful when you feel well enough; avoid pushing through significant symptoms.",
    "Follow the treatment and medication instructions provided by your healthcare professional.",
    "If symptoms are getting worse rather than better, contact a healthcare professional for advice.",
    "Keep commonly used medicines stored safely and follow the label or clinician's instructions.",
    "Taking short breaks during a busy day can help reduce fatigue and improve concentration.",
    "For recovery, prioritize rest, fluids, nutrition and any follow-up recommended by your clinician.",
]


def rotating_health_tip():
    # A new tip is selected every 25 seconds while the dashboard is open.
    tip_index = int(time.time() // 25) % len(HEALTH_TIPS)
    with st.sidebar:
        st.markdown(
            f"""
            <div class="tip-card">
                <div class="tip-head"><span style='color:#9ee6bd;font-weight:800'>🌿 Daily Health Tip</span></div>
                <div style='height:12px'></div>
                <div class="tip-label">Hydration</div>
                <div class="tip-text">{HEALTH_TIPS[tip_index]}</div>
                <div class='tip-dots'><span>●</span> ● ● ●</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-heart">💚</div>
            <div>
                <div class="brand-title">Healthcare<br>Assistant</div>
                <div class="brand-subtitle">Your trusted health companion</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("🔑 <b style='color:#14563f'>Groq API Key</b>", unsafe_allow_html=True)
    key = st.text_input("", value=GROQ_API_KEY, type="password", label_visibility="collapsed")
    if key:
        st.markdown("<div class='api-status'><span class='api-dot'></span> Connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='api-status' style='background:#f8e9e6;color:#9a4b3d !important'><span class='api-dot' style='background:#d36a58'></span> Not connected</div>", unsafe_allow_html=True)
    st.divider()

# Use a fragment when available so the sidebar tip can refresh without rebuilding
# the whole dashboard. The fallback keeps the project compatible with older Streamlit versions.
if hasattr(st, "fragment"):
    @st.fragment(run_every="25s")
    def sidebar_tip():
        rotating_health_tip()

    sidebar_tip()
else:
    rotating_health_tip()


db, llm, router, sql, rag = load_system(key)

# -----------------------------
# Main healthcare dashboard
# -----------------------------
st.markdown(
    """
    <div class="title-row">
        <div class="title-icon">🩺</div>
        <div class="main-title">Healthcare Assistant</div>
    </div>
    <div class="main-subtitle">How can I help you today?</div>
    <div class="welcome-card">
        <div class="welcome-inner">
            <div class="welcome-icon">💬</div>
            <div>
                <div class="welcome-card-title">Ask your question in plain English</div>
                <p class="welcome-card-text">Ask about patient records, hospital policies, admissions, billing, medications, privacy or general health information.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("recovery_tips"):
            render_recovery_tips(recovery_tips)
        if m.get("agent"):
            st.caption("Handled by: " + m["agent"])


def render_recovery_tips(tips):
    icons = ["🛏️", "🥤", "🥗", "💊"]
    st.markdown(
        "<div class='recovery-card'><div class='recovery-title'>🌿 Tips that may support faster recovery</div><div class='recovery-grid'>"
        + "".join(
            f"<div class='recovery-item'><div class='recovery-item-icon'>{icons[i % len(icons)]}</div><div>{tip}</div></div>"
            for i, tip in enumerate(tips)
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )


def recovery_tips_for(query):
    """Return safe, general recovery-support tips when a query is recovery-related."""
    q = query.lower()
    recovery_terms = [
        "recover", "recovery", "cold", "flu", "fever", "cough", "sore throat",
        "infection", "illness", "sick", "tired", "fatigue", "pain", "healing",
        "after surgery", "post surgery", "wound", "vomit", "diarrhea", "dehydration",
    ]
    if not any(term in q for term in recovery_terms):
        return []

    tips = [
        "Get plenty of rest and avoid strenuous activity while you feel unwell.",
        "Drink water and other suitable fluids to help prevent dehydration.",
        "Choose light, balanced meals as tolerated and continue any prescribed diet.",
        "Take medicines only according to the label or your healthcare professional's instructions.",
    ]
    return tips


def local_health_fallback(query):
    """Useful offline answers for common general-health questions when the LLM is unavailable."""
    q = query.lower().strip()
    if any(x in q for x in ["common cold", "having cold", "i have cold", "got a cold", "cold what should i do", "cold what can i do"]):
        return (
            "For a typical common cold, most people improve within about 7–10 days, although a cough or other symptoms can sometimes last longer. "
            "Rest, drink plenty of fluids, eat nourishing foods as tolerated, and use symptom-relief medicines only as directed on the label or by a healthcare professional. "
            "Seek medical care if you have trouble breathing, chest pain, severe dehydration, confusion, a very high or persistent fever, symptoms that are getting worse, or symptoms that are not improving as expected."
        )
    if "fever" in q:
        return (
            "For a mild fever, rest, drink fluids, and monitor how you feel. Follow the label or your clinician's advice if using fever-reducing medicine. "
            "Seek medical care for a very high or persistent fever, severe weakness, confusion, breathing difficulty, chest pain, or other concerning symptoms."
        )
    if "cough" in q:
        return (
            "For a mild cough, rest, stay hydrated, and avoid smoke or other irritants. Warm fluids may soothe the throat. "
            "Get medical advice if the cough is severe, lasts longer than expected, produces significant blood, or comes with breathing difficulty or chest pain."
        )
    return None


query = st.chat_input("Ask a healthcare question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    t = time.time()
    agent = router.classify(query)
    recovery_tips = recovery_tips_for(query)

    with st.chat_message("assistant"):
        if agent == "SQL_AGENT":
            r = sql.run(query)
            ans = sql.summarize(query, r["rows"])
            st.markdown(ans)

            if recovery_tips:
                render_recovery_tips(recovery_tips)

            with st.expander("View generated SQL"):
                st.code(r["sql"], language="sql")
            if r["rows"]:
                st.dataframe(
                    pd.DataFrame(r["rows"]),
                    use_container_width=True,
                    hide_index=True,
                )
            st.markdown(f"<span class='assistant-source'>🗄️ Source: Patient Records</span><span class='assistant-time'>{time.time() - t:.2f}s</span>", unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ans,
                    "agent": "Patient Records",
                    "recovery_tips": recovery_tips,
                }
            )

        elif agent == "RAG_AGENT":
            ans, sources = rag.answer(query)
            st.markdown(ans)

            if recovery_tips:
                render_recovery_tips(recovery_tips)

            with st.expander("View supporting policy sources"):
                for s in sources:
                    st.markdown(f"**{s['title']}** · similarity {s['score']}")
                    st.write(s["text"])
            st.markdown(f"<span class='assistant-source'>📚 Source: Hospital Policies · {len(sources)} sources</span><span class='assistant-time'>{time.time() - t:.2f}s</span>", unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ans,
                    "agent": "Hospital Policies",
                    "recovery_tips": recovery_tips,
                }
            )

        else:
            # GENERAL handles everyday health questions such as "I have a cold, what should I do?".
            # The LLM gives the conversational answer when available; a small offline fallback
            # keeps common questions useful even if the API key/rate limit is unavailable.
            if llm.available:
                ans = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are Healthcare Assistant, a concise and empathetic general-health information assistant. "
                                "Answer the user's question directly in plain English. For common minor illnesses such as a cold, "
                                "give practical self-care guidance, a realistic general recovery timeframe when appropriate, and clear red flags. "
                                "Do not diagnose, prescribe, or invent patient/hospital-policy facts. Never claim certainty about an individual patient's condition. "
                                "For severe, worsening, urgent, or unusual symptoms, recommend prompt professional medical care. "
                                "Keep the answer useful and reasonably concise."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    0.2,
                    650,
                )
            else:
                ans = local_health_fallback(query)
                if not ans:
                    ans = (
                        "I can help with general health information, patient-record questions, and hospital policies. "
                        "For a general health question, please describe your symptoms or concern and I will provide general guidance."
                    )

            st.markdown(ans)
            if recovery_tips:
                render_recovery_tips(recovery_tips)
            st.markdown("<span class='assistant-source'>🧠 Source: General Health Knowledge</span><span class='assistant-time'>Now</span>", unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ans,
                    "agent": "Healthcare Assistant",
                    "recovery_tips": recovery_tips,
                }
            )

st.divider()
st.caption("For demonstration and academic use with synthetic data. This assistant is not a substitute for professional medical advice.")
