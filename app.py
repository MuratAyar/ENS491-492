# app.py – Caregiver‑Child Monitor (with LLM‑as‑Judge integration)

import asyncio
import json
import logging
from typing import Dict, Any, List

import nest_asyncio
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie

from evidently import Dataset, DataDefinition

from pathlib import Path
import uuid, datetime
import re

# ‑‑‑ Streamlit page config ────────────────────────────────────────────────
st.set_page_config(page_title="Caregiver‑Child Monitor", page_icon=":baby:", layout="wide")

# ‑‑‑ Try to import Evidently descriptors (optional) ────────────────────────
try:
    from evidently.descriptors import Sentiment, TextLength
except ImportError:
    Sentiment = TextLength = None  # type: ignore
    st.warning(
        "Evidently installed but text descriptors missing — quick judges disabled."
    )
        

# ‑‑‑ Local agents ─────────────────────────────────────────────────────────
from agents.orchestration.orchestrator import Orchestrator
from agents.test.evaluator_agent import evaluate_models
from agents.test.llm_judge_agent import LLMEvaluatorAgent

# ‑‑‑ Async / logging setup ────────────────────────────────────────────────
nest_asyncio.apply()
logger = logging.getLogger("care_monitor")

# =============================================================================
#                               Session State
# =============================================================================
if "orch" not in st.session_state:
    st.session_state.orch = Orchestrator()
if "judge_agent" not in st.session_state:
    st.session_state.judge_agent = LLMEvaluatorAgent()
if "last_ctx" not in st.session_state:
    st.session_state.last_ctx = None
if "show_toxicity" not in st.session_state:
    st.session_state.show_toxicity = False
if "show_sentiment" not in st.session_state:
    st.session_state.show_sentiment = False
if "show_sarcasm" not in st.session_state:
    st.session_state.show_sarcasm = False
if "show_eval" not in st.session_state:
    st.session_state.show_eval = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

orch: Orchestrator = st.session_state.orch  # type: ignore
judge_agent: LLMEvaluatorAgent = st.session_state.judge_agent  # type: ignore

# =============================================================================
#                               Helpers / UI utils
# =============================================================================

def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=8)
        return r.json() if r.ok else None
    except Exception:
        return None

LOTTIE = _load_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
    html, body, [class*="css"]{font-family:'Quicksand',sans-serif;}
    .card{padding:1rem;border-radius:.5rem;box-shadow:0 2px 6px rgba(0,0,0,.1);margin:1rem 0;}
    .metric-card{text-align:center;border-top:4px solid #84d2f6;padding:1rem .5rem;margin-bottom:1rem;border-radius:.5rem;}
    .metric-card .label{font-size:1rem;color:#333;}
    .metric-card .value{font-size:1.4rem;font-weight:bold;margin-top:.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _metric(col, label: str, value: str):
    col.markdown(
        f"""
        <div class='metric-card'>
          <div class='label'>{label}</div>
          <div class='value'>{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def extract_caregiver_lines(transcript: str) -> list:
    caregiver_tags = ("Caregiver:", "Woman:", "Mother:", "Dad:", "Mum:")
    lines = []
    for line in transcript.splitlines():
        if any(tag in line for tag in caregiver_tags):
            clean = re.sub(r"^\s*\[\d{1,2}:\d{2}\]\s*", "", line)
            text = clean.split(":", 1)[-1].strip()
            lines.append(text)
    return lines



# =============================================================================
#                               Sidebar Nav
# =============================================================================
page = st.sidebar.radio("Go to", ["Home", "Analyze", "Trends"], index=["Home", "Analyze", "Trends"].index(st.session_state.current_page))
st.session_state.current_page = page

# >>> NEW block – global settings <<< ----------------------------------------
st.sidebar.markdown("### Settings")

auto_tr = st.sidebar.checkbox(
    "↔ Auto-translate non-English input",
    value=getattr(st.session_state, "auto_tr", False),
    help="If checked, the app will detect language and use Argos-Translate to "
         "convert non-English transcripts to English."
)
st.session_state.auto_tr = auto_tr

# Tell the orchestrator (see next section)
orch.set_translation_flag(auto_tr)
# ---------------------------------------------------------------------------

# =============================================================================
#                               HOME PAGE
# =============================================================================
if page == "Home":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<h1>Caregiver‑Child Monitor</h1>", unsafe_allow_html=True)
        st.markdown("<p><em>“It takes a big heart to shape little minds.”</em></p>", unsafe_allow_html=True)
        if st.button("Get Started"):
            st.session_state.current_page = "Analyze"
            st.experimental_rerun()
    with c2:
        if LOTTIE:
            st_lottie(LOTTIE, height=200)

# =============================================================================
#                               ANALYZE PAGE
# =============================================================================
elif page == "Analyze":

    st.markdown("## Analyze a Caregiver‑Child Conversation")

    if "transcript_history" not in st.session_state:
        st.session_state.transcript_history = ""

    txt = st.text_area("Paste transcript here (time‑tagged):", height=170, value=st.session_state.transcript_history)

    col_r, col_a = st.columns(2)

    # ── Reset button ──────────────────────────────────────────────────────────
    if col_r.button("Reset"):
        st.session_state.transcript_history = ""
        st.session_state.last_ctx = None
        st.session_state.show_toxicity = False
        st.session_state.show_sentiment = False
        st.session_state.show_sarcasm = False
        st.session_state.show_eval = False
        try:
            orch.response_generator_agent.vector_store.reset()
        except Exception:
            pass
        st.success("History cleared.")
        st.rerun()

    # ── Analyze button ───────────────────────────────────────────────────────
    if col_a.button("Analyze") and txt.strip():
        st.session_state.transcript_history = txt.strip()
        st.session_state.show_toxicity = False
        st.session_state.show_sentiment = False
        st.session_state.show_sarcasm = False
        st.session_state.show_eval = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ctx: Dict[str, Any] = {"transcript": txt.strip()}

        # ‑‑‑ FAST ANALYSIS ----------------------------------------------------
        with st.spinner("Computing quick results …"):
            t_res, a_res, c_res, s_res = loop.run_until_complete(
                asyncio.gather(
                    orch.tox_agent.run([{"content": json.dumps({"transcript": txt})}]),   # NEW
                    orch.analyzer_agent.run([{"content": json.dumps({"transcript": txt})}]),
                    orch.categorizer_agent.run([{"content": json.dumps({"transcript": txt})}]),
                    orch.sarcasm_agent.run([{"content": json.dumps({"transcript": txt})}]),
                )
            )
            ctx.update(t_res)      # NEW
            ctx.update(a_res)
            ctx.update(c_res)
            ctx.update(s_res)   

            

        # ‑‑‑ HEAVY ANALYSIS ---------------------------------------------------
        
        with st.spinner("Running detailed insights …"):
            star_raw = loop.run_until_complete(orch.star_reviewer_agent.run(ctx))
            star = json.loads(star_raw) if isinstance(star_raw, str) else star_raw
            ctx.update(star)

            heavy = loop.run_until_complete(
                asyncio.gather(
                    orch.response_generator_agent.run([{"content": json.dumps(ctx)}]),
                )
            )
            loop.close()
            for res in heavy:
                if isinstance(res, dict):
                    ctx.update(res)

        # -- JSON EXPORT --------------------------------------------------------


        def save_ctx_as_json(ctx: Dict[str, Any]) -> str:
            """Save analysis context to a timestamped JSON file and return the path."""
            out_dir = Path("data/analysis_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            uid  = uuid.uuid4().hex[:8]
            file = out_dir / f"analysis_{ts}_{uid}.json"
            with open(file, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2, ensure_ascii=False)
            logger.info(f"[App] Analysis saved → {file}")
            return str(file)

        # … ağır analiz bittikten ve ctx güncellendikten sonra:
        json_path = save_ctx_as_json(ctx)
        st.success(f"JSON output saved → {json_path}")
        # -----------------------------------------------------------------------

        st.session_state.last_ctx = ctx

    # ── RENDER OUTPUT ────────────────────────────────────────────────────────
    if st.session_state.last_ctx:
        ctx = st.session_state.last_ctx

        # Top metrics ---------------------------------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)
        sent_val = ctx.get("sentiment_score", 0)
        sent_css = "color:#e74c3c" if sent_val < -0.3 else "color:#27ae60" if sent_val > 0.3 else "color:#f1c40f"
        _metric(c1, "Sentiment", f"<span style='{sent_css}'>{sent_val:.2f}</span>")

        _metric(c2, "Category", ctx.get("primary_category", "N/A"))
        _metric(c3, "Caregiver Score", f"{ctx.get('caregiver_score', 'N/A')}/10")

        sar = ctx.get("sarcasm", 0)
        sar_css = "color:#e74c3c" if sar >= .3 else "color:#27ae60"
        _metric(c4, "Sarcasm", f"<span style='{sar_css}'>{sar:.2f}</span>")

        tox = ctx.get("toxicity", 0)
        tox_css = "color:#e74c3c" if tox >= .3 else "color:#27ae60"
        _metric(c5, "Toxicity", f"<span style='{tox_css}'>{tox:.2f}</span>")

        # Toggles just below metrics
        t0, t1, t2 = st.columns(3)
        if t0.button("💬 Show Sentiment Details"):
            st.session_state.show_sentiment = not st.session_state.show_sentiment
        if t1.button("📣 Show Toxicity Details"):
            st.session_state.show_toxicity = not st.session_state.show_toxicity
        if t2.button("🧂 Show Sarcasm Details"):
            st.session_state.show_sarcasm = not st.session_state.get("show_sarcasm", False)

        # Details: Sentiment
        if st.session_state.show_sentiment and ctx.get("sentiment_scores"):
            st.markdown("### Sentiment by Utterance (Child & Caregiver)")
            lines = ctx["transcript"].splitlines()
            scores = ctx["sentiment_scores"]
            for i, (ln, score) in enumerate(zip(lines, scores), 1):
                txt = ln.split(":",1)[-1].strip()
                st.markdown(f"**{i}.** `{txt}`  →  **Sentiment Score:** `{score:.3f}`")

        # Details: Toxicity
        if st.session_state.show_toxicity and ctx.get("toxicity_scores"):
            st.markdown("### Toxicity Scores by Caregiver Utterance")
            lines = extract_caregiver_lines(ctx["transcript"])
            scores = ctx["toxicity_scores"]
            for i, (line, score) in enumerate(zip(lines, scores), 1):
                st.markdown(f"**{i}.** `{line}`  →  **Toxicity:** `{score:.3f}`")

        # Details: Sarcasm
        if st.session_state.get("show_sarcasm") and ctx.get("sarcasm_scores"):
            st.markdown("### Sarcasm Scores by Caregiver Utterance")
            lines = extract_caregiver_lines(ctx["transcript"])
            scores = ctx["sarcasm_scores"]
            for i, (line, score) in enumerate(zip(lines, scores), 1):
                st.markdown(f"**{i}.** `{line}`  →  **Sarcasm:** `{score:.3f}`")

        # Justification & Parent Notification --------------------------------
        st.markdown(
            f"<div class='card'><h3>Caregiver Score Justification</h3><p>{ctx.get('justification','N/A')}</p></div>",
            unsafe_allow_html=True,
        )
        note = ctx.get("parent_notification", "")
        if ctx.get("send_notification"):
            st.markdown(f"<div class='card'><h3>Parent Notification</h3><p>{note}</p></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><h3>Parent Notification</h3><p>— none needed —</p></div>",
                        unsafe_allow_html=True)
        

        # Recommendations -----------------------------------------------------
        recs = ctx.get("recommendations", [])
        if recs:
            items = "".join(f"<li><strong>{r['category']}</strong>: {r['description']}</li>" for r in recs)
            st.markdown(
                f"<div class='card'><h3>Recommendations</h3><ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='card'><h3>Recommendations</h3><p>None.</p></div>", unsafe_allow_html=True)

        # Toggles -------------------------------------------------------------
        t_col = st.columns(1)[0]
        if t_col.button("🔍 Evaluate Models"):
            st.session_state.show_eval = not st.session_state.show_eval

        # Evaluation section --------------------------------------------------
        if st.session_state.show_eval:
            st.header("OpenHermes‑based Agent Outputs (raw)")
            st.subheader("CaregiverScorerAgent")
            st.write(f"**Caregiver Score:** {ctx.get('caregiver_score','N/A')} / 10")
            st.subheader("ParentNotifierAgent")
            st.write("**Parent Notification:**")
            st.write(ctx.get("parent_notification", "—"))
            st.write("**Recommendations:**")
            for rec in ctx.get("recommendations", []):
                st.write(f"- **{rec['category']}**: {rec['description']}")

            st.markdown("---")

            # Quick Evidently judge
            df_qwen = pd.DataFrame(
                {
                    "Justification": [ctx.get("justification", "")],
                    "ParentNotification": [ctx.get("parent_notification", "")],
                    "Recommendations": [" ".join(r["description"] for r in ctx.get("recommendations", []))],
                    "Category": [ctx.get("primary_category", "")],
                }
            )
            desc: List[Any] = []
            if Sentiment and TextLength:
                desc = [
                    Sentiment("Justification", alias="sent_jus"),
                    TextLength("Justification", alias="len_jus"),
                    Sentiment("ParentNotification", alias="sent_pn"),
                    TextLength("ParentNotification", alias="len_pn"),
                    TextLength("Recommendations", alias="len_recs"),
                    TextLength("Category", alias="len_cat"),
                ]
            ds_q = Dataset.from_pandas(df_qwen, data_definition=DataDefinition(), descriptors=desc)
            out_q = ds_q.as_dataframe()

            judges = {}
            if desc:
                sj, lj = out_q["sent_jus"].iloc[0], out_q["len_jus"].iloc[0]
                judges["CaregiverScorerAgent Judge"] = min(10, (sj + 1) * 5 + min(lj, 200) / 200 * 5)
                sp, lp = out_q["sent_pn"].iloc[0], out_q["len_pn"].iloc[0]
                judges["ParentNotifierAgent Judge"] = min(10, (sp + 1) * 5 + min(lp, 200) / 200 * 5)

            st.header("Evidently Quick‑Judge")
            cols = st.columns(len(judges))
            for (name, score), col in zip(judges.items(), cols):
                col.metric(name, f"{score:.1f}/10")

            st.markdown("---")

            # Deep HF evaluation ---------------------------------------------
            st.header("Deep HF Model Evaluation")
            evals = evaluate_models(ctx["transcript"])
            if not evals:
                st.warning("No evaluation data available.")
            else:
                for name, det in evals.items():
                    st.subheader(name)
                    _metric(st, "Score", f"{det['score']}/10")
                    if det.get("explanation"):
                        st.markdown(f"**Why:** {det['explanation']}")
                    if det["correct"]:
                        st.write("**Correct predictions:**")
                        for ex in det["correct"][:5]:
                            st.write(f"- {ex}")
                    if det["incorrect"]:
                        st.write("**Incorrect predictions:**")
                        for ex in det["incorrect"][:5]:
                            st.write(f"- {ex}")

            # LLM‑as‑Judge ----------------------------------------------------
            st.markdown("---")
            st.header("LLM‑as‑Judge Critique")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            feedback = loop.run_until_complete(judge_agent.run([{"content": json.dumps(ctx)}]))
            loop.close()

            if feedback.get("sentiment_feedback"):
                st.subheader("Sentiment Feedback")
                st.write(feedback["sentiment_feedback"])
            if feedback.get("category_feedback"):
                st.subheader("Category Feedback")
                st.write(feedback["category_feedback"])
            if feedback.get("justification_feedback"):
                st.subheader("Justification Feedback")
                st.write(feedback["justification_feedback"])
            if feedback.get("parent_notification_feedback"):
                st.subheader("Parent Notification Feedback")
                st.write(feedback["parent_notification_feedback"])

            for idx, line in enumerate(feedback.get("recommendations_feedback", []), 1):
                st.write(f"{idx}. {line}")

# =============================================================================
#                               TRENDS PAGE
# =============================================================================
elif page == "Trends":
    st.markdown("## Trends Over Time")
    data = {
        "Week": ["W1", "W2", "W3", "W4"],
        "Nutrition": [5, 6, 7, 8],
        "Health": [3, 4, 4, 5],
        "Learning": [4, 5, 6, 7],
    }
    df = pd.DataFrame(data).melt("Week", var_name="Category", value_name="Count")
    fig = px.line(
        df,
        x="Week",
        y="Count",
        color="Category",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(fig, use_container_width=True)
