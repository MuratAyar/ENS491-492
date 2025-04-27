# app.py – Caregiver-Child Monitor (with LLM-as-Judge integration)

import asyncio
import json
import logging
from collections import Counter
from typing import Dict, Any, List

import nest_asyncio
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_lottie import st_lottie

from evidently import Dataset, DataDefinition

# Must be first Streamlit call
st.set_page_config(
    page_title="Caregiver-Child Monitor",
    page_icon=":baby:",
    layout="wide"
)

# Evidently text descriptors
try:
    from evidently.descriptors.text_descriptors import Sentiment, TextLength
except ImportError:
    try:
        from evidently.descriptors import Sentiment, TextLength
    except ImportError:
        Sentiment = TextLength = None
        st.warning("Evidently installed but text descriptors missing—quick judges disabled.")

# Local orchestrator & evaluators
from agents.orchestrator import Orchestrator
from agents.evaluator_agent import evaluate_models
from agents.llm_judge_agent import LLMEvaluatorAgent

# Streamlit setup
nest_asyncio.apply()
logger = logging.getLogger("care_monitor")

# Persist state
if "orch" not in st.session_state:
    st.session_state.orch = Orchestrator()
if "judge_agent" not in st.session_state:
    st.session_state.judge_agent = LLMEvaluatorAgent()
if "last_ctx" not in st.session_state:
    st.session_state.last_ctx = None
if "show_toxicity" not in st.session_state:
    st.session_state.show_toxicity = False
if "show_eval" not in st.session_state:
    st.session_state.show_eval = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

orch: Orchestrator = st.session_state.orch  # type: ignore
judge_agent: LLMEvaluatorAgent = st.session_state.judge_agent  # type: ignore

# Helpers
def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=8)
        return r.json() if r.ok else None
    except Exception:
        return None

LOTTIE = _load_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
html, body, [class*="css"]{font-family:'Quicksand',sans-serif;}
.card{background:#fff;padding:1rem;border-radius:.5rem;box-shadow:0 2px 6px rgba(0,0,0,.1);margin:1rem 0;}
.metric-card{text-align:center;border-top:4px solid #84d2f6;padding:1rem .5rem;margin-bottom:1rem;border-radius:.5rem;}
.metric-card .label{font-size:1rem;color:#333;}
.metric-card .value{font-size:1.4rem;font-weight:bold;margin-top:.3rem;}
</style>
""", unsafe_allow_html=True)

def _metric(col, label: str, value: str):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='label'>{label}</div>
      <div class='value'>{value}</div>
    </div>""", unsafe_allow_html=True)

# Sidebar
page = st.sidebar.radio(
    "Go to", ["Home", "Analyze", "Trends"],
    index=["Home", "Analyze", "Trends"].index(st.session_state.current_page)
)
st.session_state.current_page = page

# Home
if page == "Home":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<h1>Caregiver-Child Monitor</h1>", unsafe_allow_html=True)
        st.markdown("<p><em>“It takes a big heart to shape little minds.”</em></p>", unsafe_allow_html=True)
        if st.button("Get Started"):
            st.session_state.current_page = "Analyze"
            st.experimental_rerun()
    with c2:
        if LOTTIE:
            st_lottie(LOTTIE, height=200)

# Analyze
elif page == "Analyze":
    st.markdown("## Analyze a Caregiver-Child Conversation")
    if "transcript_history" not in st.session_state:
        st.session_state.transcript_history = ""
    txt = st.text_area("Paste transcript here (time-tagged):", height=170, value=st.session_state.transcript_history)

    col_r, col_a = st.columns(2)
    if col_r.button("Reset"):
        st.session_state.transcript_history = ""
        st.session_state.last_ctx = None
        st.session_state.show_toxicity = False
        st.session_state.show_eval = False
        orch.response_generator_agent.vector_store.reset()
        st.success("History cleared.")

    if col_a.button("Analyze") and txt.strip():
        st.session_state.transcript_history = txt.strip()
        st.session_state.show_toxicity = False
        st.session_state.show_eval = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ctx: Dict[str, Any] = {"transcript": txt.strip()}

        with st.spinner("Computing quick results…"):
            lang = loop.run_until_complete(orch.language_agent.run(txt.strip()))
            ctx.update(lang)
            a_res, c_res = loop.run_until_complete(asyncio.gather(
                orch.analyzer_agent.run([{"content": json.dumps({"transcript": txt})}]),
                orch.categorizer_agent.run([{"content": json.dumps({"transcript": txt})}])
            ))
            ctx.update(a_res); ctx.update(c_res)

            star_raw = loop.run_until_complete(
                orch.star_reviewer_agent.run(
                    txt.strip(),
                    ctx.get("sentiment", "Neutral"),
                    ctx.get("responsiveness", "Passive")
                )
            )
            star = json.loads(star_raw) if isinstance(star_raw, str) else star_raw
            score5 = float(star.get("caregiver_score", 0))
            star["caregiver_score"] = round((score5 / 5) * 10, 1)
            if not star.get("justification"):
                star["justification"] = (
                    "The caregiver’s response was terse or dismissive, indicating low empathy. "
                    "Using kinder language would improve this."
                )
            ctx.update(star)

        with st.spinner("Running detailed insights…"):
            heavy = loop.run_until_complete(asyncio.gather(
                orch.response_generator_agent.run([{"content": json.dumps(ctx)}]),
                orch.tline_agent.run([{"content": json.dumps(ctx)}]),
                orch.emotion_agent.run([{"content": json.dumps(ctx)}]),
                orch.abuse_agent.run([{"content": json.dumps(ctx)}]),
            ))
            loop.close()
            for res in heavy:
                if isinstance(res, dict):
                    ctx.update(res)

        st.session_state.last_ctx = ctx

    # Render if available
    if st.session_state.last_ctx:
        ctx = st.session_state.last_ctx

        # Top metrics
        c1, c2, c3 = st.columns(3)
        _metric(c1, "Sentiment", ctx.get("sentiment", "N/A"))
        _metric(c2, "Category", ctx.get("primary_category", "N/A"))
        _metric(c3, "Caregiver Score", f"{ctx.get('caregiver_score')}/10")

        # Justification & Parent Notification
        st.markdown(f"<div class='card'><h3>Justification</h3><p>{ctx['justification']}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h3>Parent Notification</h3><p>{ctx['parent_notification']}</p></div>", unsafe_allow_html=True)

        # Recommendations
        recs = ctx.get("recommendations", [])
        if recs:
            items = "".join(f"<li><strong>{r['category']}</strong>: {r['description']}</li>" for r in recs)
            st.markdown(f"<div class='card'><h3>Recommendations</h3><ul>{items}</ul></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><h3>Recommendations</h3><p>None.</p></div>", unsafe_allow_html=True)

        # Timelines helper
        def _render_tl(title: str, key: str, field: str):
            data = ctx.get(key, [])
            if data:
                st.markdown(
                    f"<div class='card'><h3>{title}</h3>" +
                    "".join(f"<p><strong>{i['time']}</strong> → {i.get(field,'')}</p>" for i in data) +
                    "</div>", unsafe_allow_html=True
                )
        _render_tl("Timeline Sentiment", "timeline_sentiment", "sentiment")
        _render_tl("Timeline Categories", "timeline_categories", "category")
        _render_tl("Timeline Emotions", "timeline_emotions", "emotion")
        _render_tl("Timeline Abuse Flags", "timeline_abuse", "abusive")

        # Toggles
        t1, t2 = st.columns(2)
        if t1.button("📣 Show Toxicity Details"):
            st.session_state.show_toxicity = not st.session_state.show_toxicity
        if t2.button("🔍 Evaluate Models"):
            st.session_state.show_eval = not st.session_state.show_eval

        # Toxicity details
        if st.session_state.show_toxicity:
            st.info(f"Toxicity = **{ctx.get('toxicity',0):.3f}**, Abuse = {ctx.get('abuse_flag')}")

        # Evaluate Models
        if st.session_state.show_eval:
            # 1) Raw Qwen outputs
            st.header("Qwen-based Agent Outputs (raw)")
            st.subheader("CaregiverScorerAgent (model=qwen:7b)")
            st.write(f"**Caregiver Score:** {ctx.get('caregiver_score')} / 10")
            st.caption("_Judged on empathy, responsiveness & engagement_")

            st.subheader("ParentNotifierAgent (model=qwen:7b)")
            st.write("**Parent Notification:**")
            st.write(ctx.get("parent_notification", "—"))
            st.write("**Recommendations:**")
            for rec in ctx.get("recommendations", []):
                st.write(f"- **{rec['category']}**: {rec['description']}")

            st.markdown("---")

            # 2) Quick Evidently judge of Qwen outputs
            df_qwen = pd.DataFrame({
                "Justification":     [ctx.get("justification","")],
                "ParentNotification":[ctx.get("parent_notification","")],
                "Recommendations":   [" ".join(r["description"] for r in ctx.get("recommendations",[]))],
                "Category":          [ctx.get("primary_category","")]
            })
            desc = []
            if Sentiment and TextLength:
                desc = [
                    Sentiment("Justification",      alias="sent_jus"),
                    TextLength("Justification",     alias="len_jus"),
                    Sentiment("ParentNotification", alias="sent_pn"),
                    TextLength("ParentNotification",alias="len_pn"),
                    TextLength("Recommendations",   alias="len_recs"),
                    TextLength("Category",          alias="len_cat"),
                ]
            ds_q = Dataset.from_pandas(df_qwen, data_definition=DataDefinition(), descriptors=desc)
            out_q = ds_q.as_dataframe()

            judges = {}
            if desc:
                sj, lj = out_q["sent_jus"].iloc[0], out_q["len_jus"].iloc[0]
                judges["CaregiverScorerAgent Judge (Qwen)"] = min(10, (sj+1)*5 + min(lj,200)/200*5)
                sp, lp = out_q["sent_pn"].iloc[0], out_q["len_pn"].iloc[0]
                judges["ParentNotifierAgent Judge (Qwen)"] = min(10, (sp+1)*5 + min(lp,200)/200*5)

            st.header("Evidently Quick-Judge of Qwen Outputs")
            cols = st.columns(len(judges))
            for (name, score), col in zip(judges.items(), cols):
                col.metric(name, f"{score:.1f}/10")

            st.markdown("---")

            # 3) Deep HF Model Evaluation
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

            # 4) True LLM-as-Judge Critique
            st.markdown("---")
            st.header("LLM-as-Judge (Qwen) Critique")

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

            rec_fb: List[str] = feedback.get("recommendations_feedback", [])
            if rec_fb:
                st.subheader("Recommendations Feedback")
                for idx, line in enumerate(rec_fb, 1):
                    st.write(f"{idx}. {line}")

# Trends
elif page == "Trends":
    st.markdown("## Trends Over Time")
    data = {"Week": ["W1","W2","W3","W4"], "Nutrition": [5,6,7,8], "Health": [3,4,4,5], "Learning": [4,5,6,7]}
    df = pd.DataFrame(data).melt("Week", var_name="Category", value_name="Count")
    fig = px.line(df, x="Week", y="Count", color="Category", markers=True,
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(t=20,b=40,l=40,r=20))
    st.plotly_chart(fig, use_container_width=True)
