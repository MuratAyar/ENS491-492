import streamlit as st
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
import pandas as pd
import nest_asyncio
import asyncio

# Import your orchestrator (adjust path if needed)
from agents.orchestrator import Orchestrator

# Fix asyncio loop conflicts in Streamlit
nest_asyncio.apply()

# Set page configuration for the app
st.set_page_config(page_title="Caregiver-Child Monitor", page_icon=":baby:", layout="wide")

# Create a single instance of the Orchestrator (we'll reuse it)
if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()

# Use a session state key for navigation; default to "Home"
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Function to load Lottie animations from a URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation for the hero section
LOTTIE_URL = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_anim = load_lottieurl(LOTTIE_URL)

# Inject custom CSS styles for fonts, colors, and layout
st.markdown("""
<style>
/* Import a soft, child-friendly font */
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Quicksand', sans-serif;
}
/* General text styles */
h1, h2, h3, h4, h5, h6 {
    color: #444;
    margin-bottom: 0.5rem;
}
p {
    color: #555;
    line-height: 1.6;
}
/* Style Streamlit default buttons */
.stButton button {
    border-radius: 0.5rem;
    background-color: #84d2f6;
    color: white;
    padding: 0.5rem 1rem;
}
.stButton button:hover {
    background-color: #6cc4e5;
}
/* Card component style */
.card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.card h3 {
    color: #333;
    margin-bottom: 0.5rem;
}
.card p {
    color: #666;
    font-size: 0.95rem;
    margin: 0.5rem 0 0;
}
/* Hover effect for cards (subtle lift) */
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease-in-out;
}
/* Style for hero section columns */
.stColumns {
    align-items: stretch;
    margin-bottom: 2rem;
}
.stColumns > div:first-child {
    background-color: #f9f5ff;
    padding: 2rem;
    border-top-left-radius: 0.5rem;
    border-bottom-left-radius: 0.5rem;
}
.stColumns > div:last-child {
    background-color: #f9f5ff;
    padding: 2rem;
    text-align: center;
    border-top-right-radius: 0.5rem;
    border-bottom-right-radius: 0.5rem;
}
/* Metrics & Report styling */
.metric-card {
    text-align: center;
    border-top: 4px solid #84d2f6;
    padding: 1rem 0.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}
.metric-card .label {
    font-size: 1rem;
    color: #333;
}
.metric-card .value {
    font-size: 1.4rem;
    font-weight: bold;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

###########################
# Display analysis results
###########################
def display_analysis_results(result: dict):
    """Show top-level metrics in 'metric cards', then box everything else in separate card sections."""
    # 1) Top-level performance metrics
    col1, col2, col3 = st.columns(3)
    sentiment = result.get("sentiment", "N/A")
    category = result.get("primary_category", "N/A")
    score = result.get("caregiver_score", "N/A")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Sentiment</div>
            <div class="value">{sentiment}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Category</div>
            <div class="value">{category}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Caregiver Score</div>
            <div class="value">{score}/5</div>
        </div>
        """, unsafe_allow_html=True)

    # 2) Additional insights (Tone, Empathy, Responsiveness)
    tone = result.get("tone", "N/A")
    empathy = result.get("empathy", "N/A")
    responsiveness = result.get("responsiveness", "N/A")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Tone</div>
            <div class="value">{tone}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Empathy</div>
            <div class="value">{empathy}</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Responsiveness</div>
            <div class="value">{responsiveness}</div>
        </div>
        """, unsafe_allow_html=True)

    # Build content for Justification, Parent Notification, Recommendations, Timeline
    justification = result.get("justification", "N/A")
    parent_note = result.get("parent_notification", "N/A")

    recs = result.get("recommendations", [])
    if recs:
        recs_html = "<ul>"
        for rec in recs:
            cat = rec.get('category', 'General')
            desc = rec.get('description', '')
            recs_html += f"<li><strong>{cat}</strong>: {desc}</li>"
        recs_html += "</ul>"
    else:
        recs_html = "<p>No specific recommendations.</p>"

    timeline = result.get("timeline_categories", [])
    if timeline:
        timeline_html = ""
        for item in timeline:
            timeline_html += f"<p><strong>{item['time']}</strong> → {item['category']}</p>"
    else:
        timeline_html = "<p>No timeline data available.</p>"

    st.markdown("### Detailed Report")
    # 3) Justification & Parent Notification side by side
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"""
        <div class="card">
            <h3>Justification</h3>
            <p>{justification}</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="card">
            <h3>Parent Notification</h3>
            <p>{parent_note}</p>
        </div>
        """, unsafe_allow_html=True)

    # 4) Recommendations & Timeline side by side
    colC, colD = st.columns(2)
    with colC:
        st.markdown(f"""
        <div class="card">
            <h3>Recommendations for Improvement</h3>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)

    with colD:
        st.markdown(f"""
        <div class="card">
            <h3>Timeline Categories</h3>
            {timeline_html}
        </div>
        """, unsafe_allow_html=True)


#########################################
# SIDEBAR NAVIGATION
#########################################
page = st.sidebar.radio("Go to", ["Home", "Analyze", "Trends"],
                        index=["Home", "Analyze", "Trends"].index(st.session_state["page"]))
st.session_state["page"] = page


#########################################
# HOME PAGE
#########################################
if st.session_state["page"] == "Home":
    # Hero section with quote, branding, and call-to-action
    col1, col2 = st.columns([2, 1], gap="small")
    with col1:
        st.markdown("<h1 style='font-size:2.5rem; margin-bottom:0.5rem;'>Caregiver-Child Monitor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem; font-style:italic; color:#555;'>\"It takes a big heart to shape little minds.\"</p>", unsafe_allow_html=True)
        st.markdown("<p style='margin-top:1.5rem;'><em>Monitor empathy, sentiment, and care patterns with ease.</em></p>", unsafe_allow_html=True)
        if st.button("Get Started"):
            st.session_state["page"] = "Analyze"
            st.warning("Please click on 'Analyze' in the sidebar to proceed.")
    with col2:
        if lottie_anim:
            st_lottie(lottie_anim, height=200, key="hero_anim")
        else:
            st.write("🎈")  # fallback if animation doesn't load

    # Feature highlight cards
    st.markdown("## Key Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    feat_col1.markdown("""
    <div class="card" style="text-align:center;">
        <div style='font-size:2rem;'>🤗</div>
        <h3>Analyze Empathy & Sentiment</h3>
        <p>Get insights into the emotional tone and empathy levels in conversations.</p>
    </div>
    """, unsafe_allow_html=True)
    feat_col2.markdown("""
    <div class="card" style="text-align:center;">
        <div style='font-size:2rem;'>💬</div>
        <h3>Parent-Friendly Feedback</h3>
        <p>Receive easy-to-understand suggestions to improve caregiving.</p>
    </div>
    """, unsafe_allow_html=True)
    feat_col3.markdown("""
    <div class="card" style="text-align:center;">
        <div style='font-size:2rem;'>📊</div>
        <h3>Track Care Over Time</h3>
        <p>Monitor care categories and progress week by week.</p>
    </div>
    """, unsafe_allow_html=True)

    # Visual summary section (mock charts)
    st.markdown("## At a Glance")
    sum_col1, sum_col2 = st.columns(2)
    with sum_col1:
        st.markdown("**Common Care Categories**")
        categories = ["Nutrition", "Health", "Learning"]
        values = [40, 35, 25]  # mock distribution in percentage
        pie_fig = px.pie(values=values, names=categories, color_discrete_sequence=px.colors.qualitative.Pastel)
        pie_fig.update_traces(textinfo='percent+label')
        pie_fig.update_layout(showlegend=False, margin=dict(t=30, b=30, l=0, r=0))
        st.plotly_chart(pie_fig, use_container_width=True)
    with sum_col2:
        st.markdown("**Average Caregiver Score by Week**")
        weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
        scores = [75, 80, 85, 90]  # mock scores (0-100 scale)
        bar_fig = px.bar(x=weeks, y=scores, labels={'x': 'Week', 'y': 'Score'},
                         color_discrete_sequence=["#84d2f6"])
        bar_fig.update_layout(yaxis_range=[0, 100], margin=dict(t=30, b=30, l=0, r=0))
        st.plotly_chart(bar_fig, use_container_width=True)

#########################################
# ANALYZE PAGE
#########################################
elif st.session_state["page"] == "Analyze":
    st.markdown("## Analyze a Caregiver-Child Conversation")

    # We'll store transcript in session_state
    if "transcript_history" not in st.session_state:
        st.session_state["transcript_history"] = ""

    transcript_text = st.text_area(
        "Paste a caregiver-child transcript below (each line should start with a time bracket, e.g., (13:00) Child: ...):",
        height=150
    )

    colA, colB = st.columns([1,1])
    if colA.button("Reset Transcript History"):
        st.session_state["transcript_history"] = ""
        # Reset the persistent vector store as well:
        st.session_state["orchestrator"].response_generator_agent.vector_store.reset()
        st.success("Transcript history and vector store have been reset.")

    if colB.button("Analyze Transcript"):
        if transcript_text.strip():
            st.session_state["transcript_history"] = transcript_text.strip()
            with st.spinner("Analyzing... Please wait..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    st.session_state["orchestrator"].process_transcript(st.session_state["transcript_history"])
                )
            if result and "error" not in result:
                display_analysis_results(result)
                st.session_state["last_result"] = result
            else:
                st.error(f"Failed to process transcript: {result.get('error','Unknown error')}")
        else:
            st.error("Please enter a transcript to analyze.")

    # If there's a previously analyzed result, show it
    if "last_result" in st.session_state and st.session_state["last_result"]:
        st.markdown("---")
        st.markdown("### Previous Analysis Result:")
        display_analysis_results(st.session_state["last_result"])

#########################################
# TRENDS PAGE
#########################################
elif st.session_state["page"] == "Trends":
    st.markdown("## Trends Over Time")
    st.write("Here's how different care aspects have evolved over the past weeks (mock data):")

    trend_data = {
        "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "Nutrition": [5, 6, 7, 8],
        "Health": [3, 4, 4, 5],
        "Learning": [4, 5, 6, 7]
    }
    df_trend = pd.DataFrame(trend_data)
    df_long = df_trend.melt(id_vars="Week", var_name="Category", value_name="Count")
    line_fig = px.line(df_long, x="Week", y="Count", color="Category", markers=True,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    line_fig.update_layout(margin=dict(t=20, b=40, l=40, r=20))
    st.plotly_chart(line_fig, use_container_width=True)
