import streamlit as st
import asyncio
import base64
import pandas as pd
from agents.orchestrator import Orchestrator
import nest_asyncio  # Fixes asyncio issues with Streamlit

# Apply nest_asyncio to fix event loop conflicts
nest_asyncio.apply()

# Initialize the orchestrator
orchestrator = Orchestrator()

# Background Ayarlama Fonksiyonu
def set_background(image_file):
    # Logo'yu Base64 formatına çevirip doğrudan HTML ile ekleme
    with open("babylogo4.jpg", "rb") as logo_file:
        logo_base64 = base64.b64encode(logo_file.read()).decode()

    # Logo'yu sağ üst köşeye sabitliyoruz ve boyutunu büyütüyoruz
    st.markdown(
        f"""
        <style>
        .logo-container {{
            position: fixed;
            top: 10px;
            right: 20px;
            width: 280px; /* Logoyu daha büyük yaptım */
            z-index: 1000;
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" width="280">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Arka plan resmini Base64 formatına çevirerek sayfaya ekliyoruz
    with open(image_file, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode()
    
    # Arka plan ve UI stillerini tanımlıyoruz
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            overflow-y: auto;
        }}
        .block-container {{
            padding-top: 8vh !important;
            padding-bottom: 10vh !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            height: auto;
        }}
        header, .stDeployButton {{
            display: none !important;
        }}
        .dynamic-text {{
            font-size: 75px !important;
            font-weight: bold;
            text-align: center;
            animation: color-change 5s infinite alternate;
        }}
        .content-text {{
            font-size: 28px !important;
            font-weight: bold;
            text-align: center;
            color: black;
        }}
        .subtitle {{
            font-size: 22px;
            font-style: italic;
            font-weight: bold;
            color: #555;
            text-align: center;
            margin-bottom: -5px;
            display: block;
        }}
        @keyframes color-change {{
            0% {{ color: #ff4b4b; }}
            50% {{ color: #4b9bff; }}
            100% {{ color: #4bff88; }}
        }}
        .report-section {{
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 10px;
            margin: 10px;
            width: 80%;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar Navigation
st.sidebar.title("Navigation")
navigation_options = {
    "🏠 Home": "home",
    "📊 Analyze Caregiver": "analyze",
    "📁 Upload Transcript File": "upload",
    "📥 Download Results": "download",
}
choice = st.sidebar.radio("Go to", list(navigation_options.keys()))

# Sayfalara Özel Arka Planlar
if navigation_options[choice] == "home":
    set_background("homepage3.jpg")
    
    # **Başlığın üstüne profesyonel mesaj**
    st.markdown(
        """
        <p class="subtitle">"Ensuring comfort, care, and love for your little one."</p>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<p class='dynamic-text'>Caregiver Monitoring & Evaluation System</p>", unsafe_allow_html=True)
    st.subheader("Welcome to the Caregiver Monitoring & Evaluation System!")
    st.markdown(
        """
        <div class="content-text">
            <p>This tool evaluates caregiver-child interactions using AI.</p>
            <p><b>Analyzes sentiment, empathy, and engagement.</b></p>
            <p><b>Classifies conversations into caregiving categories (Nutrition, Learning, etc.).</b></p>
            <p><b>Scores caregiver performance (1-5).</b></p>
            <p><b>Generates automated feedback for parents.</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif navigation_options[choice] == "analyze":
    set_background("homepage.avif")
    st.header("Analyze a Caregiver-Child Conversation")
    transcript_text = st.text_area("Paste a caregiver-child transcript below:")

    if st.button("Analyze Transcript"):
        if transcript_text.strip():
            with st.spinner("Analyzing... Please wait."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(orchestrator.process_transcript(transcript_text))
                
                if result:
                    st.subheader("Caregiver Performance Report")
                    df_metrics = pd.DataFrame({
                        "Metric": ["Sentiment", "Category", "Caregiver Score"],
                        "Value": [
                            result.get("sentiment", "N/A"),
                            result.get("primary_category", "N/A"),
                            f"{result.get('caregiver_score', 'N/A')}/5"
                        ]
                    })
                    st.table(df_metrics)

                    st.markdown('<div class="report-section"><h3>Additional Insights</h3>', unsafe_allow_html=True)
                    st.write(f"**Tone:** {result.get('tone', 'N/A')}")
                    st.write(f"**Empathy Level:** {result.get('empathy', 'N/A')}")
                    st.write(f"**Responsiveness:** {result.get('responsiveness', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="report-section"><h3>Justification</h3>', unsafe_allow_html=True)
                    st.write(result.get("justification", "No justification available."))
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="report-section"><h3>Parent Notification</h3>', unsafe_allow_html=True)
                    st.write(result.get("parent_notification", "No summary available."))
                    st.markdown('</div>', unsafe_allow_html=True)

                    recommendations = result.get("recommendations", [])
                    st.markdown('<div class="report-section"><h3>Recommendations for Improvement</h3>', unsafe_allow_html=True)
                    if recommendations:
                        for rec in recommendations:
                            st.write(f"**{rec.get('category', 'General')}**: {rec.get('description', 'No details provided.')}")
                    else:
                        st.write("No specific recommendations.")
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("Failed to process the transcript. Please try again.")
        else:
            st.error("Please enter a transcript to analyze.")

elif navigation_options[choice] == "upload":
    set_background("baby.avif")
    st.header("Upload a Caregiver-Child Conversation Transcript")
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file:
        transcript_text = uploaded_file.read().decode("utf-8")
        with st.spinner("Analyzing uploaded transcript..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(orchestrator.process_transcript(transcript_text))

            if result:
                st.subheader("Caregiver Performance Report")
                df_metrics = pd.DataFrame({
                    "Metric": ["Sentiment", "Category", "Caregiver Score"],
                    "Value": [
                        result.get("sentiment", "N/A"),
                        result.get("primary_category", "N/A"),
                        f"{result.get('caregiver_score', 'N/A')}/5"
                    ]
                })
                st.table(df_metrics)

elif navigation_options[choice] == "download":
    set_background("lightmodern.avif")
    st.header("Download Processed Results")
    st.error("No previous analysis found. Please analyze a transcript first.")
