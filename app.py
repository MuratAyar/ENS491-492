import streamlit as st
import asyncio
from agents.orchestrator import Orchestrator

# Initialize the orchestrator
orchestrator = Orchestrator()

# Streamlit UI
st.title("Caregiver Monitoring & Evaluation System")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Home", "Analyze Caregiver", "Upload Transcript File", "Download Results"]
choice = st.sidebar.radio("Go to", options)

# Home Page
if choice == "Home":
    st.write("""
    ## Welcome to the Caregiver Monitoring & Evaluation System!  
    This tool evaluates caregiver-child interactions using AI.  
    - **Analyzes sentiment, empathy, and engagement**.  
    - **Classifies conversations into caregiving categories** (Nutrition, Learning, etc.).  
    - **Scores caregiver performance (1-5)**.  
    - **Generates automated feedback for parents**.  
    """)

# Analyze Caregiver Page
elif choice == "Analyze Caregiver":
    st.header("Analyze a Caregiver-Child Conversation")

    transcript_text = st.text_area("Paste a caregiver-child transcript below:")

    if st.button("Analyze Transcript"):
        if transcript_text.strip():
            with st.spinner("Analyzing conversation..."):
                result = asyncio.run(orchestrator.process_transcript(transcript_text))
                st.session_state["last_result"] = result  # 🔹 Store result in session state

                # Display results
                st.write("### Caregiver Performance Report")
                
                st.subheader("Sentiment Analysis")
                st.write(f"**Sentiment:** {result.get('sentiment', 'N/A')}")
                st.write(f"**Tone:** {result.get('tone', 'N/A')}")
                st.write(f"**Empathy Level:** {result.get('empathy', 'N/A')}")
                st.write(f"**Responsiveness:** {result.get('responsiveness', 'N/A')}")

                st.subheader("Categorization")
                st.write(f"**Primary Category:** {result.get('primary_category', 'N/A')}")
                st.write(f"**Secondary Categories:** {', '.join(result.get('secondary_categories', []))}")

                st.subheader("Caregiver Score")
                st.write(f"**Performance Score:** {result.get('caregiver_score', 'N/A')}/5")
                st.write(f"**Justification:** {result.get('justification', 'N/A')}")

                st.subheader("Parent Notification")
                st.write(result.get("parent_notification", "N/A"))
                st.write("**Recommendations:**", result.get("recommendations", "N/A"))
        else:
            st.error("Please enter a transcript to analyze.")

# Upload Transcript File Page
elif choice == "Upload Transcript File":
    st.header("Upload a Caregiver-Child Conversation Transcript")

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file:
        transcript_text = uploaded_file.read().decode("utf-8")

        with st.spinner("Analyzing uploaded transcript..."):
            result = asyncio.run(orchestrator.process_transcript(transcript_text))
            st.session_state["last_result"] = result  # 🔹 Store result in session state

            # Display results
            st.write("### Caregiver Performance Report")

            st.subheader("Sentiment Analysis")
            st.write(f"**Sentiment:** {result.get('sentiment', 'N/A')}")
            st.write(f"**Tone:** {result.get('tone', 'N/A')}")
            st.write(f"**Empathy Level:** {result.get('empathy', 'N/A')}")
            st.write(f"**Responsiveness:** {result.get('responsiveness', 'N/A')}")

            st.subheader("Categorization")
            st.write(f"**Primary Category:** {result.get('primary_category', 'N/A')}")
            st.write(f"**Secondary Categories:** {', '.join(result.get('secondary_categories', []))}")

            st.subheader("Caregiver Score")
            st.write(f"**Performance Score:** {result.get('caregiver_score', 'N/A')}/5")
            st.write(f"**Justification:** {result.get('justification', 'N/A')}")

            st.subheader("Parent Notification")
            st.write(result.get("parent_notification", "N/A"))
            st.write("**Recommendations:**", result.get("recommendations", "N/A"))

# 🔹 FIXED: Download Processed Results Page
elif choice == "Download Results":
    st.header("Download Processed Results")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]  # Retrieve last analysis result

        if st.button("Download"):
            with open("caregiver_report.txt", "w") as file:
                file.write("Caregiver Monitoring Report\n")
                file.write("=" * 40 + "\n")
                file.write(f"Sentiment: {result.get('sentiment', 'N/A')}\n")
                file.write(f"Tone: {result.get('tone', 'N/A')}\n")
                file.write(f"Empathy Level: {result.get('empathy', 'N/A')}\n")
                file.write(f"Responsiveness: {result.get('responsiveness', 'N/A')}\n\n")
                file.write(f"Primary Category: {result.get('primary_category', 'N/A')}\n")
                file.write(f"Secondary Categories: {', '.join(result.get('secondary_categories', []))}\n\n")
                file.write(f"Performance Score: {result.get('caregiver_score', 'N/A')}/5\n")
                file.write(f"Justification: {result.get('justification', 'N/A')}\n\n")
                file.write(f"Parent Notification:\n{result.get('parent_notification', 'N/A')}\n")
                file.write(f"Recommendations:\n{result.get('recommendations', 'N/A')}\n")

            with open("caregiver_report.txt", "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="caregiver_report.txt",
                    mime="text/plain",
                )
    else:
        st.error("No previous analysis found. Please analyze a transcript first.")
