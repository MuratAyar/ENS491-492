import streamlit as st
import pandas as pd
import os
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from agents.orchestrator import Orchestrator

# Initialize the orchestrator
orchestrator = Orchestrator()

# Function to clean the uploaded CSV file
def clean_csv_file(uploaded_file):
    try:
        # Save the uploaded file to a temporary local file
        temp_input_path = "temp_uploaded_file.csv"
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        cleaned_file_path = "cleaned_reviews.csv"
        with open(temp_input_path, "r", errors="replace") as infile, open(cleaned_file_path, "w", newline="") as outfile:
            reader = csv.reader(infile, quoting=csv.QUOTE_MINIMAL)  # Properly handle quotes
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)  # Ensure quotes in output
            for row in reader:
                try:
                    writer.writerow(row)
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
        return cleaned_file_path
    except Exception as e:
        st.error(f"Error during file cleaning: {e}")
        st.stop()


# Streamlit UI
st.title("AI Google Play Store Reviews Responder App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Home", "Upload CSV", "Analyze Single Review", "Download Results", "Visualization"]
choice = st.sidebar.radio("Go to", options)

# Home Page
if choice == "Home":
    st.write("""
    ## Welcome to the AI Google Play Store Reviews Responder App!
    This app allows you to:
    - Upload a CSV file of reviews and analyze them.
    - Analyze a single review and get sentiment, category, and response.
    - Download the processed results as a CSV file.
    - Generate visualizations from the analyzed results.
    """)

# Upload CSV Page
elif choice == "Upload CSV":
    st.header("Upload Reviews CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        # Clean the uploaded CSV file
        cleaned_file_path = clean_csv_file(uploaded_file)

        # Read the cleaned CSV file
        try:
            reviews_df = pd.read_csv(
                cleaned_file_path,
                quoting=3,  # Treat all quotes as regular characters
                escapechar="\\",  # Handle escape sequences
                on_bad_lines="warn",  # Skip problematic rows
                engine="python"  # Use the Python engine for flexibility
            )
            st.write("Uploaded Reviews:")
            st.dataframe(reviews_df, height=400)

            # Check for required columns
            if "translated_content" not in reviews_df.columns or "score" not in reviews_df.columns:
                st.error("The uploaded file must contain 'translated_content' and 'score' columns!")
                st.stop()

            # Always display the "Analyze" button after successful upload
            if st.button("Process Reviews"):
                with st.spinner("Processing reviews..."):
                    results = []
                    for index, row in reviews_df.iterrows():
                        review = row["translated_content"]
                        score = row["score"]
                        result = asyncio.run(orchestrator.process_review(review))
                        result["score"] = score
                        results.append(result)

                    results_df = pd.DataFrame(results)
                    results_path = "results.csv"
                    results_df.to_csv(results_path, index=False)
                    st.success("Reviews processed successfully!")
                    st.write("Processed Results:")
                    st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error reading cleaned CSV: {e}")
            st.stop()



# Rest of your Streamlit app remains unchanged


# Analyze Single Review Page
elif choice == "Analyze Single Review":
    st.header("Analyze a Single Review")

    review_text = st.text_area("Enter a review to analyze:")

    if st.button("Analyze Review"):
        if review_text.strip():
            with st.spinner("Analyzing review..."):
                result = asyncio.run(orchestrator.process_review(review_text))
                
                # Display the result with better formatting
                st.write("### Analysis Result")
                st.subheader("Review")
                st.write(result["review"])

                st.subheader("Analyzing Sentiment")
                st.write(result["analyzing_sentiment"])

                st.subheader("Category")
                st.write(result["category"])

                st.subheader("Response")
                st.write(result["response"])
        else:
            st.error("Please enter a review to analyze!")


# Download Results Page
elif choice == "Download Results":
    st.header("Download Processed Results")
    if os.path.exists("results.csv"):
        with open("results.csv", "rb") as file:
            st.download_button(
                label="Download Results",
                data=file,
                file_name="results.csv",
                mime="text/csv",
            )
    else:
        st.error("No results file found! Please process reviews first.")

# Visualization Page
elif choice == "Visualization":
    st.header("Generate Visualizations")
    uploaded_file = st.file_uploader("Upload Processed Results CSV", type="csv")

    if uploaded_file:
        try:
            reviews_df = pd.read_csv(
                uploaded_file,
                quoting=csv.QUOTE_MINIMAL,  # Handle quotes correctly
                escapechar="\\",           # Handle escape sequences
                on_bad_lines="skip",       # Skip problematic rows
                engine="python"            # Use Python engine for flexibility
            )

            # Validate column headers
            expected_columns = ["review", "analyzing_sentiment", "category", "response", "score", "expected_stars"]
            if not all(column in reviews_df.columns for column in expected_columns):
                st.error(f"CSV file is not formatted correctly. Expected columns: {expected_columns}")
                st.stop()

            # Display the uploaded dataframe
            st.dataframe(reviews_df, height=400)

            # Generate visualizations if the file is correct
            if st.button("Generate Visualizations"):
                with st.spinner("Generating visualizations..."):
                    # Compare Score vs Expected Stars
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=reviews_df, x="score", y="expected_stars", errorbar="sd")
                    plt.title("Score vs Expected Stars")
                    plt.xlabel("User Score")
                    plt.ylabel("Expected Stars")
                    st.pyplot(plt)

                    # Number of reviews per category
                    plt.figure(figsize=(10, 6))
                    category_counts = reviews_df["category"].value_counts()
                    sns.barplot(x=category_counts.index, y=category_counts.values)
                    plt.title("Number of Reviews per Category")
                    plt.xlabel("Category")
                    plt.ylabel("Number of Reviews")
                    st.pyplot(plt)

                    # Sentiment distribution
                    plt.figure(figsize=(10, 6))
                    sns.countplot(data=reviews_df, x="analyzing_sentiment")
                    plt.title("Sentiment Distribution")
                    plt.xlabel("Sentiment")
                    plt.ylabel("Count")
                    st.pyplot(plt)

                    # Score vs. Expected Stars
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=reviews_df, x="score", y="expected_stars", hue="analyzing_sentiment", style="category", s=100)
                    plt.title("Score vs. Expected Stars")
                    plt.xlabel("User Given Score")
                    plt.ylabel("Model Predicted Expected Stars")
                    plt.legend(title="Sentiment")
                    st.pyplot(plt)

                    st.success("Visualizations generated successfully!")
        except pd.errors.ParserError as e:
            st.error(f"Parser error: {e}. Please check the CSV file for formatting issues.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}. Please check the file format.")
            st.stop()
