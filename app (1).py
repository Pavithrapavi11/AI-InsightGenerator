import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import zipfile
import os
from io import BytesIO

# --- Configuration and Secrets ---
# The app will automatically find OPENAI_API_KEY from Streamlit Cloud secrets

# --- Utility Function to Load Data from ZIP ---
# @st.cache_data makes data loading fast after the first run
@st.cache_data
def load_data_from_zip(uploaded_file):
    """Extracts a CSV or Excel file from the uploaded ZIP and loads it into a DataFrame."""
    try:
        # Create an in-memory ZIP file object
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Look for the first CSV or Excel file
            data_files = [f for f in file_list if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
            
            if not data_files:
                return None, None

            file_to_load = data_files[0]
            
            with zip_ref.open(file_to_load) as file:
                if file_to_load.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                else: # .xlsx or .xls
                    df = pd.read_excel(file)
                return df, file_to_load
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

# --- Generative AI Function ---
def generate_insights(df, task):
    """Generates business insights using the OpenAI API."""
    
    # 1. Prepare Prompt Data
    head_text = df.head(3).to_markdown(index=False)
    summary_text = df.describe(include='number').to_markdown()

    prompt = f"""
    You are an expert business data analyst. Analyze the provided dataset and generate a concise report 
    with key business insights and actionable recommendations.
    
    Data Head (First 3 Rows):
    {head_text}
    
    Data Summary (Statistics):
    {summary_text}
    
    Specific Request/Task: {task}
    
    Provide your analysis and recommendations in a professional, markdown-formatted business report.
    """
    
    # 2. Call OpenAI API
    try:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
             return "⚠️ **OpenAI API Key Not Found.** Please set your `OPENAI_API_KEY` in the Streamlit Cloud Secrets."
             
        # Initialize the OpenAI client using the key from secrets
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.5 
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred during AI generation. Error: {e}"


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="AI-Powered Insight Generator")
st.title("💡 AI-Powered Business Insight Generator")
st.markdown("Automate business reports with data analysis and Generative AI.")

# 1. Data Upload and Configuration Sidebar
with st.sidebar:
    st.header("1️⃣ Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your ZIP file containing a single CSV or Excel file.",
        type=["zip"]
    )
    
    st.markdown("---")
    st.header("⚙️ AI Configuration")
    ai_task = st.text_area(
        "What specific insights/recommendations do you need?",
        "Identify top 3 revenue drivers and suggest strategies to reduce customer churn.",
        height=100
    )
    generate_button = st.button("Generate Business Report")

df = None
file_name = None

if uploaded_file is not None:
    # Load data only if a file is uploaded
    df, file_name = load_data_from_zip(uploaded_file)
    
    if df is not None:
        st.success(f"Successfully loaded data from **{file_name}**.")
    else:
        st.error("Could not load a valid CSV or Excel file from the ZIP.")

if df is not None:
    
    # --- Data Display and Summary ---
    st.header("2️⃣ Data Overview")
    
    tab1, tab2 = st.tabs(["Data Preview", "Statistical Summary"])
    
    with tab1:
        st.subheader(f"Data Preview ({df.shape[0]} rows, {df.shape[1]} columns)")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all'), use_container_width=True)

    st.markdown("---")

    # --- Data Visualization ---
    st.header("3️⃣ Key Visualizations")
    
    numeric_cols = df.select_dtypes(include='number').columns
    
    if len(numeric_cols) > 0:
        
        col_plot, col_select = st.columns([3, 1])
        
        with col_select:
            # Let the user pick a column to plot
            selected_col = st.selectbox(
                "Select a numeric column for visualization:", 
                numeric_cols, 
                index=0
            )

        with col_plot:
            st.subheader(f"Distribution of {selected_col}")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, bins=30) 
            st.pyplot(fig)
            
    else:
        st.info("⚠️ No numeric columns found for plotting.")

    st.markdown("---")

    # --- Generative AI Insight ---
    st.header("4️⃣ AI-Powered Report")
    
    if generate_button:
        with st.spinner("Analyzing data and generating report..."):
            report = generate_insights(df.copy(), ai_task)
            st.markdown(report)
            
else:
    # Initial instruction when no file is uploaded
    st.info("👆 Please upload a ZIP file containing your business data (CSV or Excel) to begin the analysis and generate the report.")
