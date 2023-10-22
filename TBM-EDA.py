import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
import streamlit.components.v1 as components


# Web App Title
st.markdown('''
# **The TBM EDA App**

This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** library.

**Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT` by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field))

---
''')
# Option to choose online or batch data loading
data_loading_option = st.radio("Select data loading option:", ("Online Data", "Batch Data"))


# Online Data Loading
if data_loading_option == "Online Data":
    with st.sidebar:
        st.header('Press to use Online Data Loading')
        st.markdown("You can choose to load the online dataset or upload your own CSV or Excel file. The example dataset is from TBM literature")
        st.markdown("[Example Online Dataset](https://github.com/kilickursat/WebApp/blob/main/TBM_Performance.xlsx)")
        online_button = st.button('Use Online Dataset')

        if online_button:
            # Load the online dataset
            online_data_link = "https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx"
            df = pd.read_excel(online_data_link, engine='openpyxl')
            # Show the data as a table
            st.dataframe(df)
            # Show statistics on the data
            st.write(df.describe())

            pr = ProfileReport(df, explorative=True)

            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

# Batch Data Loading
if data_loading_option == "Batch Data":
    with st.sidebar:
        st.header('Batch Data Loading')
        st.markdown("You can choose to upload your own CSV or Excel file.")
    
    uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
    st.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

    if uploaded_file is not None:
        @st.cache
        def load_data(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                return pd.read_excel(file, engine='openpyxl')

        df = load_data(uploaded_file)
        st.dataframe(df)
        st.write(df.describe())
        pr = ProfileReport(df, explorative=True)

        st.header('**Input DataFrame**')
        st.write(df)
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
        

image = Image.open('Kursat_Artificial_intelligence_and_a_tbm.png')

st.image(image, caption='Intelligent-TBM')
# Add two JPG images, one aligned left and the other aligned right
