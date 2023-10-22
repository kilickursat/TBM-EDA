import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.markdown('''
# **The TBM EDA App**

This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** library.

**Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT`by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field))

---
''')

# Option to choose online or batch data loading
data_loading_option = st.radio("Select data loading option:", ("Online Data", "Batch Data"))

# Online Data Loading
if data_loading_option == "Online Data":
    with st.sidebar:
        st.header('Online Data Loading')
        uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
        st.markdown("""
        [Example CSV input file](https://github.com/kilickursat/WebApp/blob/main/TBM_Performance.xlsx)
        """)

    if uploaded_file is not None:
        @st.cache
        def load_data(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                return pd.read_excel(file, engine='openpyxl')

        df = load_data(uploaded_file)
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
        st.markdown("You can choose to load an example dataset or upload your own CSV or Excel file.")
        example_button = st.button('Use Example Dataset')

        if example_button:
            # Example data
            @st.cache
            def load_example_data():
                a = pd.DataFrame(
                    np.random.rand(100, 5),
                    columns=['a', 'b', 'c', 'd', 'e']
                )
                return a

            df = load_example_data()
            pr = ProfileReport(df, explorative=True)

            st.header('**Input DataFrame**')
            st.write(df)
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

        else:
            uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
            st.markdown("""
            [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
            """)

            if uploaded_file is not None:
                @st.cache
                def load_data(file):
                    if file.name.endswith('.csv'):
                        return pd.read_csv(file)
                    elif file.name.endswith('.xlsx'):
                        return pd.read_excel(file, engine='openpyxl')

                df = load_data(uploaded_file)
                pr = ProfileReport(df, explorative=True)

                st.header('**Input DataFrame**')
                st.write(df)
                st.header('**Pandas Profiling Report**')
                st_profile_report(pr)
