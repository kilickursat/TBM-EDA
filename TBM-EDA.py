import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pickle

# Web App Title
st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit using the **pandas-profiling** library.

**Credit:** App built in `Python` + `Streamlit` by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))

---
''')

# Upload CSV or Excel data
with st.sidebar.header('1. Upload your CSV or Excel data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv", "xlsx"])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

# Online Data Loading
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

    # Save the EDA application to a pickle file
    with open('eda_app.pkl', 'wb') as pickle_file:
        pickle.dump({'data': df, 'report': pr}, pickle_file)
else:
    st.info('Awaiting for CSV or Excel file to be uploaded.')
    if st.button('Press to use Example Dataset'):
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
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

        # Save the EDA application to a pickle file
        with open('eda_app.pkl', 'wb') as pickle_file:
            pickle.dump({'data': df, 'report': pr}, pickle_file)
