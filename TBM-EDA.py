import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
import streamlit.components.v1 as components

col1, col2 = st.columns(2)  # Set up two columns for the initial content

# Set up the layout in the first column (col1)
with col1:
    #st.set_page_config(layout='wide')
    
    image = Image.open('Kursat_Artificial_intelligence_and_TBM.png')
    st.success('Welcome!')
    st.image(image, caption='Intelligent-TBM-by-Midjourney', width=500)
    
    st.markdown('''
        # **The TBM EDA App**

        This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** and **pyWalker** libraries.

        **Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT` + `pyWalker` by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field)

        ---
    ''')

# Define components for online data loading in the second column (col2)
with col2:
    data_loading_option = st.radio("Select data loading option:", ("Online Data", "Batch Data"))

    if data_loading_option == "Online Data":
        with st.sidebar:
            st.header('Press to use Online Data Loading')
            st.markdown("You can choose to load the online dataset or upload your own CSV or Excel file. The example dataset is from TBM literature")
            st.markdown("[Example Online Dataset](https://github.com/kilickursat/WebApp/blob/main/TBM_Performance.xlsx)")
            online_button = st.button('Press to Use Online Dataset')

        if online_button:
            online_data_link = "https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx"
            df = pd.read_excel(online_data_link, engine='openpyxl')
            st.dataframe(df)
            st.write(df.describe())

            pr = ProfileReport(df, explorative=True)
            st.header('orange[pyWalker EDA]')
            st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just by doing drag and drop")
            pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")
            components.html(pyg_html, height=1000, scrolling=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

# Separate column for batch data loading
col3 = st.columns(1)[0]

if data_loading_option == "Batch Data":
    with col3:
        with st.sidebar:
            st.header('Batch Data Loading')
            st.markdown("You can choose to upload your own CSV or Excel file.")

        uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
        st.markdown("[Example excel input file](https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx)")

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
            st.header('pyWalker Page')
            st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just by doing drag and drop")
            pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")
            components.html(pyg_html, height=1000, scrolling=True)
            st.header('**Input DataFrame**')
            st.write(df)

st.link_button("Go to pyWalker", "https://docs.kanaries.net/pygwalker")
st.link_button("Go to pandas-profiling", "https://github.com/ydataai/ydata-profiling")
