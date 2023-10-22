import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
import streamlit.components.v1 as components

image = Image.open('Kursat_Artificial_intelligence_and_TBM.png')

st.success('Welcome!')

st.image(image, caption='Intelligent-TBM-by-Midjourney',width=500)
# Add two JPG images, one aligned left and the other aligned right
# Web App Title
st.markdown('''
# **The TBM EDA App**

This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** and **pyWalker** libraries.

**Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT` + `pyWalker` by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field))

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
        online_button = st.button('Press to Use Online Dataset')

        if online_button:
            # Load the online dataset
            online_data_link = "https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx"
            df = pd.read_excel(online_data_link, engine='openpyxl')
            
            # Show the data as a table 
            st.dataframe(df)
            # Show statistics on the data
            st.write(df.describe())

            # Generate the HTML using PygWalker
            st.header('orange[pyWalker EDA]')
            st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just doing drag and drop")
            pyg_html = pyg.to_html(df,hideDataSourceConfig=True,themekey="vega",dark="media")  # Replace 'pyg' with the correct pygWalker object

            # Embed the HTML into the Streamlit app
            components.html(pyg_html, height=1000, scrolling=True)

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
def download_report(report):
    """Downloads the report as a PDF or HTML file.
    Args:
        report: A ydata_profiling.ProfileReport object.
    """

    file_type = st.selectbox('Select file type:', ['PDF', 'HTML'])
    if file_type == 'PDF':
        report.to_pdf('report.pdf')
    elif file_type == 'HTML':
        report.to_html('report.html')
    pr = ProfileReport(df, explorative=True)
    st.button('Download report', on_click=download_report, args=(report,))

df = load_data(uploaded_file)
def plot_pygwalker(df):
  """Plots the data using pyWalker.

  Args:
    df: A Pandas DataFrame containing the data to plot.
  """

  pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")

  # Add a filter to the plot
  pyg_html += '''
    <script>
      pyg.filter({
        'type': 'dropdown',
        'label': 'Filter',
        'values': [
          {
            'label': 'All data',
            'value': 'all'
          },
          {
            'label': 'Data with A > 10',
            'value': 'A > 10'
          }
        ]
      });
    </script>
  '''

  # Embed the HTML into the Streamlit app
  components.html(pyg_html, height=1000, scrolling=True)

# Plot the data
plot_pygwalker(df)







    
        #pr = ProfileReport(df, explorative=True)
        # Generate the HTML using PygWalker
        #st.header('pyWalker Page')
        #st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just doing drag and drop")
        #pyg_html = pyg.to_html(df,hideDataSourceConfig=True,themekey="vega",dark="media")  # Replace 'pyg' with the correct pygWalker object
        
        # Embed the HTML into the Streamlit app
        #components.html(pyg_html, height=1000, scrolling=True)

        # Embed the HTML into the Streamlit app
        #components.html(pyg_html, height=1000, scrolling=True)
        #st.header('**Input DataFrame**')
        #st.write(df)
        #st.header('**Pandas Profiling Report**')
        #st_profile_report(pr)
        
st.link_button("Go to pyWalker", "https://docs.kanaries.net/pygwalker")
st.link_button("Go to pandas-profiling","https://github.com/ydataai/ydata-profiling")

