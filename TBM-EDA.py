import pickle
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport

def load_data(mode='online'):
  """Loads data from a CSV or Excel file.

  Args:
    mode: The data loading mode. Can be either 'online' or 'batch'.

  Returns:
    A Pandas DataFrame containing the loaded data.
  """

  if mode == 'online':
    # Load data from an uploaded file
    uploaded_file = st.file_uploader('Upload a CSV or Excel file')
    if uploaded_file is None:
      return None
    else:
      return pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

  elif mode == 'batch':
    # Load data from a pre-existing file
    file_path = st.text_input('Enter the path to the CSV or Excel file')
    if file_path is None or file_path == '':
      return None
    else:
      return pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

  else:
    raise ValueError('Invalid data loading mode: {}'.format(mode))

# Load the data
df = load_data(mode='online')

# Pickle the data
with open('data.pickle', 'wb') as f:
  pickle.dump(df, f)

# Load the pickled data
with open('data.pickle', 'rb') as f:
  df = pickle.load(f)

# Create a ydata-profiling report
report = ydata_profiling.ProfileReport(df)

# Display the report in Streamlit
st.title('TBM EDA Report')
st.write(report.to_html())
