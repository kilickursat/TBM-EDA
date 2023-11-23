import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Set page layout to 'wide'
st.set_page_config(layout='wide')

# Page image and markdowns
image = Image.open('Firefly Illustration.jpg')
st.success('Welcome!')
st.image(image, caption='Intelligent-TBM-by-Adobe Firefly', width=500)

st.markdown('''
    # **The TBM EDA App**

    This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** and **pyGWalker** libraries.

    **Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT` + `pyGWalker` by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field)

    ---
''')

# Sidebar navigation
def sidebar():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Go to:', ('Main Page', 'Online Data', 'Batch Data'))

    if selection == 'Main Page':
        main_page()
    elif selection == 'Online Data':
        online_data()
    elif selection == 'Batch Data':
        batch_data()

# Main page content
def main_page():
    st.write("This is the main page of the TBM EDA App.")

# Online data section content
# Online data section content
def online_data():
    st.header('Online Data Section')
    st.write('This is the online data section.')

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
        st.header('pyGWalker - tableau')
        st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just by doing drag and drop")
        pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")
        components.html(pyg_html, height=1000, scrolling=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

# Batch data section content
def batch_data():
    st.header('Batch Data Section')
    st.write('This is the batch data section.')

    with st.sidebar:
        st.header('Batch Data Loading')
        st.markdown("You can choose to upload your own CSV or Excel file.")

    uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
    st.markdown("[Example excel input file](https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx)")

    # Add an input field to let the user specify the target column
    st.header("**Random forest regressor**")
    target_column = st.text_input("**Enter the name of the target column:**", "Please type the name of the target column in your dataset.")

    if uploaded_file is not None:
        @st.cache_data
        def load_data(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                return pd.read_excel(file, engine='openpyxl')

        df = load_data(uploaded_file)
        st.dataframe(df)
        st.write(df.describe())

        st.header('Pandas Profiling Report for Batch Data')
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)

        st.header('pyGWalker - tableau')
        st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just by doing drag and drop")
        pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")
        components.html(pyg_html, height=1000, scrolling=True)

        # Adding RandomForest regression for batch data
        st.header('RandomForest Regression for Batch Data')

        # Split the data into features (X) and the user-specified target variable (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)

        st.write(f'Mean Squared Error: {round(mse, 2)}')
        st.write(f'R-squared: {round(r2, 2)}')
        st.write(f'Mean Absolute Error: {round(mae, 2)}')
       
        # Plot Predicted vs. Actual
        st.header('Predicted vs. Actual Values')
        
        # Create a figure with 600 DPI
        fig, ax = plt.subplots(dpi=600)  
        
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs. Actual Values')
        st.pyplot(fig)


        # Convert y_test and y_pred to numpy arrays for indexing
        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)
        
        # Fit a linear regression line to the scatter plot
        m, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, m * y_test + b, color='red', label='Best Fit Line')

        ax.legend()
        st.pyplot(fig)

    st.link_button("Go to pyGWalker", "https://docs.kanaries.net/pygwalker")
    st.link_button("Go to pandas-profiling", "https://github.com/ydataai/ydata-profiling")
    st.link_button("Go to random forest regressor","https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html")


if __name__ == '__main__':
    sidebar()
