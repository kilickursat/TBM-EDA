import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page layout to 'wide'
st.set_page_config(layout='wide')

# Page 1: Introduction
def page_intro():
    image = Image.open('Kursat_Artificial_intelligence_and_TBM.png')
    st.success('Welcome!')
    st.image(image, caption='Intelligent-TBM-by-Midjourney', width=500)

    st.markdown('''
        # **The TBM EDA App**

        This is the **TBM EDA App** created in Streamlit using the **pandas-profiling** and **pyWalker** libraries.

        **Credit:** App built in `Python` + `Streamlit Cloud` + `ChatGPT` + `pyWalker` by [Kursat Kilic](https://github.com/kilickursat) (Researcher for TUST&AI field)

        ---
    ''')

    # Option to choose online or batch data loading
    data_loading_option = st.radio("Select data loading option:", ("Online Data", "Batch Data"))

    if data_loading_option == "Online Data":
        # Online data loading code
        pass
    elif data_loading_option == "Batch Data":
        # Batch data loading code
        pass

# Page 2: Random Forest Application
def page_random_forest():
    st.header('RandomForest Regression for Batch Data')

    # Assuming you have a 'target_column' that you want to predict
    target_column = st.text_input("Enter the name of the target column:", "default_target_column")

    # Batch data loading code (copy from the Batch Data section of your previous code)
    if st.button("Load Batch Data"):
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
            st_profile_report(pr)

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

            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R-squared: {r2}')

# App Navigation
app_pages = {
    "Introduction": page_intro,
    "Random Forest Application": page_random_forest,
}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(app_pages.keys()))

# Display the selected page
app_pages[page]()
