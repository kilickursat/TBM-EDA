import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
import pygwalker as pyg
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve  

# Set page layout to 'wide'
st.set_page_config(layout='wide')

# Initialize the model variable
model =  RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize X and y
X = df.drop(target_column, axis=1)
y = df[target_column]

# Page image and markdowns
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
        st.header('[pyWalker EDA]')
        st.markdown("This is the pyWalker. Please play with X-axis and Y-axis just by doing drag and drop")
        pyg_html = pyg.to_html(df, hideDataSourceConfig=True, themekey="vega", dark="media")
        components.html(pyg_html, height=1000, scrolling=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

if data_loading_option == "Batch Data":
    with st.sidebar:
        st.header('Batch Data Loading')
        st.markdown("You can choose to upload your own CSV or Excel file.")
    
    uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])
    st.markdown("[Example excel input file](https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx)")

    # Add an input field to let the user specify the target column
    target_column = st.text_input("🚀**Enter the name of the target column:**", "Please enter the name of your target column")

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
        
        st.header('Pandas Profiling Report for Batch Data')
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
        # Plot Predicted vs. Actual
        st.header('Predicted vs. Actual Values')

        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs. Actual Values')
        st.pyplot()

        # Plot Learning Curve
        st.header('Learning Curve')

# Define a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    axes.legend(loc="best")
    plt.show()

# Plot the learning curve
plot_learning_curve(model, "Learning Curve", X, y)
st.pyplot()
    
st.link_button("Go to pyWalker", "https://docs.kanaries.net/pygwalker")
st.link_button("Go to pandas-profiling", "https://github.com/ydataai/ydata-profiling")

