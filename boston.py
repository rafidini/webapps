# Imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ssl

from sklearn.datasets import load_boston

# Deal with SSL request
ssl._create_default_https_context = ssl._create_unverified_context

# Config
st.set_page_config(
    page_title="Boston House Prices - Beginner",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title('Boston House Prices Dataset')

# Load the data
@st.cache
def load_data():
    # Load the dataset
    data = load_boston()

    # Process
    frame = pd.DataFrame(data["data"], columns=data['feature_names'])
    frame['TARGET'] = data['target']
    return frame

# Info text for loading state
data_load_state = st.markdown('Loading data...')
data = load_data()
data_load_state.text(f"Done!")

# Show table if checkbox checked
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Numerical column distribution
distribution_header = st.subheader('Variable distribution')
selected_column = st.selectbox('Select a column', options=data.columns.tolist(), key="col-histogram")
log_scale = st.checkbox('Log scale?')
distribution_col = data[selected_column].apply(np.log) if log_scale else data[selected_column]

if st.checkbox('Show boxplot?'):
    fig, ax = plt.subplots(2, 1)
    sns.kdeplot(distribution_col, ax=ax[0], color='violet')
    sns.boxplot(distribution_col, ax=ax[1], color='violet')
else:
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(distribution_col, ax=ax, color='violet')

st.pyplot(fig)
st.write(distribution_col.describe().transpose())

# Correlation
st.subheader('Correlation')
options = st.multiselect( 'Which columns?', options=data.columns.tolist(), default=data.columns.tolist())
annotations = st.checkbox('Annotations?')
fig, ax = plt.subplots(1, 1)
corr = data[options].corr() # Correlation matrix
mask = np.triu(np.ones_like(corr, dtype=bool)) # Upper triangle mask
sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, cmap='vlag', annot=annotations, annot_kws={"fontsize":6})
st.pyplot(fig)

# Scatter plot
st.subheader('Scatter plot')
col1, col2 = st.beta_columns(2)
xaxis, yaxis = None, None

with col1:
    xaxis = st.selectbox('x-axis', options=data.columns.tolist(), key="x-scatterplot")

with col2:
    yaxis = st.selectbox('y-axis', options=data.columns.tolist(), key="y-scatterplot")

fig, ax = plt.subplots(1, 1)
sns.scatterplot(x=data[xaxis], y=data[yaxis], ax=ax,  alpha=.5, color='violet')
st.pyplot(fig)

# Regression
st.subheader('Regression')

# Log scaling target
target_log = st.checkbox("Log scale target? (recommended)")

# Feature selection
features = data.drop(columns='TARGET').columns.tolist()
options = st.multiselect('Which features', options=features, default=features)

# Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

models = [
    LinearRegression(),
    KNeighborsRegressor(5),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

X = data[features]
y = data['TARGET'].apply(np.log) if target_log else data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

results = dict()
results['Model'] = list()
results['Train'] = list()
results['Test'] = list()
for model in models:
    # Fit the model
    model.fit(X_train, y_train)

    # RMSE
    train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)

    # Append
    results['Model'].append(type(model).__name__)
    results['Train'].append(train_rmse)
    results['Test'].append(test_rmse)

st.markdown('Models results')
st.markdown(f"""
- Train no. of rows : {X_train.shape[0]} 
- Test  no. of rows : {X_test.shape[0]} 
- Train/Test splitting : **0.3**
- Scores : **RMSE**, mean_squared_error from sklearn package with *squared=False*
""")

frame_results = pd.DataFrame.from_dict(results).set_index('Model')
st.dataframe(frame_results)

st.markdown(f'**{frame_results[frame_results["Test"] == frame_results["Test"].min()].index.values[0]}** may be the best ✅')

st.markdown('**Feature importances** only for GradientBoostingRegressor...')

GBR = GradientBoostingRegressor()
GBR.fit(X, y)

importances = pd.Series(GBR.feature_importances_, index=X.columns)
fig, ax = plt.subplots(1, 1)
importances.apply(np.abs).sort_values(ascending=True).plot.barh(color='violet')
st.pyplot(fig)
