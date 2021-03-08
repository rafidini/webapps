# Imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import ssl
import urllib.request, json 
import datetime

def process_json(items):
    frame = pd.DataFrame(None)
    items = items.get('features')

    if not items:
        return None

    for item in items:
        series = pd.Series(item['properties'])

        try:
            series['longitude'] = item['geometry']['coordinates'][0]
            series['latitude'] = item['geometry']['coordinates'][1]

        except:
            pass

        frame = frame.append(series.to_frame().T, ignore_index=True)

    return frame

# Deal with SSL request
ssl._create_default_https_context = ssl._create_unverified_context

# Config
st.set_page_config(
    page_title="Earthquakes USGS",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title('🌎 Earthquakes USGS')

# Data loading preparation
BASE_URL = """
https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time}&endtime={end_time}&minmagnitude={minmagnitude}&maxmagnitude={maxmagnitude}
"""

@st.cache
def load_data(start_time=None, end_time=None, minmagnitude=None, maxmagnitude=None):
    # Format API request
    frame = pd.DataFrame(None)
    request_url = BASE_URL.format(start_time=start_time, end_time=end_time, minmagnitude=minmagnitude,
    maxmagnitude=maxmagnitude)

    # URL Request
    with urllib.request.urlopen(request_url) as url:
        data = json.loads(url.read().decode())

    data = process_json(data)

    return data

data_load_state = st.text('Loading data...')

# Sidebar
st.sidebar.title('Sidebar')
start_time = st.sidebar.date_input("Start time", datetime.datetime(2014, 1, 1))
end_time = st.sidebar.date_input("End time", datetime.datetime(2014, 1, 2))
magnitude = st.sidebar.slider('Magnitude (Richter scale)', 0.0, 10.0, (2.0, 7.0))
st.sidebar.text('Generated url')
st.sidebar.write(BASE_URL.format(start_time=start_time.strftime("%Y-%m-%d"), end_time=end_time.strftime("%Y-%m-%d"), minmagnitude=magnitude[0], maxmagnitude=magnitude[1]))

# Load the data
data = load_data(start_time=start_time.strftime("%Y-%m-%d"), end_time=end_time.strftime("%Y-%m-%d"), minmagnitude=magnitude[0], maxmagnitude=magnitude[1])
data_load_state.text(f"Done, {data.shape[0]} events loaded!")

# World Map
st.subheader('World Map')
st.map(data)

col1, col2, col3 = st.beta_columns([5, 1, 5])

with col1:
    # Histogram
    st.subheader('Magnitude distribution')
    fig, ax = plt.subplots(1, 1)
    sns.distplot(data["mag"], color='brown')
    st.pyplot(fig)

with col2:
    pass

with col3:
    # Histogram
    st.subheader('Significance distribution')
    fig, ax = plt.subplots(1, 1)
    sns.distplot(data["sig"], color='brown')
    st.pyplot(fig)
