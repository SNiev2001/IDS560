'source .venv/bin/activate'

import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="YouTube Trending Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    # Set the path to the file you'd like to load
    file_path = ""

    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rsrishav/youtube-trending-video-dataset",
        "US_youtube_trending_data.csv",
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Sidebar Navigation
st.sidebar.image("https://www.example.com/sample-logo.png", use_container_width=True)
st.sidebar.title("📊 YouTube Trending Analysis")
menu = st.sidebar.radio("📌 Select a Page", ["Overview", "Trending Videos", "Analytics", "Predictive Model", "About Us"])

# Overview Page
if menu == "Overview":
    st.title("🚀 YouTube Trending Video Analysis")
    st.markdown("### Gain insights into trending videos and predict their longevity.")
    st.image("https://www.example.com/sample-image.jpg", use_container_width=True)
    
    # Dropdown for category selection
    category_selection = st.selectbox("Select a Category", df['categoryId'].unique())
    st.write(f"You selected: {category_selection}")

# Trending Videos
elif menu == "Trending Videos":
    st.title("🔥 Trending Videos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Top Trending Categories")
        category_counts = df['categoryId'].value_counts().reset_index()
        fig = px.bar(category_counts, x='index', y='categoryId', labels={'index': 'Category', 'categoryId': 'Count'},
                     title="Trending Video Categories", color='categoryId')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔝 Top Trending Videos")
        top_videos = df[['title', 'channelTitle', 'view_count', 'likes']].sort_values(by='view_count', ascending=False).head(10)
        st.dataframe(top_videos)

# Analytics Section
elif menu == "Analytics":
    st.title("📊 YouTube Video Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='categoryId', title='🎥 Video Categories Distribution', color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x='categoryId', y='view_count', title='📈 Views Distribution by Category', color='categoryId')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis on Video Descriptions")
    df['sentiment'] = df['description'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
    fig3 = px.histogram(df, x='sentiment', title='📊 Sentiment Distribution', color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig3, use_container_width=True)
    
    # WordCloud for Keywords
    st.subheader("🔍 Most Frequent Words in Descriptions")
    text = " ".join(desc for desc in df['description'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Predictive Model
elif menu == "Predictive Model":
    st.title("🎯 Predict Video Longevity")
    title_input = st.text_input("Enter Video Title", placeholder="E.g. Amazing Science Experiment")
    category_input = st.selectbox("Select Category", df['categoryId'].unique())
    
    if st.button("🚀 Predict"):
        st.warning("Model not available yet. Please check back later!")

# About Us
elif menu == "About Us":
    st.title("📢 About This App")
    st.markdown("""
        This interactive dashboard provides insights into YouTube's trending videos, allowing users to analyze trends, predict video longevity, and explore analytics.
        
        **Features:**
        - 📊 Data Visualization for trending videos
        - 🔥 AI-powered trend predictions
        - 🎭 Sentiment analysis of audience reactions
        - 🌎 Regional comparisons
        
        Developed with ❤️ for content creators and data analysts!
    """)

st.sidebar.success("💡 Developed with ❤️ for YouTube content insights.")
