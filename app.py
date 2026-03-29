import streamlit as st
import requests
import pandas as pd
import plotly.express as px

FASTAPI_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

st.title("🎥 YouTube Comment Sentiment Analyzer")
st.markdown("Analyze the sentiment of a YouTube video's comment section using our Quantized XLM-RoBERTa model.")

url_input = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
analyze_button = st.button("Analyze Sentiment")

if analyze_button and url_input:
    with st.spinner("Fetching comments and running inference (this may take a moment)..."):
        try:
            response = requests.post(FASTAPI_URL, json={"url": url_input, "max_comments": 100})
            
            if response.status_code == 200:
                data = response.json()
                
                if "message" in data:
                    st.warning(data["message"])
                else:
                    st.success(f"Successfully analyzed {data['total_analyzed']} comments!")
                    
                    # Create Tabs for the two views
                    tab1, tab2 = st.tabs(["Overall Sentiment", "Comment by Comment"])
                    
                    # View 1: Overall Sentiment
                    with tab1:
                        st.subheader(f"Dominant Sentiment: **{data['overall_sentiment']}**")
                        
                        # Prepare data for pie chart
                        df_percentages = pd.DataFrame(
                            list(data["percentages"].items()), 
                            columns=["Sentiment", "Percentage"]
                        )
                        
                        # Plotly Pie Chart
                        fig = px.pie(
                            df_percentages, 
                            values="Percentage", 
                            names="Sentiment",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#28a745", 
                                "Neutral": "#ffc107", 
                                "Negative": "#dc3545"
                            },
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # View 2: Comment by Comment
                    with tab2:
                        st.subheader("Individual Comment Breakdown")
                        df_comments = pd.DataFrame(data["comments"])
                        
                        # Apply color coding to the sentiment column
                        def color_sentiment(val):
                            color = 'green' if val == 'Positive' else 'red' if val == 'Negative' else 'orange'
                            return f'color: {color}'
                            
                        st.dataframe(
                            df_comments.style.map(color_sentiment, subset=['sentiment']),
                            use_container_width=True,
                            height=400
                        )

            else:
                st.error(f"Error from backend: {response.json().get('detail', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Is the FastAPI server running?")