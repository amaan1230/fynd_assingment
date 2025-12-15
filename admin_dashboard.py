import streamlit as st
import pandas as pd
from utils import load_reviews, analyze_reviews_for_admin

st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("📊 Customer Feedback Admin Dashboard")

# Load Data
reviews = load_reviews()

if not reviews:
    st.info("No reviews yet.")
else:
    # Stats
    df = pd.DataFrame(reviews)
    avg_rating = df['rating'].mean()
    total_reviews = len(df)
    
    col1, col2 = st.columns(2)
    col1.metric("Average Rating", f"{avg_rating:.1f}/5")
    col2.metric("Total Reviews", total_reviews)
    
    # Display Table
    st.subheader("Live Feed")
    st.dataframe(df[['timestamp', 'rating', 'text', 'ai_response']].sort_values(by='timestamp', ascending=False))
    
    # AI Analysis Section
    st.markdown("---")
    st.header("🤖 AI Insights")
    
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing recent reviews..."):
            summary, recommendations = analyze_reviews_for_admin(reviews)
            
            st.subheader("Executive Summary")
            st.write(summary)
            
            st.subheader("Recommended Actions")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
