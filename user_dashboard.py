import streamlit as st
from utils import save_review, generate_user_response

st.set_page_config(page_title="Feedback User Dashboard", layout="wide")

st.title("🌟 We Value Your Feedback!")
st.write("Please rate your experience and leave a review.")

with st.form("review_form"):
    rating = st.slider("Rating (1-5)", 1, 5, 5)
    review_text = st.text_area("Your Review", placeholder="Tell us what you liked or how we can improve...")
    
    submitted = st.form_submit_button("Submit Review")
    
    if submitted:
        if not review_text.strip():
            st.error("Please enter a review text.")
        else:
            with st.spinner("Processing your feedback..."):
                # Generate AI Response
                ai_reply = generate_user_response(rating, review_text)
                
                # Save Data
                save_review(rating, review_text, ai_reply)
                
                st.success("Thank you for your feedback!")
                
                # Display AI Response
                st.markdown("### Our Response")
                st.info(ai_reply)
