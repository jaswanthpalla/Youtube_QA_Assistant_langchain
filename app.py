import streamlit as st
from youtube_qa import answer_question

st.title("YouTube Video Q&A with Hugging Face API & Langchain")
st.write("Enter a YouTube video URL, your Hugging Face API key, and ask a question about the video's content.")

api_key = st.text_input("Hugging Face API Key", type="password")
video_url = st.text_input("YouTube Video URL (e.g., https://www.youtube.com/watch?v=ssmwkXPFMU)")
question = st.text_input("Question")
if st.button("Get Answer"):
    if api_key and video_url and question:
        with st.spinner("Processing..."):
            answer = answer_question(video_url, question, api_key)
            st.write("**Answer:**")
            st.write(answer)
    else:
        st.error("Please provide API key, video URL, and question.")
