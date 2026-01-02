import streamlit as st
try:
    from openai import OpenAI
    import os
except ImportError as e:
    st.error(f"Library Import Error: {e}")
    st.stop()

st.set_page_config(page_title="AI Interior Designer", layout="centered")

# Accessing the API Key safely
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key not found! Please add OPENAI_API_KEY to your Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("🏡 AI Interior Designer")

# Inputs
room = st.selectbox("Room Type", ["Living Room", "Bedroom", "Kitchen"])
color = st.color_picker("Pick a primary color", "#00f900")
style = st.text_input("Design Style", "Modern Minimalist")

if st.button("Generate Design"):
    with st.spinner("Generating..."):
        try:
            prompt = f"A professional 8k interior design of a {room} with {color} color palette, style is {style}."
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1
            )
            st.image(response.data[0].url)
        except Exception as e:
            st.error(f"AI Generation Error: {e}")
