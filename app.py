import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client using the secret you pasted in Streamlit Cloud
# Make sure the secret name in Streamlit is exactly: OPENAI_API_KEY
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("⚠️ API Key missing! Go to App Settings > Secrets and add: OPENAI_API_KEY = 'your_key'")
    st.stop()

# Page Configuration
st.set_page_config(page_title="AI Interior Designer", page_icon="🏡")

st.title("🏡 Smart AI Interior Designer")
st.info("Input your room details below and let AI design your space.")

# --- USER INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    room_name = st.text_input("What is this room called?", placeholder="e.g. CEO's Office, Cozy Bedroom")
    room_shape = st.selectbox("Room Shape", ["Square", "Rectangular", "L-Shaped", "Open Floor Plan"])

with col2:
    budget = st.select_slider("Select Budget Range", options=["Economy", "Standard", "Premium", "Ultra-Luxury"])
    theme_color = st.color_picker("Pick a Primary Theme Color", "#3498db")

# --- GENERATION LOGIC ---
if st.button("Generate My Interior Design ✨", use_container_width=True):
    if not room_name:
        st.warning("Please give your room a name first!")
    else:
        # Constructing the AI Prompt
        prompt = (
            f"A high-end, photorealistic 8k interior design for a {room_name}. "
            f"The room shape is {room_shape}. The design style should reflect a {budget} budget "
            f"with a color palette focusing on {theme_color}. Professional lighting, "
            f"detailed textures, architectural photography style."
        )
        
        with st.spinner(f"Designing your {room_name}... This may take 20 seconds."):
            try:
                # API Call to DALL-E 3
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard"
                )
                
                # Display the Result
                image_url = response.data[0].url
                st.divider()
                st.subheader(f"✨ Result: {room_name} ({budget} Style)")
                st.image(image_url, caption=f"AI Interpretation of your {room_shape} {room_name}")
                st.success("Design Generated Successfully!")
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
