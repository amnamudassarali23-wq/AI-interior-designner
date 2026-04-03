import streamlit as st
import pandas as pd
# Import plotly with a fallback check
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from datetime import datetime

# Page config
st.set_page_config(
    page_title="🏠 Luxury Interiors",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .luxury-title { font-family: 'Playfair Display', serif; font-size: 3rem; margin: 0; }
    .subtitle { font-family: 'Inter', sans-serif; font-size: 1.2rem; opacity: 0.9; margin-top: 1rem; }
    .product-card { 
        background: linear-gradient(145deg, #ffffff, #f8f9ff); 
        border-radius: 15px; 
        padding: 1.5rem; 
        margin: 1rem 0; 
        border: 1px solid #e1e8ed;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    .price-tag { 
        background: linear-gradient(45deg, #ffd700, #ffed4e); 
        color: #1a1a1a; 
        padding: 10px 20px; 
        border-radius: 25px; 
        font-weight: bold; 
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Data
products = [
    {"id": 1, "name": "Velvet Chesterfield Sofa", "price": 2999, "category": "Sofas", "emoji": "🛋️", "stock": 12},
    {"id": 2, "name": "Modern L-Shape Sofa", "price": 2499, "category": "Sofas", "emoji": "🛋️", "stock": 8},
    {"id": 3, "name": "Luxury Leather Sectional", "price": 4999, "category": "Sofas", "emoji": "🛋️", "stock": 5},
    {"id": 4, "name": "Marble Coffee Table", "price": 899, "category": "Tables", "emoji": "☕", "stock": 15},
    {"id": 5, "name": "Oak Dining Table", "price": 1799, "category": "Tables", "emoji": "🍽️", "stock": 10},
    {"id": 6, "name": "Glass Side Table", "price": 299, "category": "Tables", "emoji": "☕", "stock": 20},
    {"id": 7, "name": "Eames Lounge Chair", "price": 1299, "category": "Chairs", "emoji": "🪑", "stock": 18},
    {"id": 8, "name": "Velvet Armchair", "price": 799, "category": "Chairs", "emoji": "🪑", "stock": 25},
    {"id": 9, "name": "Bar Stool Set", "price": 599, "category": "Chairs", "emoji": "🪑", "stock": 30},
    {"id": 10, "name": "Crystal Chandelier", "price": 2499, "category": "Lighting", "emoji": "💡", "stock": 6},
    {"id": 11, "name": "Modern Floor Lamp", "price": 399, "category": "Lighting", "emoji": "💡", "stock": 22},
    {"id": 12, "name": "Wall Sconces", "price": 299, "category": "Lighting", "emoji": "💡", "stock": 15},
    {"id": 13, "name": "Persian Rug 8x10", "price": 2999, "category": "Decor", "emoji": "🧳", "stock": 4},
    {"id": 14, "name": "Wall Art Set", "price": 599, "category": "Decor", "emoji": "🎨", "stock": 12},
    {"id": 15, "name": "Marble Vase", "price": 199, "category": "Decor", "emoji": "🪴", "stock": 35}
]

if 'cart' not in st.session_state: st.session_state.cart = []
if 'total' not in st.session_state: st.session_state.total = 0.0

# Header
st.markdown("""<div class="main-header"><h1 class="luxury-title">🏠 Luxury Interiors</h1><p class="subtitle">Premium Furniture & Home Decor</p></div>""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Choose Page", ["Catalog", "Cart", "Checkout", "Analytics"])

if page == "Catalog":
    st.header("✨ Premium Collection")
    search_term = st.text_input("🔍 Search products...")
    filtered = [p for p in products if not search_term or search_term.lower() in p['name'].lower()]
    
    for product in filtered:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col2:
            st.markdown(f'<div class="product-card"><h3>{product["emoji"]} {product["name"]}</h3><p>{product["category"]}</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='price-tag'>${product['price']:,}</div>", unsafe_allow_html=True)
            if st.button("🛒 Add", key=f"btn_{product['id']}"):
                st.session_state.cart.append(product)
                st.session_state.total += product['price']
                st.toast(f"Added {product['name']}!")

elif page == "Cart":
    st.header("🛒 Your Cart")
    if not st.session_state.cart:
        st.info("Empty cart.")
    else:
        for item in st.session_state.cart:
            st.write(f"• {item['name']} - ${item['price']}")
        st.metric("Total Bill", f"${st.session_state.total:,.2f}")
        if st.button("Clear Cart"):
            st.session_state.cart = []
            st.session_state.total = 0.0
            st.rerun()

elif page == "Analytics":
    st.header("📊 Business Analytics")
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is not installed. Please check your requirements.txt file.")
    else:
        df = pd.DataFrame(products)
        fig = px.pie(df, names='category', values='stock', title="Inventory Distribution")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Checkout":
    st.header("💳 Checkout")
    name = st.text_input("Name")
    if st.button("Complete Order") and name:
        st.balloons()
        st.success(f"Thank you, {name}!")
