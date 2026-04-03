import streamlit as st
import json
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Luxury Interiors",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Luxury Interior Design Theme
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
    
    .luxury-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        opacity: 0.9;
        margin-top: 1rem;
    }
    
    .product-card {
        background: linear-gradient(145deg, #ffffff, #f0f4f8);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .product-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .price-tag {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #1a1a1a;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.4rem;
        box-shadow: 0 8px 20px rgba(255,215,0,0.3);
    }
    
    .add-to-cart-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        margin-top: 1rem;
    }
    
    .add-to-cart-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(102,126,234,0.4);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
    }
    
    .stMetric > label {
        color: white !important;
        font-size: 1.2rem;
    }
    
    .stMetric > div > div {
        color: white !important;
        font-size: 2.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load products
@st.cache_data
def load_products():
    products_data = {
        "sofas": [
            {"id": 1, "name": "Velvet Chesterfield Sofa", "price": 2999, "category": "sofas", "image": "🛋️", "stock": 12},
            {"id": 2, "name": "Modern L-Shape Sofa", "price": 2499, "category": "sofas", "image": "🛋️", "stock": 8},
            {"id": 3, "name": "Luxury Leather Sectional", "price": 4999, "category": "sofas", "image": "🛋️", "stock": 5}
        ],
        "tables": [
            {"id": 4, "name": "Marble Coffee Table", "price": 899, "category": "tables", "image": "☕", "stock": 15},
            {"id": 5, "name": "Oak Dining Table", "price": 1799, "category": "tables", "image": "🍽️", "stock": 10},
            {"id": 6, "name": "Glass Side Table", "price": 299, "category": "tables", "image": "☕", "stock": 20}
        ],
        "chairs": [
            {"id": 7, "name": "Eames Lounge Chair", "price": 1299, "category": "chairs", "image": "🪑", "stock": 18},
            {"id": 8, "name": "Velvet Armchair", "price": 799, "category": "chairs", "image": "🪑", "stock": 25},
            {"id": 9, "name": "Bar Stool Set", "price": 599, "category": "chairs", "image": "🪑", "stock": 30}
        ],
        "lighting": [
            {"id": 10, "name": "Crystal Chandelier", "price": 2499, "category": "lighting", "image": "💡", "stock": 6},
            {"id": 11, "name": "Modern Floor Lamp", "price": 399, "category": "lighting", "image": "💡", "stock": 22},
            {"id": 12, "name": "Wall Sconces (Pair)", "price": 299, "category": "lighting", "image": "💡", "stock": 15}
        ],
        "decor": [
            {"id": 13, "name": "Persian Rug 8x10", "price": 2999, "category": "decor", "image": "🧳", "stock": 4},
            {"id": 14, "name": "Wall Art Set", "price": 599, "category": "decor", "image": "🎨", "stock": 12},
            {"id": 15, "name": "Marble Vase", "price": 199, "category": "decor", "image": "🪴", "stock": 35}
        ]
    }
    return [item for sublist in products_data.values() for item in sublist]

products = load_products()

# Session state
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'total' not in st.session_state:
    st.session_state.total = 0
if 'selected' not in st.session_state:
    st.session_state.selected = "catalog"

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🏠 Luxury Interiors")
    selected = option_menu(
        menu_title=None,
        options=["🏠 Catalog", "🛒 Cart", "💳 Checkout", "📊 Dashboard"],
        icons=["house", "cart", "credit-card", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0!important", "--hover-color": "#eee"},
            "nav-link-selected": {"background": "rgba(255, 255, 255, 0.2)"},
        }
    )

# Header
st.markdown("""
    <div class="main-header">
        <h1 class="luxury-title">Luxury Interiors</h1>
        <p class="subtitle">Premium Furniture & Home Decor for Discerning Tastes</p>
    </div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
        <div class="metric-container">
            <div class="stMetric">
                <label>Total Products</label>
                <div>{}</div>
            </div>
        </div>
        """.format(len(products)),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div class="metric-container">
            <div class="stMetric">
                <label>Cart Items</label>
                <div>{}</div>
            </div>
        </div>
        """.format(len(st.session_state.cart)),
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="metric-container">
            <div class="stMetric">
                <label>Order Total</label>
                <div>${:,.2f}</div>
            </div>
        </div>
        """.format(st.session_state.total),
        unsafe_allow_html=True
    )
with col4:
    total_stock = sum(p['stock'] for p in products)
    st.markdown(
        """
        <div class="metric-container">
            <div class="stMetric">
                <label>In Stock</label>
                <div>{}</div>
            </div>
        </div>
        """.format(total_stock),
        unsafe_allow_html=True
    )

# Page Content
if selected == "🏠 Catalog":
    st.header("✨ Premium Collection")
    
    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search products...", key="search")
    with col2:
        price_range = st.slider("💰 Price Range", 0, 5000, (0, 5000), key="price_range")
    
    # Filter products
    filtered_products = [
        p for p in products 
        if (not search or search.lower() in p['name'].lower()) and 
           price_range[0] <= p['price'] <= price_range[1]
    ]
    
    # Category tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛋️ Sofas", "☕ Tables", "🪑 Chairs", "💡 Lighting", "🎨 Decor"])
    
    with tab1:
        sofas = [p for p in filtered_products if p['category'] == 'sofas']
        display_products(sofas)
    
    with tab2:
        tables = [p for p in filtered_products if p['category'] == 'tables']
        display_products(tables)
    
    with tab3:
        chairs = [p for p in filtered_products if p['category'] == 'chairs']
        display_products(chairs)
    
    with tab4:
        lighting = [p for p in filtered_products if p['category'] == 'lighting']
        display_products(lighting)
    
    with tab5:
        decor = [p for p in filtered_products if p['category'] == 'decor']
        display_products(decor)

elif selected == "🛒 Cart":
    st.header("🛒 Shopping Cart")
    if not st.session_state.cart:
        st.info("👋 Your cart is empty. Start shopping!")
    else:
        display_cart()

elif selected == "💳 Checkout":
    st.header("💳 Secure Checkout")
    if not st.session_state.cart:
        st.warning("🛒 Add items to cart first!")
    else:
        st.success(f"**Order Total: ${st.session_state.total:,.2f}**")
        st.button("✅ Complete Order", type="primary", on_click=complete_order)

elif selected == "📊 Dashboard":
    st.header("📊 Sales Dashboard")
    show_dashboard()

# Helper Functions
def display_products(products_list):
    for product in products_list:
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.markdown(f"### {product['image']}")
            with col2:
                st.markdown(f"""
                    <div class="product-card">
                        <h3>{product['name']}</h3>
                        <p><strong>In Stock:</strong> {product['stock']} units</p>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="price-tag">${product['price']:,.2f}</div>
                """, unsafe_allow_html=True)
            
            if st.button(f"🛒 Add to Cart", key=f"add_{product['id']}"):
                add_to_cart(product)
                st.rerun()

def add_to_cart(product):
    cart_item = next((item for item in st.session_state.cart if item['id'] == product['id']), None)
    if cart_item:
        cart_item['quantity'] += 1
    else:
        st.session_state.cart.append({'id': product['id'], **product, 'quantity': 1})
    st.session_state.total = sum(item['price'] * item['quantity'] for item in st.session_state.cart)
    st.success(f"✅ {product['name']} added to cart!")

def display_cart():
    for i, item in enumerate(st.session_state.cart):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{item['name']}**")
        with col2:
            st.write(f"Qty: {item['quantity']}")
        with col3:
            st.write(f"${item['price']:,.2f}")
        with col4:
            st.write(f"${(item['price'] * item['quantity']):,.2f}")
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.cart.pop(i)
                st.session_state.total = sum(item['price'] * item['quantity'] for item in st.session_state.cart)
                st.rerun()

def complete_order():
    st.session_state.cart = []
    st.session_state.total = 0
    st.balloons()
    st.success("🎉 Order placed successfully! Thank you for shopping with Luxury Interiors!")

def show_dashboard():
    df = pd.DataFrame(products)
    fig = px.scatter(df, x='price', y='stock', size='price', 
                    color='category', hover_name='name',
                    title="Product Analytics")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='category', values='stock')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(df, x='category', y='price', title="Avg Price by Category")
        st.plotly_chart(fig_bar, use_container_width=True)
