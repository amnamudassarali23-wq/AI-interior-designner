import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config - Fixed
st.set_page_config(
    page_title="🏠 Luxury Interiors",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed & Optimized
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
.add-cart-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    border: none !important;
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

# Fixed Products Data - No JSON issues
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

# Initialize session state - Fixed
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'total' not in st.session_state:
    st.session_state.total = 0.0

# Header
st.markdown("""
<div class="main-header">
    <h1 class="luxury-title">🏠 Luxury Interiors</h1>
    <p class="subtitle">Premium Furniture & Home Decor Collection</p>
</div>
""", unsafe_allow_html=True)

# Luxury Metrics - Fixed
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Total Products</h3>
        <h2 style='color: white; font-size: 2.5rem;'>15</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Cart Items</h3>
        <h2 style='color: white; font-size: 2.5rem;'>{len(st.session_state.cart)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Order Total</h3>
        <h2 style='color: white; font-size: 2.5rem;'>${st.session_state.total:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_stock = sum(p['stock'] for p in products)
    st.markdown(f"""
    <div class="metric-card">
        <h3>In Stock</h3>
        <h2 style='color: white; font-size: 2.5rem;'>{total_stock}</h2>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Navigation - Fixed (No external dependency)
st.sidebar.title("🏠 Navigation")
page = st.sidebar.selectbox("Choose Page", ["Catalog", "Cart", "Checkout", "Analytics"])

# Main Pages
if page == "Catalog":
    st.header("✨ Premium Collection")
    
    # Filters
    col1, col2 = st.columns([3, 2])
    with col1:
        search_term = st.text_input("🔍 Search products...")
    with col2:
        min_price, max_price = st.slider("💰 Price Range", 0, 5000, (0, 5000))
    
    # Filter products
    filtered = [p for p in products 
                if (not search_term or search_term.lower() in p['name'].lower())
                and min_price <= p['price'] <= max_price]
    
    # Category selection
    category = st.selectbox("Select Category", ["All"] + list(set(p['category'] for p in products)))
    if category != "All":
        filtered = [p for p in filtered if p['category'] == category]
    
    # Display products
    if filtered:
        for product in filtered:
            col1, col2, col3 = st.columns([1, 4, 2])
            with col1:
                st.markdown(f"**{product['emoji']}**")
            with col2:
                st.markdown(f"""
                <div class="product-card">
                    <h3>{product['name']}</h3>
                    <p>📦 In Stock: {product['stock']} | 🏷️ {product['category']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='price-tag'>${product['price']:,.0f}</div>", unsafe_allow_html=True)
                if st.button("🛒 Add to Cart", key=f"add_{product['id']}", help=f"Add {product['name']}"):
                    st.session_state.cart.append(product)
                    st.session_state.total += product['price']
                    st.success(f"✅ {product['name']} added!")
                    st.rerun()
    else:
        st.info("🔍 No products found. Try different filters.")

elif page == "Cart":
    st.header("🛒 Shopping Cart")
    if not st.session_state.cart:
        st.info("👋 Your cart is empty. Start shopping!")
    else:
        cart_df = pd.DataFrame(st.session_state.cart)
        cart_df['Subtotal'] = cart_df['price']
        st.dataframe(cart_df[['name', 'price', 'category', 'Subtotal']], use_container_width=True)
        
        st.markdown(f"### 💰 **Total: ${st.session_state.total:,.0f}**")
        if st.button("🗑️ Clear Cart", type="secondary"):
            st.session_state.cart = []
            st.session_state.total = 0.0
            st.rerun()

elif page == "Checkout":
    st.header("💳 Secure Checkout")
    if not st.session_state.cart:
        st.warning("🛒 Your cart is empty!")
    else:
        st.success(f"**Order Summary**")
        for item in st.session_state.cart:
            st.write(f"• {item['name']} - ${item['price']:,.0f}")
        
        st.markdown(f"### 💰 **Grand Total: ${st.session_state.total:,.0f}**")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email")
        with col2:
            address = st.text_area("Delivery Address")
            phone = st.text_input("Phone")
        
        if st.button("✅ Place Order", type="primary"):
            st.balloons()
            st.success(f"🎉 Order placed successfully for ${name}!")
            st.session_state.cart = []
            st.session_state.total = 0.0
            st.rerun()

elif page == "Analytics":
    st.header("📊 Business Analytics")
    df = pd.DataFrame(products)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df, names='category', values='stock', title="Stock by Category")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(df, x='category', y='price', title="Average Price by Category")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("💰 Price Distribution")
    fig_hist = px.histogram(df, x='price', nbins=10, title="Product Price Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)
