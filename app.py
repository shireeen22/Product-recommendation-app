import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

@st.cache_resource
def load_data():
    df = pd.read_csv('Product_df.csv') # Load the df...

    # Load the tfidf_matrix...
    with open('tfidf_model.pkl','rb') as f:
        tfidf_matrix = pickle.load(f)

    # Load the cosine_matrix...
    with open('cosine_sim.pkl','rb') as f:
        cosine_sim = pickle.load(f)

    return df, tfidf_matrix, cosine_sim

df, tfidf_matrix, cosine_sim = load_data() # store the return values...

# Recommendation function...

def recommend_products_category(product_input, n=5):

    # Determine whether input is ProductID or Product name
    if str(product_input).isdigit():
        idx_list = df.index[df['ProductID'] == int(product_input)].tolist()
    else:
        idx_list = df.index[df['Product'].str.lower() == str(product_input).lower()].tolist()

    # If product not found
    if len(idx_list) == 0:
        return f"Product '{product_input}' not found!"

    idx = idx_list[0]

    # Get category of input product
    category = df.loc[idx, 'Category_x']

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Store unique recommended products
    recommended = []
    added_products = set()   # prevent duplicates

    for i, score in sim_scores:
        if len(recommended) == n:
            break

        if i == idx:
            continue  # skip same product

        # Check same category
        if df.loc[i, 'Category_x'] == category:

            product_name = df.loc[i, 'Product']

            # Prevent duplicate names
            if product_name not in added_products:
                recommended.append(i)
                added_products.add(product_name)

    # Return selected recommendations
    return df[['ProductID', 'Product', 'Category_x']].iloc[recommended]

# Dictonary for {Product:images}...
product_images = {
    'Wall Clock': 'clock.jpg',
    'Ceramic Mug': 'mug.jpg',
    'Desk Organizer': 'Desk Organizer.jpg',
    'Home Decor Lamp': 'home decor lamps.jpg',
    'Storage Box': 'storage new.jpg',
    'Bluetooth Speaker': 'Bluetooth Speaker.jpg',
    'LED Light Strip': 'Led light strip.jpg',
    'Notebook': 'notebook new.jpg',
    'Wireless Mouse': 'wireless mouse.jpg',
    'Water Bottle': 'Water Bottle.jpg',
    'Color-Changing Aroma Lamp': 'Color-Changing Aroma Lamp.jpg',
    'Digital Kitchen Scale': 'Digital Kitchen Scale.jpg',
    'Solar Garden Fairy Lights': 'solar garden fairy lights.jpg',
    'Portable Power Bank (10,000 mAh)': 'Portable Power Bank.jpg',
    'Under-Bed Storage Bag': 'bad_storage.jpg',
    'Geometric Wall Shelf': 'geoGeometric Wall Shelf.jpg',
    'Decorative Hourglass Timer': 'hour glass.jpg',
    'Rotating Desk Pen Holder': 'Rotating Desk Pen Holder.jpg',
    'Fragrance Oil Diffuser': 'Fragrance Oil Diffuser.jpg',
    'Motion Sensor Night Light': 'Motion Sensor Night Light.jpg',
    'Reusable Metal Straw Kit': 'Reusable Metal Straw Kit.jpg',
    'Hanging Closet Organizer': 'Hanging Closet Organizer.jpg',
    'Silicone Baking Mat Set': 'Silicone Baking Mat Set.jpg',
    'Wireless Number Pad': 'Wireless Number Pad.jpg',
    'Blue-Light Blocking Glasses': 'Light Blocking Glasses.jpg'
}
# Create a function for image loading...
def image_loader(product_name):
    path = product_images.get(product_name,'images/default.jpg')
    return path if os.path.exists(path) else 'images/default.jpg'

# Create a price_df for unique products...
price_df = (df.groupby('Product')['Price'].mean().reset_index()) # Avg. price...

# Stremlit...

st.sidebar.title('Nevigation📌')
menu = st.sidebar.radio("Select Option",['Product Recommendation','Trending Produts','ProductPrice'])
# if Product Recommendation selected...
if menu == 'Product Recommendation':
    st.title('Product Recommendation Dashboard💫🔍')
    st.image('recommendation.png',use_container_width=False)
    st.subheader('Enhance your shopping experience with recommended produts💖')
    # Dropdown for product selection...
    prod_list = df['Product'].unique() # to get all the unique product names...
    selected_prod = st.selectbox("Select a product here👇",prod_list) # store the selected product...
    # Get the recommendations...
    if st.button('Recommend'):
        recomm = recommend_products_category(selected_prod)
        st.subheader('Top Recommended products⭐⭐⭐⭐⭐')

        for _, row in recomm.iterrows():
            with st.container():
                col1, col2, = st.columns([1,3])
                with col1:
                    img_path = image_loader(row["Product"])
                    if os.path.exists(img_path):
                        st.image(img_path,width=100)
                    else:
                        st.image('images/default.jpg',width=100)
                with col2:
                    st.subheader(row['Product'])
                    st.write(f'Category: {row['Category_x']}')
            st.divider()
    # if Trending products selected...
elif menu == 'Trending Produts':
    st.title("Trending Products are here🔥")
    st.image('tranding products.webp',use_container_width=False)
    trending_prod = (df.sort_values('Recency').drop_duplicates(subset=['Product']).head(10))# sorting...
    for _, row in trending_prod.iterrows():
        col1, col2 = st.columns([1,3])
        with col1:
            img_path = image_loader(row["Product"])
            if os.path.exists(img_path):
                st.image(img_path,width=100)
        with col2:
            st.subheader(row['Product'])
            st.write(f'Category: {row['Category_x']}')
            st.write(f'Price: {row['Price']}')
        st.divider()
else:
    st.title('Product price page💰')
    st.subheader("Know the average,get the best spot of price🤗✨")
    for _, row in price_df.iterrows():
        col1, col2 = st.columns([1,4])
        with col1:
            img = image_loader(row['Product'])
            st.image(img,width=140)
        with col2:
            st.subheader(row['Product'])
            st.write(f'Price: {round(row['Price'],2)}')
        st.divider()
 


                
                
      

