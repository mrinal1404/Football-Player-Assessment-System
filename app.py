import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

st.title("Player Assessment and Generative AI for Football Data")

# File uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    
    # Basic descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Visualize distributions for a few key features
    st.subheader("Feature Distributions")
    features_to_plot = df.columns.drop('overall_rating', errors='ignore')[:5]
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(16, 4))
    for ax, col in zip(axes, features_to_plot):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(25,12))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Drop non-feature columns if necessary
    if 'id' in df.columns:
        df_processed = df.drop("id", axis=1)
    else:
        df_processed = df.copy()

    # Define target and features (overall_rating is target)
    target = "overall_rating"
    feature_cols = df_processed.columns.drop(target)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed[feature_cols])
    y = df_processed[target].values

    # -------------------------------
    # Section 1: Clustering for Player Assessment
    # -------------------------------
    st.header("Player Clustering for Assessment")
    
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform KMeans clustering (e.g., 3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # Add cluster labels to the DataFrame for analysis
    df_clustered = df_processed.copy()
    df_clustered["Cluster"] = clusters

    # Visualize clusters using scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=50)
    ax.set_title("Player Clusters (PCA & KMeans)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    st.write("The clustering groups players with similar performance characteristics. Below is a summary for each cluster:")
    
    # Calculate cluster summary statistics
    cluster_summary = df_clustered.groupby("Cluster").mean().round(2)
    st.subheader("Cluster Summary Statistics")
    st.write(cluster_summary)
    
    # Interpretation text based on overall_rating means in each cluster
    st.write("""
    **Cluster Interpretation:**
    - Clusters with higher average overall ratings might represent high-potential or elite players.
    - Clusters with lower averages might indicate developing or specialized roles.
    - Use the summary statistics to explore differences in key attributes (such as stamina, passing, etc.) that characterize each cluster.
    """)
    
    # -------------------------------
    # Section 2: Regression for Overall Rating Prediction
    # -------------------------------
    st.header("Overall Rating Prediction (Model Evaluation)")
    
    # Use scaled features for regression
    X_reg = X_scaled

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y, test_size=0.3, random_state=42
    )

    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write("Random Forest Mean Squared Error:", mse)
    
    # Visualize Actual vs Predicted Overall Ratings
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.subheader("Actual vs Predicted Overall Ratings")
    st.write(results_df.head())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color="navy", alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Overall Ratings")
    st.pyplot(fig)

    # -------------------------------
    # Section 3: Interactive Prediction for a New Player
    # -------------------------------
    st.header("Predict Overall Rating for a New Player")
    st.write("Enter the player's feature values below. The model will predict the overall rating based on the inputs.")

    # Create a form for inputting the player's features (excluding overall_rating)
    with st.form(key="player_form"):
        new_player = {}
        for col in feature_cols:
            default_value = float(df_processed[col].mean())
            new_player[col] = st.number_input(f"{col}", value=default_value)
        submit_button = st.form_submit_button(label="Predict Overall Rating")
    
    # If form is submitted, make a prediction
    if submit_button:
        new_player_df = pd.DataFrame([new_player])
        new_player_scaled = scaler.transform(new_player_df)
        predicted_rating = rf.predict(new_player_scaled)
        st.success(f"Predicted Overall Rating: {predicted_rating[0]:.2f}")

    # -------------------------------
    # Section 4: Generative AI using Variational Autoencoder (VAE)
    # -------------------------------
    st.header("Synthetic Player Profile Generation using VAE")
    
    # Define VAE architecture
    original_dim = X_scaled.shape[1]  
    latent_dim = 2                   
    intermediate_dim = 16            
    
    # Encoder
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation="relu")(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # Decoder (part of the VAE)
    decoder_h = Dense(intermediate_dim, activation="relu")
    decoder_output = Dense(original_dim, activation="sigmoid")
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_output(h_decoded)
    
    # VAE Model
    vae = Model(inputs, x_decoded_mean)
    
    # Define loss: Reconstruction + KL divergence
    reconstruction_loss = tf.keras.losses.mse(inputs, x_decoded_mean)
    reconstruction_loss *= original_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")
    
    st.write("Training the VAE. This may take a moment...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, X_scaled)).batch(16)
    vae.fit(train_dataset, epochs=50, verbose=0)
    st.success("VAE training complete!")
    
    # Build a standalone decoder model for synthetic data generation.
    latent_inputs = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(latent_inputs)
    decoded_output = decoder_output(_h_decoded)
    decoder = Model(latent_inputs, decoded_output)
    
    # Function to generate synthetic player profiles using the decoder
    def generate_synthetic_players(num_samples=5):
        z_samples = np.random.normal(size=(num_samples, latent_dim))
        generated = decoder.predict(z_samples)
        return scaler.inverse_transform(generated)
    
    num_samples = st.number_input("Number of Synthetic Players to Generate", min_value=1, max_value=20, value=5)
    if st.button("Generate Synthetic Profiles"):
        synthetic_players = generate_synthetic_players(num_samples)
        synth_df = pd.DataFrame(synthetic_players, columns=feature_cols)
        st.write("Synthetic Player Profiles:")
        st.dataframe(synth_df)
    
    # -------------------------------
    # Section 5: Overall Player Assessment and Insights
    # -------------------------------
    st.header("Overall Player Assessment")
    st.write("""
    This integrated approach combines multiple techniques:
    
    **Clustering:**  
    - Uses PCA and KMeans to group players with similar performance metrics.  
    - The cluster summary statistics help identify which clusters might represent high potential, balanced, or specialized players.
    
    **Regression:**  
    - A Random Forest model predicts overall ratings based on measurable attributes, supporting objective player comparisons.
    
    **Generative AI (VAE):**  
    - Learns a latent representation of player data and generates synthetic player profiles.  
    - These synthetic profiles allow exploration of hypothetical scenarios and data augmentation.
    
    **Additional Analysis:**  
    - Feature distributions, correlation heatmaps, and visualizations provide initial insights into the data.
    - They help in understanding relationships between variables and guide further analysis.
    """)
    
else:
    st.info("Awaiting CSV file to be uploaded.")
