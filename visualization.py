import streamlit as st
import pandas as pd
import plotly.express as px

def visualize_data(data):
    df = pd.DataFrame(data)
    st.write("Data Overview")
    st.write(df)

    # Circular graph for claim accuracy
    st.subheader("Claim Classification Breakdown")
    classification_counts = df["classification"].value_counts()
    fig = px.pie(values=classification_counts, names=classification_counts.index, title="Classification Breakdown")
    st.plotly_chart(fig)

    # Heatmap (Placeholder, no real geolocation data)
    st.subheader("Misinformation Heatmap")
    df['latitude'] = [20] * len(df)
    df['longitude'] = [78] * len(df)
    fig = px.density_mapbox(df, lat="latitude", lon="longitude", z="classification",
                            radius=10, center={"lat": 20, "lon": 78},
                            mapbox_style="open-street-map")
    st.plotly_chart(fig)
