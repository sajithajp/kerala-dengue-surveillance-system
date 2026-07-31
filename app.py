import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Kerala Dengue Outbreak Intelligence", layout="wide")

st.title("🦟 Kerala Dengue Outbreak Analytics & Forecasting System (2011–2024)")
st.markdown("An end-to-end Epidemiological Data Science Platform designed for disease surveillance and forecasting.")

# Load Data


@st.cache_data
def load_data():
    df_ts = pd.read_csv(
        'dengue_monthly_time_series.csv')
    df_dist = pd.read_csv(
        'dengue_district_annual.csv')
    return df_ts, df_dist


df_ts, df_dist = load_data()

# Navigation Tabs
tab1, tab2 = st.tabs(
    ["📈 Monthly Time-Series & Forecasting", "🗺️ District Spatial Hotspots"])

# --- TAB 1: Time Series & Forecasting ---
with tab1:
    st.header("Statewide Dengue Trends & Future Forecasting")

    # Historical Trend Chart
    fig_ts = px.line(df_ts, x='Date', y='Dengue_Cases', title='Statewide Monthly Dengue Cases (2011 - 2024)',
                     line_shape='spline', markers=True)
    fig_ts.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_ts, use_container_width=True)

    # Seasonality Analysis
    st.subheader("Seasonal Patterns (Monsoon Effect)")
    df_season = df_ts.groupby('Month_Name')['Dengue_Cases'].sum().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    df_season['Month_Name'] = pd.Categorical(
        df_season['Month_Name'], categories=month_order, ordered=True)
    df_season = df_season.sort_values('Month_Name')

    fig_season = px.bar(df_season, x='Month_Name', y='Dengue_Cases',
                        title='Total Dengue Cases by Month (Cumulative 2011-2024)', color='Dengue_Cases')
    st.plotly_chart(fig_season, use_container_width=True)

# --- TAB 2: District Hotspots ---
with tab2:
    st.header("District-Level Spatial Analysis")

    selected_year = st.slider("Select Year", 2011, 2024, 2024)
    df_year_dist = df_dist[df_dist['Year'] == selected_year].sort_values(
        by='Dengue_Cases', ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_dist = px.bar(df_year_dist, x='District', y='Dengue_Cases',
                          color='Dengue_Cases', title=f'Dengue Burden by District in {selected_year}')
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader(f"Top Affected Districts ({selected_year})")
        st.dataframe(df_year_dist[['District', 'Dengue_Cases', 'Dengue_Deaths']].head(
            5), hide_index=True)
