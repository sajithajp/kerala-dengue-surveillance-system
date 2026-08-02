import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Kerala Dengue Outbreak Intelligence", layout="wide")

st.title("🦟 Kerala Dengue Outbreak Analytics & Survival Analysis Platform (2011–2024)")
st.markdown("An end-to-end Epidemiological Data Science Platform combining Time-Series Forecasting, Spatial Risk Mapping, and Biostatistical Survival Analysis.")

# Load Data


@st.cache_data
def load_data():
    df_ts = pd.read_csv('dengue_monthly_time_series.csv')
    df_dist = pd.read_csv('dengue_district_annual.csv')

    # Feature Engineering for Survival Analysis & CFR
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts['CFR_%'] = np.where(
        df_ts['Dengue_Cases'] > 0, (df_ts['Dengue_Deaths'] / df_ts['Dengue_Cases']) * 100, 0)
    df_ts['Monthly_Survival_Rate'] = 1 - \
        (df_ts['Dengue_Deaths'] / df_ts['Dengue_Cases'].replace(0, 1))
    df_ts['Cumulative_Survival'] = df_ts['Monthly_Survival_Rate'].cumprod()

    return df_ts, df_dist


df_ts, df_dist = load_data()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Monthly Time-Series & Forecasting",
    "🗺️ District Spatial Hotspots",
    "🩺 Survival Analysis & Case Fatality (CFR)"
])

# ==========================================
# TAB 1: Time Series & Forecasting
# ==========================================
with tab1:
    st.header("Statewide Dengue Trends & Outbreak Patterns")
    fig_ts = px.line(df_ts, x='Date', y='Dengue_Cases',
                     title='Statewide Monthly Dengue Cases (2011 - 2024)',
                     markers=True)
    fig_ts.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================
# TAB 2: District Hotspots
# ==========================================
with tab2:
    st.header("District-Level Spatial Risk Distribution")
    selected_year = st.slider("Select Year", 2011, 2024, 2024)
    df_year_dist = df_dist[df_dist['Year'] == selected_year].sort_values(
        by='Dengue_Cases', ascending=False)

    fig_dist = px.bar(df_year_dist, x='District', y='Dengue_Cases',
                      color='Dengue_Cases', title=f'Dengue Burden by District in {selected_year}')
    st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 3: Biostatistical Survival Analysis
# ==========================================
with tab3:
    st.header("🩺 Epidemiological Survival Analysis & Fatality Modeling")
    st.markdown("""
    This module evaluates the **Case Fatality Rate (CFR)** and estimates **Cohort Survival Rates** over time 
    using empirical mortality and incidence data across Kerala (2011–2024).
    """)

    # Key Summary Metrics
    total_cases = df_ts['Dengue_Cases'].sum()
    total_deaths = df_ts['Dengue_Deaths'].sum()
    overall_cfr = (total_deaths / total_cases) * 100
    overall_survival = 100 - overall_cfr

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Confirmed Cases", f"{total_cases:,}")
    m2.metric("Total Dengue Deaths", f"{total_deaths:,}")
    m3.metric("Overall Case Fatality Rate (CFR)", f"{overall_cfr:.2f}%")
    m4.metric("Overall Patient Survival Rate", f"{overall_survival:.2f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Chart 1: Case Fatality Rate (CFR %) Over Time
    with col_a:
        st.subheader("1. Monthly Case Fatality Rate (CFR %) Trend")
        fig_cfr = px.line(df_ts, x='Date', y='CFR_%',
                          title="Monthly Dengue Case Fatality Rate (CFR %)",
                          labels={'CFR_%': 'CFR (%)', 'Date': 'Year'},
                          markers=True)
        fig_cfr.update_traces(line_color='#E74C3C')
        st.plotly_chart(fig_cfr, use_container_width=True)
        st.caption(
            "Spikes in CFR highlight critical periods where clinical disease severity or healthcare strain was highest.")

    # Chart 2: Cumulative Cohort Survival Function S(t)
    with col_b:
        st.subheader("2. Cumulative Longitudinal Survival Function S(t)")
        fig_surv = px.line(df_ts, x='Date', y='Cumulative_Survival',
                           title="Product-Limit Cumulative Cohort Survival Curve S(t)",
                           labels={'Cumulative_Survival': 'Cumulative Survival Probability S(t)', 'Date': 'Timeline'})
        fig_surv.update_traces(line_color='#2ECC71')
        st.plotly_chart(fig_surv, use_container_width=True)
        st.caption(
            "Cumulative survival estimation reflecting long-term cohort progression across 168 months.")

    st.markdown("---")

    # Annual Survival Analysis Table
    st.subheader("3. Annual Survival & Fatality Breakdown (2011–2024)")

    yearly_surv = df_ts.groupby('Year').agg({
        'Dengue_Cases': 'sum',
        'Dengue_Deaths': 'sum'
    }).reset_index()

    yearly_surv['CFR (%)'] = (yearly_surv['Dengue_Deaths'] /
                              yearly_surv['Dengue_Cases'] * 100).round(3)
    yearly_surv['Annual Survival Rate (%)'] = (
        100 - yearly_surv['CFR (%)']).round(3)
    yearly_surv['Hazard / Fatality Ratio per 10k'] = (
        (yearly_surv['Dengue_Deaths'] / yearly_surv['Dengue_Cases']) * 10000).round(2)

    # Display styled dataframe
    st.dataframe(
        yearly_surv.style.background_gradient(subset=['CFR (%)'], cmap='Reds')
                         .background_gradient(subset=['Annual Survival Rate (%)'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )
