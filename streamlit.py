"""
Customer Revenue Uplift Simulator - Streamlit Interface

Interactive dashboard for exploring customer uplift predictions and campaign simulations.
Built with Streamlit for easy deployment and sharing.

Created by Damelia, 2025
"""

# Imports Library
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Customer Revenue Uplift Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load DataFrame
@st.cache_data
def load_data():
    master_features = pd.read_csv('model/result/master_features.csv')
    simulation_results = pd.read_csv('model/result/simulation_results.csv')
    plan_perf = pd.read_csv('model/result/plan_perf.csv')
    geo_perf = pd.read_csv('model/result/geo_perf.csv')
    df_influence = pd.read_csv('model/result/df_influence.csv')
    feature_importance_shap = pd.read_csv('model/result/feature_importance_shap.csv')
    return master_features, simulation_results, plan_perf, geo_perf, df_influence, feature_importance_shap

# Load Models
@st.cache_resource
def load_models():
    G = joblib.load('model/network_graph.gpickle')
    rf_conversion = joblib.load('model/rf_conversion.pkl')
    rf_revenue = joblib.load('model/rf_revenue.pkl')
    rf_treatment = joblib.load('model/rf_treatment.pkl')
    rf_control = joblib.load('model/rf_control.pkl')
    le_dict_uplift = joblib.load('model/label_encoders_uplift.pkl')
    le_dict = joblib.load('model/label_encoders.pkl')
    return G, rf_conversion, rf_revenue, rf_treatment, rf_control, le_dict_uplift, le_dict

# Load data and models
master_features, simulation_results, plan_perf, geo_perf, df_influence, feature_importance_shap = load_data()
G, rf_conversion, rf_revenue, rf_treatment, rf_control, le_dict_uplift, le_dict = load_models()

# Function to run what-if analysis
def run_what_if(target_pct, cost_per_customer, revenue_multiplier):
    # Simulate campaign
    ranked_customers = master_features.sort_values('uplift_score', ascending=False).reset_index(drop=True)
    n_target = int(len(ranked_customers) * target_pct / 100)
    targeted_customers = ranked_customers.head(n_target)
    expected_conversions = targeted_customers['treatment_prob'].sum()
    avg_arpu = targeted_customers['ARPU'].mean() if n_target > 0 else 0
    expected_revenue = expected_conversions * avg_arpu * revenue_multiplier
    campaign_cost = n_target * cost_per_customer
    net_revenue = expected_revenue - campaign_cost
    roi = (net_revenue / campaign_cost) * 100 if campaign_cost > 0 else 0
    avg_uplift = targeted_customers['uplift_score'].mean() if n_target > 0 else 0
    
    return {
        'targeted_customers': n_target,
        'expected_conversions': expected_conversions,
        'expected_revenue': expected_revenue,
        'campaign_cost': campaign_cost,
        'net_revenue': net_revenue,
        'roi': roi,
        'avg_uplift': avg_uplift,
        'avg_arpu': avg_arpu
    }

# Main Dashboard
def main():
    st.title("🚀 Customer Revenue Uplift Dashboard")

    # --- Section 1: Ranking & Targeting ---
    st.header("🏆 Top 10 Customers for Next Campaign Targeting by Uplift Score & ARPU")
    cols = [
        'customer_id', 'age', 'gender', 'city', 'plan_type', 'uplift_score', 'ARPU',
        'data_user_type', 'social_user_type', 'gaming_user_type', 'network_quality_score',
        'campaign_count', 'campaign_converted', 'campaign_uplift_mean',
        'treatment_prob', 'control_prob', 'campaign_network_interaction', 'campaign_data_interaction',
        'data_usage_gb_rolling7', 'data_arpu_interaction', 'data_usage_gb_sum', 'data_usage_gb_mean',
        'days_since_last_campaign', 'complaints_count'
    ]
    df = master_features.sort_values(by=['uplift_score', 'ARPU'], ascending=[False, False])
    df_display = df.copy()
    df_display["gender"] = df_display["gender"].map({"M": "Male", "F": "Female"})
    df_display["ARPU"] = df_display["ARPU"].apply(lambda x: f"Rp {int(x):,}")
    df_display["network_quality_score"] = df_display["network_quality_score"].apply(lambda x: f"{int(x*100)}%")
    df_display["campaign_uplift_mean"] = df_display["campaign_uplift_mean"].apply(lambda x: f"Rp {int(round(x)):,}")
    df_display["uplift_score"] = df_display["uplift_score"].apply(lambda x: f"{x*100:.1f}%")
    df_display["treatment_prob"] = df_display["treatment_prob"].apply(lambda x: f"{x*100:.1f}%")
    df_display["control_prob"] = df_display["control_prob"].apply(lambda x: f"{x*100:.1f}%")
    df_display["campaign_network_interaction"] = df_display["campaign_network_interaction"].apply(lambda x: int(round(x)))
    df_display["campaign_data_interaction"] = df_display["campaign_data_interaction"].apply(lambda x: int(round(x)))
    df_display["data_usage_gb_rolling7"] = df_display["data_usage_gb_rolling7"].apply(lambda x: f"{x:.2f} GB")
    df_display["data_arpu_interaction"] = df_display["data_arpu_interaction"].apply(lambda x: int(round(x)))
    df_display["data_usage_gb_sum"] = df_display["data_usage_gb_sum"].apply(lambda x: f"{x} GB")
    df_display["data_usage_gb_mean"] = df_display["data_usage_gb_mean"].apply(lambda x: f"{x} GB")
    df_display["days_since_last_campaign"] = df_display["days_since_last_campaign"].apply(lambda x: f"{int(x)} days")
    df_display["complaints_count"] = df_display["complaints_count"].astype(int)
    top10 = df_display[cols].head(10).reset_index(drop=True)
    st.dataframe(top10, use_container_width=True)

    # --- Section 2: Campaign Simulation & What-If ---
    st.header("📊 Campaign Simulation Scenarios & Optimal Strategy Insight")
    # Format simulation results for display
    df = pd.DataFrame(simulation_results) if not isinstance(simulation_results, pd.DataFrame) else simulation_results.copy()
    # Add formatted columns
    df["Target %"] = df["target_percentage"].apply(lambda x: f"{int(x)}%")
    cols = ["Target %"] + [col for col in df.columns if col not in ["Target %", "target_percentage"]]
    df_display = df[cols].copy()
    df_display["targeted_customers"] = df_display["targeted_customers"].apply(lambda x: f"{int(x):,}")
    df_display["expected_conversions"] = df_display["expected_conversions"].apply(lambda x: f"{int(round(x)):,}")
    df_display["expected_revenue"] = df_display["expected_revenue"].apply(lambda x: f"Rp {x:,.0f}")
    df_display["campaign_cost"] = df_display["campaign_cost"].apply(lambda x: f"Rp {x:,.0f}")
    df_display["net_revenue"] = df_display["net_revenue"].apply(lambda x: f"Rp {x:,.0f}")
    df_display["roi"] = df_display["roi"].apply(lambda x: f"{int(round(x))}%")
    df_display["avg_uplift"] = df_display["avg_uplift"].apply(lambda x: f"{x*100:.1f}%")
    df_display["avg_arpu"] = df_display["avg_arpu"].apply(lambda x: f"Rp {x:,.0f}")

    st.dataframe(df_display, use_container_width=True)

    # Take the best scenario for insight
    best_row = df.loc[df["roi"].idxmax()]
    best_pct = int(best_row["target_percentage"])
    n_customers = f"**{int(best_row['targeted_customers']):,}**"
    roi = f"**{int(round(best_row['roi']))}%**"
    revenue = f"**Rp {best_row['expected_revenue']:,.0f}**"
    net_revenue = f"**Rp {best_row['net_revenue']:,.0f}**"
    expected_conversions = f"**{int(round(best_row['expected_conversions'])):,}**"
    avg_uplift = f"**{best_row['avg_uplift']*100:.1f}%**"

    # Insight text
    insight = (
        f"Based on the campaign simulation results, the most optimal strategy is to target the top **{best_pct}%** of customers ({n_customers} customers). "
        f"With this approach, the expected ROI is {roi}, expected conversions are {expected_conversions}, average uplift is {avg_uplift}, generating an estimated revenue of {revenue} and a net profit of {net_revenue}. "
        "This strategy offers the best balance between the number of targeted customers, potential revenue, and campaign cost efficiency."
    )

    st.markdown(f"**💡 Insight:**\n\n{insight}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Graph 1: Net Revenue & ROI vs Target %
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(
            go.Scatter(x=df['Target %'], y=df['net_revenue'], name='Net Revenue', mode='lines+markers'),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(x=df['Target %'], y=df['roi'], name='ROI (%)', mode='lines+markers'),
            secondary_y=True,
        )
        fig1.update_xaxes(title_text="Target %")
        fig1.update_yaxes(title_text="Net Revenue (Rp)", secondary_y=False)
        fig1.update_yaxes(title_text="ROI (%)", secondary_y=True)
        fig1.update_layout(title_text="Net Revenue & ROI vs Target %")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Graph 2: Average Uplift vs Target %
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df['Target %'], 
            y=df['avg_uplift']*100, 
            mode='lines+markers+text',
            name='Avg Uplift',
            text=[f"{v:.1f}%" for v in df['avg_uplift']*100],
            textposition="top center"
        ))
        mean_uplift = (df['avg_uplift']*100).mean()
        fig3.add_hline(y=mean_uplift, line_dash="dash", annotation_text=f"Mean: {mean_uplift:.1f}%")
        fig3.update_layout(
            title="Average Uplift vs Target %",
            xaxis_title="Target %",
            yaxis_title="Average Uplift (%)"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Bar Chart for Revenue/Cost/Net Revenue
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=df['Target %'],
            y=df['expected_revenue'],
            name='Expected Revenue',
            text=[f"{int(v):,}" for v in df['expected_revenue']],
            textposition='auto'
        ))
        
        fig2.add_trace(go.Bar(
            x=df['Target %'],
            y=df['campaign_cost'],
            name='Cost',
            text=[f"{int(v):,}" for v in df['campaign_cost']],
            textposition='auto'
        ))
        
        fig2.add_trace(go.Bar(
            x=df['Target %'],
            y=df['net_revenue'],
            name='Net Revenue',
            text=[f"{int(v):,}" for v in df['net_revenue']],
            textposition='auto'
        ))
        
        fig2.update_layout(
            title="Expected Revenue, Cost, Net Revenue per Target %",
            xaxis_title="Target %",
            yaxis_title="Amount (Rp)",
            barmode='group'
        )
        
        st.plotly_chart(fig2, use_container_width=True)

        # What-If Analysis
        st.markdown("---")
        st.header("📊 What-If Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_pct = st.slider("Target % Customer", 1, 100, 5)
            cost_per_customer = st.number_input("Biaya per Customer", value=5000)
            revenue_multiplier = st.number_input("Revenue Multiplier", value=1.2, step=0.1)
            
            if st.button("Run Prediction"):
                result = run_what_if(target_pct, cost_per_customer, revenue_multiplier)
                
                with col2:
                    st.subheader("Hasil Analisis")
                    st.write(f"👥 Targeted customers: {result['targeted_customers']:,}")
                    st.write(f"🤝 Expected conversions: {result['expected_conversions']:,.0f}")
                    st.write(f"📈 Expected revenue: Rp {result['expected_revenue']:,.0f}")
                    st.write(f"💸 Campaign cost: Rp {result['campaign_cost']:,.0f}")
                    st.write(f"📊 Net profit: Rp {result['net_revenue']:,.0f}")
                    st.write(f"💰 ROI: {result['roi']:,.1f}%")
                    st.write(f"🚀 Avg uplift per customer: {result['avg_uplift']*100:.1f}%")
                    st.write(f"📱 Avg ARPU: Rp {result['avg_arpu']:,.0f}")


    # --- Section 3: Network Influence Analysis ---
    st.header("🌐 Network & Influence Insights")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", f"{num_nodes:,}")
    with col2:
        st.metric("Total Connections", f"{num_edges:,}")
    with col3:
        st.metric("Network Density", f"{density:.4f}")
    st.markdown("---")
    st.header("🏆 Top 10 Influential Customers")
    cols_inf = [
        'customer_id', 'influence_score', 'network_influence', 'degree_centrality', 'betweenness_centrality',
        'eigenvector_centrality', 'pagerank', 'uplift_score', 'ARPU', 'connections', 'city', 'plan_type', 'age'
    ]
    top_influencers = df_influence.sort_values('influence_score', ascending=False).head(10).copy()
    top_influencers = top_influencers[cols_inf]
    top_influencers['influence_score'] = top_influencers['influence_score'].apply(lambda x: f"{x:.4f}")
    top_influencers['network_influence'] = top_influencers['network_influence'].apply(lambda x: f"{x:.4f}")
    top_influencers['degree_centrality'] = top_influencers['degree_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['betweenness_centrality'] = top_influencers['betweenness_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['eigenvector_centrality'] = top_influencers['eigenvector_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['pagerank'] = top_influencers['pagerank'].apply(lambda x: f"{x:.4f}")
    top_influencers['uplift_score'] = top_influencers['uplift_score'].apply(lambda x: f"{x*100:.2f}%")
    top_influencers['ARPU'] = top_influencers['ARPU'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(top_influencers, use_container_width=True)
    top_row = top_influencers.iloc[0]
    insight_influencer = (
        f"💡 **Insight**: The most influential customers, such as customer ID **{top_row['customer_id']}** "
        f"(influence score: **{top_row['influence_score']}**), have a strong potential to **spread influence** across the network. "
        f"Prioritizing campaigns to these top influencers can accelerate message reach and **boost revenue**."
    )
    st.markdown(insight_influencer)


    # --- Section 4: Customer Segmentation ---
    st.header("📱 Plan Type Performance")
    plan_perf_fmt = plan_perf.copy()
    plan_perf_fmt['Avg Uplift'] = plan_perf_fmt['Avg Uplift'].astype(float)
    plan_perf_fmt['Avg ARPU'] = plan_perf_fmt['Avg ARPU'].astype(float)
    top_plan = plan_perf_fmt.sort_values('Avg Uplift', ascending=False).iloc[0]
    avg_uplift_plan = plan_perf_fmt['Avg Uplift'].mean()
    uplift_diff_plan = (top_plan['Avg Uplift'] - avg_uplift_plan) / avg_uplift_plan * 100 if avg_uplift_plan != 0 else 0
    pot_revenue_plan = int(top_plan['Customers'] * top_plan['Avg ARPU'] * top_plan['Avg Uplift'])
    insight_plan = (
        f"💡 **Insight**: The plan type **{top_plan['plan_type']}** shows the highest uplift at **{top_plan['Avg Uplift']*100:.2f}%** "
        f"(**{int(top_plan['Customers'])}** customers, ARPU **Rp {top_plan['Avg ARPU']:,.0f}**), "
        f"**{uplift_diff_plan:.1f}%** above the average of other plan types. "
        f"If all **{top_plan['plan_type']}** customers are targeted, the potential additional revenue is **Rp {pot_revenue_plan:,.0f}**. "
        f"**Recommendation**: Launch a dedicated campaign for this segment to maximize business impact."
    )
    plan_perf_display = plan_perf_fmt.copy()
    plan_perf_display['Avg Uplift'] = plan_perf_display['Avg Uplift'].apply(lambda x: f"{x*100:.2f}%")
    plan_perf_display['Avg ARPU'] = plan_perf_display['Avg ARPU'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(plan_perf_display, use_container_width=True)
    st.markdown(insight_plan)
    st.markdown("---")
    st.header("🗺️ Geographic Performance")
    geo_perf_fmt = geo_perf.copy()
    geo_perf_fmt['Avg Uplift'] = geo_perf_fmt['Avg Uplift'].astype(float)
    geo_perf_fmt['Avg ARPU'] = geo_perf_fmt['Avg ARPU'].astype(float)
    top_city = geo_perf_fmt.sort_values('Avg Uplift', ascending=False).iloc[0]
    avg_uplift_city = geo_perf_fmt['Avg Uplift'].mean()
    uplift_diff_city = (top_city['Avg Uplift'] - avg_uplift_city) / avg_uplift_city * 100 if avg_uplift_city != 0 else 0
    pot_revenue_city = int(top_city['Customers'] * top_city['Avg ARPU'] * top_city['Avg Uplift'])
    insight_geo = (
        f"💡 **Insight**: The city **{top_city['city']}** stands out with an uplift of **{top_city['Avg Uplift']*100:.2f}%** "
        f"(**{int(top_city['Customers'])}** customers, ARPU **Rp {top_city['Avg ARPU']:,.0f}**), "
        f"**{uplift_diff_city:.1f}%** above the average of other cities. "
        f"**Recommendation**: Focus marketing budget and activities in **{top_city['city']}** to further increase conversion and revenue."
    )
    geo_perf_display = geo_perf_fmt.copy()
    geo_perf_display['Avg Uplift'] = geo_perf_display['Avg Uplift'].apply(lambda x: f"{x*100:.2f}%")
    geo_perf_display['Avg ARPU'] = geo_perf_display['Avg ARPU'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(geo_perf_display, use_container_width=True)
    st.markdown(insight_geo)


    # --- Section 4: Model Explainability ---
    st.header("🏆 Top 10 Features By SHAP Importance")
    top10 = feature_importance_shap.head(10).copy()
    top10['SHAP Importance'] = top10['shap_importance'].apply(lambda x: f"{x*100:.2f}%")
    top10_table = top10[['fitur', 'SHAP Importance']]
    top10_table.columns = ['Feature', 'SHAP Importance']
    st.dataframe(top10_table, use_container_width=True)
    st.markdown("💡 **Insight**: These features have the highest impact on conversion predictions according to SHAP analysis.")
    fig = px.bar(
        top10.head(10), 
        x='shap_importance', 
        y='fitur',
        orientation='h',
        title='Top 10 SHAP Feature Importance',
        labels={'shap_importance': 'SHAP Importance', 'fitur': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()