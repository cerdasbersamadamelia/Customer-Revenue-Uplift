"""
Customer Revenue Uplift Simulator - Gradio Interface

Interactive dashboard for exploring customer uplift predictions and campaign simulations.
Built with Gradio for easy deployment and sharing.

Created by Damelia, 2025
"""

# Imports Library
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx

# Load DataFrame
master_features = pd.read_csv('model/result/master_features.csv')
simulation_results = pd.read_csv('model/result/simulation_results.csv')
plan_perf = pd.read_csv('model/result/plan_perf.csv')
geo_perf = pd.read_csv('model/result/geo_perf.csv')
df_influence = pd.read_csv('model/result/df_influence.csv')
feature_importance_shap = pd.read_csv('model/result/feature_importance_shap.csv')

# Load Network Graph
G = joblib.load('model/network_graph.gpickle')

# Load Models
rf_conversion = joblib.load('model/rf_conversion.pkl')
rf_revenue = joblib.load('model/rf_revenue.pkl')
rf_treatment = joblib.load('model/rf_treatment.pkl')
rf_control = joblib.load('model/rf_control.pkl')

# Load LabelEncoders
le_dict_uplift = joblib.load('model/label_encoders_uplift.pkl')
le_dict = joblib.load('model/label_encoders.pkl')


# -----------------------------------------------------------------------------------#
# Tab 1: Ranking & Targeting
def tab1() -> gr.Blocks:
    cols = [
        'customer_id', 'age', 'gender', 'city', 'plan_type', 'uplift_score', 'ARPU',
        'data_user_type', 'social_user_type', 'gaming_user_type', 'network_quality_score',
        'campaign_count', 'campaign_converted', 'campaign_uplift_mean',
        'treatment_prob', 'control_prob', 'campaign_network_interaction', 'campaign_data_interaction',
        'data_usage_gb_rolling7', 'data_arpu_interaction', 'data_usage_gb_sum', 'data_usage_gb_mean',
        'days_since_last_campaign', 'complaints_count'
    ]
    df = master_features.sort_values(by=['uplift_score', 'ARPU'], ascending=[False, False])

    # Format columns for better readability
    df["gender"] = df["gender"].map({"M": "Male", "F": "Female"})
    df["ARPU"] = df["ARPU"].apply(lambda x: f"Rp {int(x):,}")
    df["network_quality_score"] = df["network_quality_score"].apply(lambda x: f"{int(x*100)}%")
    df["campaign_uplift_mean"] = df["campaign_uplift_mean"].apply(lambda x: f"Rp {int(round(x)):,}")
    df["uplift_score"] = df["uplift_score"].apply(lambda x: f"{x*100:.1f}%")
    df["treatment_prob"] = df["treatment_prob"].apply(lambda x: f"{x*100:.1f}%")
    df["control_prob"] = df["control_prob"].apply(lambda x: f"{x*100:.1f}%")
    df["campaign_network_interaction"] = df["campaign_network_interaction"].apply(lambda x: int(round(x)))
    df["campaign_data_interaction"] = df["campaign_data_interaction"].apply(lambda x: int(round(x)))
    df["data_usage_gb_rolling7"] = df["data_usage_gb_rolling7"].apply(lambda x: f"{x:.2f} GB")
    df["data_arpu_interaction"] = df["data_arpu_interaction"].apply(lambda x: int(round(x)))
    df["data_usage_gb_sum"] = df["data_usage_gb_sum"].apply(lambda x: f"{x} GB")
    df["data_usage_gb_mean"] = df["data_usage_gb_mean"].apply(lambda x: f"{x} GB")
    df["days_since_last_campaign"] = df["days_since_last_campaign"].apply(lambda x: f"{int(x)} days")
    df["complaints_count"] = df["complaints_count"].astype(int)

    # Display top 10 customers
    top10 = df[cols].head(10).reset_index(drop=True)
    top10.insert(0, 'No', range(1, len(top10) + 1))
    with gr.Blocks(title="Ranking & Targeting") as demo:
        gr.Markdown("## 🏆 Top 10 Customers for Next Campaign Targeting by Uplift Score & ARPU")
        gr.Dataframe(top10, interactive=False)
    return demo


# -----------------------------------------------------------------------------------#
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
    result = f"""👥 Targeted customers: {n_target}
🤝 Expected conversions: {expected_conversions:,.0f}
📈 Expected revenue: Rp {expected_revenue:,.0f}
💸 Campaign cost: Rp {campaign_cost:,.0f}
📊 Net profit: Rp {net_revenue:,.0f}
💰 ROI: {roi:,.1f}%
🚀 Avg uplift per customer: {avg_uplift*100:.1f}%
📱 Avg ARPU: Rp {avg_arpu:,.0f}"""
    return result

# Tab 2: Campaign Simulation & What-If
def tab2() -> gr.Blocks:
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

    # Take the best scenario for insight
    best_row = df.loc[df["roi"].idxmax()]
    best_pct = int(best_row["target_percentage"])
    n_customers = f"<b>{int(best_row['targeted_customers']):,}</b>"
    roi = f"<b>{int(round(best_row['roi']))}%</b>"
    revenue = f"<b>Rp {best_row['expected_revenue']:,.0f}</b>"
    net_revenue = f"<b>Rp {best_row['net_revenue']:,.0f}</b>"
    expected_conversions = f"<b>{int(round(best_row['expected_conversions'])):,}</b>"
    avg_uplift = f"<b>{best_row['avg_uplift']*100:.1f}%</b>"

    # Insight text
    insight = (
        f"Based on the campaign simulation results, the most optimal strategy is to target the top <b>{best_pct}%</b> of customers ({n_customers} customers). "
        f"With this approach, the expected ROI is {roi}, expected conversions are {expected_conversions}, average uplift is {avg_uplift}, generating an estimated revenue of {revenue} and a net profit of {net_revenue}. "
        "This strategy offers the best balance between the number of targeted customers, potential revenue, and campaign cost efficiency."
    )

    # Graph 1: Line Chart - Net Revenue & ROI vs Target %
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Net Revenue (Rp)', color=color)
    ax1.set_title("Net Revenue & ROI vs Target %")
    line1 = ax1.plot(df['Target %'], df['net_revenue'], color=color, marker='o', label='Net Revenue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    for i, v in enumerate(df['net_revenue']):
        ax1.annotate(f"Rp {int(v):,}", (i, v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('ROI (%)', color=color)
    line2 = ax2.plot(df['Target %'], df['roi'], color=color, marker='x', label='ROI')
    ax2.tick_params(axis='y', labelcolor=color)
    for i, v in enumerate(df['roi']):
        ax2.annotate(f"{int(round(v))}%", (i, v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=color)


    # Highlight max Net Revenue
    max_net_idx = df['net_revenue'].idxmax()
    ax1.plot(max_net_idx, df['net_revenue'][max_net_idx], marker='o', color='red', markersize=10, label='Max Net Revenue')
    ax1.annotate('Max Net Revenue', (max_net_idx, df['net_revenue'][max_net_idx]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, color='red')

    # Highlight max ROI
    max_idx = df['roi'].idxmax()
    ax2.plot(max_idx, df['roi'][max_idx], marker='o', color='red', markersize=10, label='Max ROI')
    ax2.annotate('Max ROI', (max_idx, df['roi'][max_idx]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, color='red')

    fig1.tight_layout()

    # Graph 2: Bar Chart - Expected Revenue, Cost, Net Revenue per Target % + ROI Line + Data Labels + Highlight
    fig2, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Expected Revenue, Cost, Net Revenue per Target %")
    x = np.arange(len(df['Target %']))
    width = 0.25
    bars1 = ax.bar(x - width, df['expected_revenue'], width, label='Expected Revenue', color='#4e79a7')
    bars2 = ax.bar(x, df['campaign_cost'], width, label='Cost', color='#f28e2b')
    bars3 = ax.bar(x + width, df['net_revenue'], width, label='Net Revenue', color='#59a14f')
    
    # Data labels
    for bar in bars1: ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=8)
    for bar in bars2: ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=8)
    for bar in bars3: ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=8)

    ax2b = ax.twinx()
    ax2b.plot(x, df['roi'], color='tab:red', marker='o', label='ROI (%)')
    for i, v in enumerate(df['roi']):
        ax2b.annotate(f"{int(round(v))}%", (x[i], v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='tab:red')

    # Highlight Max Net Revenue
    max_net_idx = df['net_revenue'].idxmax()
    ax.annotate('Max Net Revenue', (x[max_net_idx] + width, df['net_revenue'][max_net_idx]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, color='green')

    # Highlight Max ROI
    max_roi_idx = df['roi'].idxmax()
    ax2b.annotate('Max ROI', (x[max_roi_idx], df['roi'][max_roi_idx]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, color='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['Target %'])
    ax.set_xlabel('Target')
    ax.legend(loc='upper left')
    ax2b.legend(loc='upper right')
    ax.set_ylabel('Amount (Rp)')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax2b.set_ylabel('ROI (%)')
    fig2.tight_layout()

    # Graph 3: Line Chart - Average Uplift vs Target % + Data Labels + Mean Line + Highlight Drop
    fig3, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Average Uplift vs Target %")
    ax.plot(df['Target %'], df['avg_uplift']*100, marker='o', color='#1f77b4', label='Avg Uplift')
    for i, v in enumerate(df['avg_uplift']*100):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='#1f77b4')
    mean_uplift = (df['avg_uplift']*100).mean()
    ax.axhline(mean_uplift, color='gray', linestyle='--', label=f'Mean: {mean_uplift:.1f}%')

    # Highlight biggest drop: merah
    uplift_vals = df['avg_uplift']*100
    diffs = np.diff(uplift_vals)
    if len(diffs) > 0:
        drop_idx = np.argmin(diffs)
        ax.plot(drop_idx+1, uplift_vals.iloc[drop_idx+1], marker='o', color='red', markersize=10)
        ax.annotate('Significant Drop', (drop_idx+1, uplift_vals.iloc[drop_idx+1]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=9, color='red')

    ax.set_xlabel('Target %')
    ax.set_ylabel('Average Uplift (%)')
    ax.legend()
    fig3.tight_layout()

    # Display in Gradio
    with gr.Blocks(title="Campaign Simulation & What-If") as demo:
        gr.Markdown("## 📊 Campaign Simulation Scenarios & Optimal Strategy Insight")
        gr.Dataframe(df_display, interactive=False)
        gr.Markdown(f"**💡 Insight:**\n\n{insight}", elem_id="insight")
        with gr.Row():
            gr.Plot(fig1, label="Net Revenue & ROI vs Target %")
            gr.Plot(fig2, label="Expected Revenue, Cost, Net Revenue per Target %")
            gr.Plot(fig3, label="Average Uplift vs Target %")
        
        gr.Markdown("---")
        gr.Markdown("## 📊 What-If Analysis")
        with gr.Row():
            with gr.Column():
                target_pct = gr.Slider(1, 100, value=5, label="Target % Customer")
                cost_per_customer = gr.Number(value=5000, label="Biaya per Customer")
                revenue_multiplier = gr.Number(value=1.2, label="Revenue Multiplier")
                btn = gr.Button("Run Prediction")
            with gr.Column():
                output = gr.Textbox(label="Hasil Analisis", lines=11)
        btn.click(run_what_if, inputs=[target_pct, cost_per_customer, revenue_multiplier], outputs=output)

    # Conclusion
    return demo


# -----------------------------------------------------------------------------------#
# Tab 3: Network Influence Analysis
def tab3() -> None:
    # Network & Influence Insights
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    network_insight = (
        f"- Total number of nodes in the network: **{num_nodes:,}**\n"
        f"- Total number of connections (edges) between nodes: **{num_edges:,}**\n"
        f"- Network density (proportion of actual to possible connections): **{density:.4f}**\n"
    )

    # Top 10 Influential Customers Table
    top_influencers = df_influence.sort_values('influence_score', ascending=False).head(10).copy()
    cols = [
        'customer_id', 'influence_score', 'network_influence', 'degree_centrality', 'betweenness_centrality',
        'eigenvector_centrality', 'pagerank', 'uplift_score', 'ARPU', 'connections', 'city', 'plan_type', 'age'
    ]
    top_influencers = top_influencers[cols]
    top_influencers.insert(0, 'No', range(1, len(top_influencers) + 1))

    # Format numeric columns for display
    top_influencers['influence_score'] = top_influencers['influence_score'].apply(lambda x: f"{x:.4f}")
    top_influencers['network_influence'] = top_influencers['network_influence'].apply(lambda x: f"{x:.4f}")
    top_influencers['degree_centrality'] = top_influencers['degree_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['betweenness_centrality'] = top_influencers['betweenness_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['eigenvector_centrality'] = top_influencers['eigenvector_centrality'].apply(lambda x: f"{x:.4f}")
    top_influencers['pagerank'] = top_influencers['pagerank'].apply(lambda x: f"{x:.4f}")
    top_influencers['uplift_score'] = top_influencers['uplift_score'].apply(lambda x: f"{x*100:.2f}%")
    top_influencers['ARPU'] = top_influencers['ARPU'].apply(lambda x: f"Rp {x:,.0f}")

    # Get top influencer details for potential further insights
    top_row = top_influencers.iloc[0]
    insight_influencer = (
        f"💡 **Insight**: The most influential customers, such as customer ID **{top_row['customer_id']}** "
        f"(influence score: **{top_row['influence_score']}**), have a strong potential to **spread influence** across the network. "
        f"Prioritizing campaigns to these top influencers can accelerate message reach and **boost revenue**."
    )

    with gr.Blocks(title="Network Influence Analysis") as demo:
        gr.Markdown("## 🌐 Network & Influence Insights")
        gr.Markdown(network_insight)

        gr.Markdown("---")
        gr.Markdown("## 🏆 Top 10 Influential Customers")
        gr.Dataframe(top_influencers, interactive=False)
        gr.Markdown(insight_influencer)
    return demo


# -----------------------------------------------------------------------------------#
# Tab 4: Customer Segmentation with Actionable Insights
def tab4() -> None:
    # Format plan_perf and geo_perf for display (do not overwrite original)
    plan_perf_fmt = plan_perf.copy()
    geo_perf_fmt = geo_perf.copy()
    plan_perf_fmt['Avg Uplift'] = plan_perf_fmt['Avg Uplift'].astype(float)
    plan_perf_fmt['Avg ARPU'] = plan_perf_fmt['Avg ARPU'].astype(float)
    geo_perf_fmt['Avg Uplift'] = geo_perf_fmt['Avg Uplift'].astype(float)
    geo_perf_fmt['Avg ARPU'] = geo_perf_fmt['Avg ARPU'].astype(float)

    # Generate actionable insight for plan type
    top_plan = plan_perf_fmt.sort_values('Avg Uplift', ascending=False).iloc[0]
    avg_uplift_plan = plan_perf_fmt['Avg Uplift'].mean()
    uplift_diff_plan = (top_plan['Avg Uplift'] - avg_uplift_plan) / avg_uplift_plan * 100 if avg_uplift_plan != 0 else 0
    pot_revenue_plan = int(top_plan['Customers'] * top_plan['Avg ARPU'] * top_plan['Avg Uplift'])
    insight_plan = (
        f"💡 <b>Insight:</b> The plan type <b>{top_plan['plan_type']}</b> shows the highest uplift at <b>{top_plan['Avg Uplift']*100:.2f}%</b> "
        f"<i>(<b>{int(top_plan['Customers'])}</b> customers, ARPU <b>Rp {top_plan['Avg ARPU']:,.0f}</b>)</i>, "
        f"<b>{uplift_diff_plan:.1f}%</b> above the average of other plan types. "
        f"If all <b>{top_plan['plan_type']}</b> customers are targeted, the potential additional revenue is <b>Rp {pot_revenue_plan:,.0f}</b>. "
        f"<u>Recommendation:</u> Launch a dedicated campaign for this segment to maximize business impact."
    )

    # Format for display
    plan_perf_fmt['Avg Uplift'] = plan_perf_fmt['Avg Uplift'].apply(lambda x: f"{x*100:.2f}%")
    plan_perf_fmt['Avg ARPU'] = plan_perf_fmt['Avg ARPU'].apply(lambda x: f"Rp {x:,.0f}")

    # Generate actionable insight for city
    top_city = geo_perf_fmt.sort_values('Avg Uplift', ascending=False).iloc[0]
    avg_uplift_city = geo_perf_fmt['Avg Uplift'].mean()
    uplift_diff_city = (top_city['Avg Uplift'] - avg_uplift_city) / avg_uplift_city * 100 if avg_uplift_city != 0 else 0
    pot_revenue_city = int(top_city['Customers'] * top_city['Avg ARPU'] * top_city['Avg Uplift'])
    insight_geo = (
        f"💡 <b>Insight:</b> The city <b>{top_city['city']}</b> stands out with an uplift of <b>{top_city['Avg Uplift']*100:.2f}%</b> "
        f"<i>(<b>{int(top_city['Customers'])}</b> customers, ARPU <b>Rp {top_city['Avg ARPU']:,.0f}</b>)</i>, "
        f"<b>{uplift_diff_city:.1f}%</b> above the average of other cities. "
        f"<u>Recommendation:</u> Focus marketing budget and activities in <b>{top_city['city']}</b> to further increase conversion and revenue."
    )

    geo_perf_fmt['Avg Uplift'] = geo_perf_fmt['Avg Uplift'].apply(lambda x: f"{x*100:.2f}%")
    geo_perf_fmt['Avg ARPU'] = geo_perf_fmt['Avg ARPU'].apply(lambda x: f"Rp {x:,.0f}")

    with gr.Blocks(title="Customer Segmentation") as demo:
        gr.Markdown("## 📱 Plan Type Performance")
        gr.Dataframe(plan_perf_fmt, interactive=False)
        gr.Markdown(insight_plan)

        gr.Markdown("---")
        gr.Markdown("## 🗺️ Geographic Performance")
        gr.Dataframe(geo_perf_fmt, interactive=False)
        gr.Markdown(insight_geo)
    return demo


# -----------------------------------------------------------------------------------#
# Tab 5: Model Explainability
def tab5():
    # Top 10 Feature Importance Table by SHAP
    top10 = feature_importance_shap.head(10).copy()
    top10['No'] = range(1, len(top10) + 1)
    top10['SHAP Importance'] = top10['shap_importance'].apply(lambda x: f"{x*100:.2f}%")
    top10_table = top10[['No', 'fitur', 'SHAP Importance']]
    top10_table.columns = ['No', 'Feature', 'SHAP Importance']

    with gr.Blocks() as tab:
        gr.Markdown("## 🏆 Top 10 Features By SHAP Importance")
        gr.Dataframe(top10_table, interactive=False)
        gr.Markdown(f"💡 **Insight**: These features have the highest impact on conversion predictions according to SHAP analysis.")

    return tab


# -----------------------------------------------------------------------------------#
# Main Dashboard with Tabs
def customer_revenue_uplift_dashboard():
    with gr.Blocks(title="Customer Revenue Uplift Dashboard") as demo:
        gr.Markdown("# 🚀 Customer Revenue Uplift Dashboard")
        with gr.Tabs():
            with gr.TabItem("Ranking & Targeting"):
                tab1()
            with gr.TabItem("Campaign Simulation & What-If"):
                tab2()
            with gr.TabItem("Network Influence Analysis"):
                tab3()
            with gr.TabItem("Customer Segmentation"):
                tab4()
            with gr.TabItem("Model Explainability"):
                tab5()
    return demo

# Launch the dashboard
dashboard = customer_revenue_uplift_dashboard()
dashboard.launch()