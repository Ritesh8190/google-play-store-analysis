import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Google Play Store Analytics Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling, clean metric cards, and responsive charts
st.markdown("""
<style>
    /* Global App Background & Font Settings */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header Card Styling */
    .header-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.15);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0px;
        opacity: 0.92;
        color: #e2e8f0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.07);
    }
    .metric-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 3px;
    }

    /* Section Header */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 20px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA LOADING & CLEANING (Exact logic from your Colab Notebook)
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    """
    Loads Google Play Store dataset and applies the exact preprocessing steps
    established in the Jupyter Notebook:
    1. df.drop_duplicates()
    2. df['Rating'].fillna(df['Rating'].median())
    3. df['Reviews'].fillna(0)
    4. df.dropna()
    5. Clean 'Installs' (remove commas & plus signs, convert to numeric)
    6. Clean 'Price' (remove dollar sign, convert to numeric)
    7. Create 'App_Type' column (Free if Price == 0, else Paid)
    8. Cast 'Reviews' to numeric and cap outliers using IQR method
    """
    # Robust dataset resolution: looks in script directory first (ideal for Linux/Render)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "googleplaystore.csv"),
        "googleplaystore.csv",
        os.path.join(base_dir, "data", "googleplaystore.csv"),
        os.path.join("data", "googleplaystore.csv"),
        os.path.join("..", "googleplaystore.csv"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        return None

    # Load dataset
    df = pd.read_csv(file_path)

    # 1. Remove duplicate records (Notebook Cell 3)
    df = df.drop_duplicates()

    # 2. Fill missing values in Rating with median (Notebook Cell 3)
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())

    # 3. Fill missing values in Reviews (Notebook Cell 3)
    df['Reviews'] = df['Reviews'].fillna(0)

    # 4. Remove rows with any remaining missing values (Notebook Cell 3)
    df = df.dropna()

    # 5. Convert Installs and Price into numbers (Notebook Cell 4)
    df['Installs'] = pd.to_numeric(
        df['Installs'].astype(str).str.replace(r'[,+]', '', regex=True),
        errors='coerce'
    ).fillna(0)

    df['Price'] = pd.to_numeric(
        df['Price'].astype(str).str.replace('$', '', regex=False),
        errors='coerce'
    ).fillna(0)

    # 6. Identify Free and Paid apps (Notebook Cell 4)
    df['App_Type'] = np.where(df['Price'] == 0, 'Free', 'Paid')

    # 7. Convert Reviews to numeric integer (Notebook Cell 5)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0).astype(int)

    # 8. Handle outliers in Reviews using IQR capping (Notebook Cell 5)
    q1 = df['Reviews'].quantile(0.25)
    q3 = df['Reviews'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['Reviews_Capped'] = df['Reviews'].clip(lower_bound, upper_bound)

    return df


raw_df = load_and_clean_data()

# Handle case where file is missing
if raw_df is None:
    st.error("⚠️ `googleplaystore.csv` not found in the project folder!")
    st.info("Please ensure `googleplaystore.csv` is placed in the same folder as `app.py`.")
    st.stop()


# ==============================================================================
# 3. SIDEBAR FILTERS
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Play_Arrow_logo.svg", width=55)
st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("Filter and slice data interactively:")

# Filter 1: App Category Multi-Select
all_categories = sorted(raw_df['Category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "📂 Select Category",
    options=all_categories,
    default=[],
    help="Select one or multiple app categories. Leave empty to include all categories."
)

# Filter 2: App Type (Free / Paid / All)
type_options = ["All", "Free", "Paid"]
selected_type = st.sidebar.radio(
    "💳 App Type",
    options=type_options,
    horizontal=True,
    help="Filter apps by monetization model (Free or Paid)."
)

# Filter 3: Rating Slider
min_rating, max_rating = float(raw_df['Rating'].min()), float(raw_df['Rating'].max())
selected_rating = st.sidebar.slider(
    "⭐ Minimum Rating Range",
    min_value=1.0,
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.1,
    help="Select the minimum and maximum app ratings to display."
)

# Filter 4: Content Rating
all_content_ratings = sorted(raw_df['Content Rating'].unique().tolist())
selected_content = st.sidebar.multiselect(
    "🔞 Content Rating",
    options=all_content_ratings,
    default=[],
    help="Filter by target age demographic (Everyone, Teen, Mature 17+, etc.)."
)

# Filter 5: Search App by Keyword
search_query = st.sidebar.text_input(
    "🔍 Search App Name",
    placeholder="e.g. WhatsApp, Subway Surfers...",
    help="Search apps matching text in their name."
)

# Reset filters button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Student Project:** Google Play Store Analysis  \n"
    "**Course:** Data Processing & Analysis  \n"
    "**Libraries:** Streamlit, Pandas, Plotly"
)


# ==============================================================================
# 4. FILTERING LOGIC
# ==============================================================================
filtered_df = raw_df.copy()

# Apply Category filter
if selected_categories:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]

# Apply App Type filter
if selected_type != "All":
    filtered_df = filtered_df[filtered_df['App_Type'] == selected_type]

# Apply Rating filter
filtered_df = filtered_df[
    (filtered_df['Rating'] >= selected_rating[0]) & 
    (filtered_df['Rating'] <= selected_rating[1])
]

# Apply Content Rating filter
if selected_content:
    filtered_df = filtered_df[filtered_df['Content Rating'].isin(selected_content)]

# Apply Search query
if search_query:
    filtered_df = filtered_df[filtered_df['App'].str.contains(search_query, case=False, na=False)]


# ==============================================================================
# 5. HEADER COMPONENT
# ==============================================================================
st.markdown("""
<div class="header-box">
    <h1 class="header-title">📱 Google Play Store Data Analytics Dashboard</h1>
    <p class="header-subtitle">
        An interactive exploratory dashboard analyzing app categories, ratings, downloads, pricing models, and user reviews.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. KPI METRICS CARDS
# ==============================================================================
total_apps = len(filtered_df)
avg_rating = filtered_df['Rating'].mean() if total_apps > 0 else 0.0
total_installs = filtered_df['Installs'].sum() if total_apps > 0 else 0
free_apps_count = (filtered_df['App_Type'] == 'Free').sum() if total_apps > 0 else 0
paid_apps_count = (filtered_df['App_Type'] == 'Paid').sum() if total_apps > 0 else 0
free_pct = (free_apps_count / total_apps * 100) if total_apps > 0 else 0.0

# Format installs in compact readable form (Billions/Millions)
def format_number(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return f"{num:,.0f}"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Apps</div>
        <div class="metric-value">{total_apps:,}</div>
        <div class="metric-sub">Filtered / {len(raw_df):,} total</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Rating</div>
        <div class="metric-value">⭐ {avg_rating:.2f}</div>
        <div class="metric-sub">Scale: 1.0 - 5.0 stars</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Installs</div>
        <div class="metric-value">{format_number(total_installs)}</div>
        <div class="metric-sub">Cumulative downloads</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Free Apps</div>
        <div class="metric-value">{free_apps_count:,}</div>
        <div class="metric-sub">{free_pct:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Paid Apps</div>
        <div class="metric-value">{paid_apps_count:,}</div>
        <div class="metric-sub">{100 - free_pct:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# Warning if filter returns 0 records
if total_apps == 0:
    st.warning("⚠️ No apps match your selected filters. Please adjust your sidebar criteria.")
    st.stop()


# ==============================================================================
# 7. INTERACTIVE VISUALIZATIONS
# ==============================================================================
st.markdown('<div class="section-title">📊 Visual Analysis & Distributions</div>', unsafe_allow_html=True)

# Row 1: Top 10 Categories & Ratings Distribution (Notebook Cells 6 & 7)
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    # Top 10 Categories (Exact logic from Notebook Cell 6)
    top_cat = filtered_df['Category'].value_counts().head(10).reset_index()
    top_cat.columns = ['Category', 'Number of Apps']
    
    fig_cat = px.bar(
        top_cat,
        x='Number of Apps',
        y='Category',
        orientation='h',
        color='Number of Apps',
        color_continuous_scale='Blues',
        text='Number of Apps',
        title="<b>Top 10 App Categories by Volume</b>"
    )
    fig_cat.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Number of Apps",
        yaxis_title="Category",
        showlegend=False,
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    fig_cat.update_traces(textposition='outside')
    st.plotly_chart(fig_cat, use_container_width=True)

with row1_col2:
    # Distribution of Ratings (Exact logic from Notebook Cell 7)
    fig_rating = px.histogram(
        filtered_df,
        x='Rating',
        nbins=20,
        marginal='box',
        color_discrete_sequence=['#3b82f6'],
        title="<b>Distribution of App Ratings (with Boxplot)</b>"
    )
    fig_rating.update_layout(
        xaxis_title="App Rating (1 to 5 Stars)",
        yaxis_title="Number of Apps",
        bargap=0.08,
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    st.plotly_chart(fig_rating, use_container_width=True)


# Row 2: Free vs Paid Pie Chart & App Installs by Category
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    # Free vs Paid Apps Breakdown
    type_counts = filtered_df['App_Type'].value_counts().reset_index()
    type_counts.columns = ['App_Type', 'Count']
    
    fig_pie = px.pie(
        type_counts,
        names='App_Type',
        values='Count',
        hole=0.45,
        color='App_Type',
        color_discrete_map={'Free': '#10b981', 'Paid': '#f59e0b'},
        title="<b>Free vs Paid Apps Proportion</b>"
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0.05])
    fig_pie.update_layout(
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with row2_col2:
    # Total Installs by Category (Top 10)
    installs_by_cat = filtered_df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10).reset_index()
    installs_by_cat.columns = ['Category', 'Total Installs']
    
    fig_installs = px.bar(
        installs_by_cat,
        x='Category',
        y='Total Installs',
        color='Total Installs',
        color_continuous_scale='Viridis',
        title="<b>Top 10 Categories by Total Installs (Downloads)</b>"
    )
    fig_installs.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Installs",
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    st.plotly_chart(fig_installs, use_container_width=True)


# Row 3: App Ratings by Category & Reviews Distribution
row3_col1, row3_col2 = st.columns([1, 1])

with row3_col1:
    # Mean Rating by Category (Top 15 Categories)
    rating_by_cat = (
        filtered_df.groupby('Category')['Rating']
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    rating_by_cat.columns = ['Category', 'Average Rating']
    
    fig_cat_rating = px.bar(
        rating_by_cat,
        x='Average Rating',
        y='Category',
        orientation='h',
        color='Average Rating',
        color_continuous_scale='Teal',
        title="<b>Top Categories by Average Rating (Mean Score)</b>"
    )
    fig_cat_rating.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[3.5, 5.0]),
        xaxis_title="Average Rating",
        yaxis_title="Category",
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    st.plotly_chart(fig_cat_rating, use_container_width=True)

with row3_col2:
    # Reviews Distribution (Capped Reviews to show clean spread without extreme skew)
    fig_reviews = px.histogram(
        filtered_df,
        x='Reviews_Capped',
        nbins=25,
        color_discrete_sequence=['#8b5cf6'],
        title="<b>Reviews Distribution (IQR Outlier-Capped from Notebook)</b>"
    )
    fig_reviews.update_layout(
        xaxis_title="Reviews (Capped by IQR logic)",
        yaxis_title="Number of Apps",
        bargap=0.08,
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )
    st.plotly_chart(fig_reviews, use_container_width=True)


# ==============================================================================
# 8. SEARCHABLE DATA TABLE & CSV DOWNLOAD
# ==============================================================================
st.markdown('<div class="section-title">📋 Searchable & Filterable Data Table</div>', unsafe_allow_html=True)

# Select columns to display cleanly
display_cols = ['App', 'Category', 'Rating', 'Reviews', 'Installs', 'App_Type', 'Price', 'Content Rating', 'Genres']
display_df = filtered_df[display_cols].sort_values(by=['Installs', 'Rating'], ascending=False)

# Download button in an organized row
col_table_info, col_download = st.columns([3, 1])

with col_table_info:
    st.write(f"Showing **{len(display_df):,}** apps matching current filter criteria:")

with col_download:
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_data,
        file_name="filtered_googleplaystore_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# Interactive Streamlit DataFrame
st.dataframe(
    display_df,
    use_container_width=True,
    height=360,
    column_config={
        "Rating": st.column_config.NumberColumn("Rating ⭐", format="%.1f"),
        "Installs": st.column_config.NumberColumn("Installs 📥", format="%d"),
        "Price": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
        "Reviews": st.column_config.NumberColumn("Reviews 💬", format="%d"),
    }
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Google Play Store Data Analysis & Visualization • College Mini-Project • Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
