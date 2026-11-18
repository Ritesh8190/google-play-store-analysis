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

    /* Pipeline Step Box */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        margin: 15px 0 25px 0;
    }
    .pipeline-step {
        flex: 1;
        min-width: 140px;
        background: #ffffff;
        border-top: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .pipeline-step-num {
        font-size: 0.75rem;
        font-weight: 700;
        color: #3b82f6;
        text-transform: uppercase;
    }
    .pipeline-step-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 2px;
    }
    .pipeline-arrow {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA LOADING & PROCESSING (Exact logic from your Colab Notebook)
# ==============================================================================
@st.cache_data
def load_all_stages():
    """
    Loads Google Play Store dataset and runs through each step of the Colab notebook:
    Returns both the intermediate artifacts (for the Data Preparation page)
    and the final cleaned dataframe (for the Analytics Dashboard).
    """
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
        return None, None

    # Step 0: Raw Dataset (Notebook Cell 1 & 2)
    raw_df_initial = pd.read_csv(file_path)
    orig_shape = raw_df_initial.shape
    orig_dtypes = raw_df_initial.dtypes.astype(str).to_dict()
    missing_before = raw_df_initial.isnull().sum()
    duplicate_rows_count = int(raw_df_initial.duplicated().sum())

    # Step 1: Remove duplicates (Notebook Cell 3)
    df_dedup = raw_df_initial.drop_duplicates()
    dedup_shape = df_dedup.shape

    # Step 2: Fill Rating missing with median (Notebook Cell 3)
    rating_median = float(df_dedup['Rating'].median())
    rating_missing_count = int(df_dedup['Rating'].isnull().sum())
    df_dedup['Rating'] = df_dedup['Rating'].fillna(rating_median)

    # Step 3: Fill Reviews missing with 0 (Notebook Cell 3)
    reviews_missing_count = int(df_dedup['Reviews'].isnull().sum())
    df_dedup['Reviews'] = df_dedup['Reviews'].fillna(0)

    # Step 4: Drop remaining rows with missing values (Notebook Cell 3)
    missing_before_drop = df_dedup.isnull().sum().to_dict()
    df_clean = df_dedup.dropna()
    clean_shape = df_clean.shape
    dropped_rows = dedup_shape[0] - clean_shape[0]

    # Pre-transformation sample
    transformation_sample_before = df_clean[['App', 'Installs', 'Price']].head(6).copy()

    # Step 5: Data Transformation - Installs & Price (Notebook Cell 4)
    df_clean['Installs'] = pd.to_numeric(
        df_clean['Installs'].astype(str).str.replace(r'[,+]', '', regex=True),
        errors='coerce'
    ).fillna(0).astype('int64')

    df_clean['Price'] = pd.to_numeric(
        df_clean['Price'].astype(str).str.replace('$', '', regex=False),
        errors='coerce'
    ).fillna(0).astype('float64')

    # Step 6: Create App_Type derived feature (Notebook Cell 4)
    df_clean['App_Type'] = np.where(df_clean['Price'] == 0, 'Free', 'Paid')

    # Post-transformation sample
    transformation_sample_after = df_clean[['App', 'Installs', 'Price', 'App_Type']].head(6).copy()

    # Step 7: Convert Reviews to integer (Notebook Cell 5)
    df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce').fillna(0).astype('int64')

    final_dtypes = df_clean.dtypes.astype(str).to_dict()

    # Step 8: IQR Outlier Detection on Reviews (Notebook Cell 5)
    Q1 = float(df_clean['Reviews'].quantile(0.25))
    Q3 = float(df_clean['Reviews'].quantile(0.75))
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (df_clean['Reviews'] < lower_bound) | (df_clean['Reviews'] > upper_bound)
    outliers_count = int(outliers_mask.sum())

    # Step 9: Outlier Capping using clip() (Notebook Cell 5)
    # Notice: Original 'Reviews' column is preserved; new column 'Reviews_Capped' is created
    df_clean['Reviews_Capped'] = df_clean['Reviews'].clip(lower_bound, upper_bound)

    prep_stats = {
        'orig_shape': orig_shape,
        'orig_dtypes': orig_dtypes,
        'missing_before': missing_before,
        'duplicate_rows_count': duplicate_rows_count,
        'dedup_shape': dedup_shape,
        'rating_median': rating_median,
        'rating_missing_count': rating_missing_count,
        'reviews_missing_count': reviews_missing_count,
        'missing_before_drop': missing_before_drop,
        'clean_shape': clean_shape,
        'dropped_rows': dropped_rows,
        'transformation_sample_before': transformation_sample_before,
        'transformation_sample_after': transformation_sample_after,
        'final_dtypes': final_dtypes,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': outliers_count,
    }

    return df_clean, prep_stats


clean_df, prep_stats = load_all_stages()

# Handle case where file is missing
if clean_df is None:
    st.error("⚠️ `googleplaystore.csv` not found in the project folder!")
    st.info("Please ensure `googleplaystore.csv` is placed in the same folder as `app.py`.")
    st.stop()


# ==============================================================================
# 3. SIDEBAR NAVIGATION & FILTERS
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Play_Arrow_logo.svg", width=55)
st.sidebar.title("Navigation")
current_page = st.sidebar.radio(
    "Choose View:",
    options=["📊 Analytics Dashboard", "🛠️ Data Preparation & Processing"],
    index=0
)

st.sidebar.markdown("---")

if current_page == "📊 Analytics Dashboard":
    st.sidebar.title("Dashboard Filters")
    st.sidebar.markdown("Filter and slice data interactively:")

    # Filter 1: App Category Multi-Select
    all_categories = sorted(clean_df['Category'].unique().tolist())
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
    min_rating, max_rating = float(clean_df['Rating'].min()), float(clean_df['Rating'].max())
    selected_rating = st.sidebar.slider(
        "⭐ Minimum Rating Range",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.1,
        help="Select the minimum and maximum app ratings to display."
    )

    # Filter 4: Content Rating
    all_content_ratings = sorted(clean_df['Content Rating'].unique().tolist())
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
    "**Notebook:** `DWP_mini_project.ipynb`"
)


# ==============================================================================
# 4. VIEW ROUTING: DATA PREPARATION & PROCESSING
# ==============================================================================
if current_page == "🛠️ Data Preparation & Processing":
    # Header
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🛠️ Data Preparation & Processing</h1>
        <p class="header-subtitle">
            Demonstrating the exact Data Cleaning, Transformation, Preprocessing, and Outlier Handling performed in the Jupyter Notebook (<code>DWP_mini_project.ipynb</code>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. Visual Pipeline
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-title">🔄 5. Data Processing Workflow Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-step-num">Step 1</div>
            <div class="pipeline-step-title">📁 Raw Dataset</div>
            <div style="font-size: 0.78rem; color: #64748b;">10,841 rows × 13 cols</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="pipeline-step-num">Step 2</div>
            <div class="pipeline-step-title">🧹 Data Cleaning</div>
            <div style="font-size: 0.78rem; color: #64748b;">Duplicates & Missing Imputation</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="pipeline-step-num">Step 3</div>
            <div class="pipeline-step-title">🔄 Transformation</div>
            <div style="font-size: 0.78rem; color: #64748b;">Installs, Price & App_Type</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="pipeline-step-num">Step 4</div>
            <div class="pipeline-step-title">🎯 Outlier Capping</div>
            <div style="font-size: 0.78rem; color: #64748b;">IQR Clipping on Reviews</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div class="pipeline-step-num">Step 5</div>
            <div class="pipeline-step-title">📈 EDA & Dashboard</div>
            <div style="font-size: 0.78rem; color: #64748b;">Visual Charts & Metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 6. Data Preprocessing (Comprehensive Overview)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-title">⚙️ 6. Data Preprocessing Overview</div>', unsafe_allow_html=True)
    st.write(
        "Before performing statistical analysis or exploratory visualizations, the dataset was inspected and "
        "preprocessed to fix inconsistencies, replace missing observations, remove redundant rows, and cast columns into appropriate numeric types."
    )

    c_prep1, c_prep2, c_prep3 = st.columns(3)
    with c_prep1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Original Shape</div>
            <div class="metric-value">{prep_stats['orig_shape'][0]:,} × {prep_stats['orig_shape'][1]}</div>
            <div class="metric-sub">Rows × Columns before preprocessing</div>
        </div>
        """, unsafe_allow_html=True)
    with c_prep2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cleaned Shape</div>
            <div class="metric-value">{prep_stats['clean_shape'][0]:,} × {prep_stats['clean_shape'][1] + 1}</div>
            <div class="metric-sub">Rows × Columns after preprocessing & feature creation</div>
        </div>
        """, unsafe_allow_html=True)
    with c_prep3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Derived Features Created</div>
            <div class="metric-value">2 Columns</div>
            <div class="metric-sub"><code>App_Type</code> and <code>Reviews_Capped</code></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Preprocessing Table: Column, Original Dtype, Final Dtype, Operation Performed
    st.markdown("##### 📋 Column Preprocessing & Data Type Transformation Summary")
    preprocessing_summary = [
        {"Column": "App", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Retained as text identifier for apps."},
        {"Column": "Category", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Retained as primary app classification category."},
        {"Column": "Rating", "Original Type": "float64", "Final Type": "float64", "Notebook Preprocessing Operation": f"Imputed 1,465 missing values using median ({prep_stats['rating_median']})."},
        {"Column": "Reviews", "Original Type": "object (string)", "Final Type": "int64", "Notebook Preprocessing Operation": "Filled missing values with 0, converted to integer via pd.to_numeric()."},
        {"Column": "Size", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Retained in original alphanumeric format (e.g. '19M', 'Varies with device')."},
        {"Column": "Installs", "Original Type": "object (string)", "Final Type": "int64", "Notebook Preprocessing Operation": "Removed commas and '+' symbols (regex: '[,+]'), cast to numeric integer."},
        {"Column": "Type", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "1 row with null dropped in df.dropna()."},
        {"Column": "Price", "Original Type": "object (string)", "Final Type": "float64", "Notebook Preprocessing Operation": "Stripped '$' symbol, cast to numeric float via pd.to_numeric()."},
        {"Column": "Content Rating", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "1 row with null dropped in df.dropna()."},
        {"Column": "Genres", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Retained as sub-genre tag."},
        {"Column": "Last Updated", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Retained as update date string."},
        {"Column": "Current Ver", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "8 rows with null dropped in df.dropna()."},
        {"Column": "Android Ver", "Original Type": "object (string)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "3 rows with null dropped in df.dropna()."},
        {"Column": "App_Type (Derived)", "Original Type": "None (New)", "Final Type": "object (string)", "Notebook Preprocessing Operation": "Created using np.where: Price == 0 ➔ 'Free', else ➔ 'Paid'."},
        {"Column": "Reviews_Capped (Derived)", "Original Type": "None (New)", "Final Type": "float64", "Notebook Preprocessing Operation": f"Created via .clip({prep_stats['lower_bound']:.0f}, {prep_stats['upper_bound']:.0f}) to handle extreme outliers."},
    ]
    st.table(pd.DataFrame(preprocessing_summary))

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 1. Data Cleaning
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-title">🧹 1. Data Cleaning (Notebook Cell 3)</div>', unsafe_allow_html=True)
    st.write(
        "The notebook followed a clear 4-step sequence to handle duplicates and missing entries: "
        "removing duplicates first, imputing missing ratings with median, filling reviews with 0, and lastly dropping remaining nulls."
    )

    clean_c1, clean_c2, clean_c3, clean_c4 = st.columns(4)
    with clean_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Original Rows</div>
            <div class="metric-value">{prep_stats['orig_shape'][0]:,}</div>
            <div class="metric-sub">{prep_stats['orig_shape'][1]} initial columns</div>
        </div>
        """, unsafe_allow_html=True)
    with clean_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duplicates Removed</div>
            <div class="metric-value">{prep_stats['duplicate_rows_count']:,}</div>
            <div class="metric-sub"><code>df.drop_duplicates()</code></div>
        </div>
        """, unsafe_allow_html=True)
    with clean_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rating Imputation</div>
            <div class="metric-value">Median ({prep_stats['rating_median']})</div>
            <div class="metric-sub">{prep_stats['rating_missing_count']:,} null ratings imputed</div>
        </div>
        """, unsafe_allow_html=True)
    with clean_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Final Cleaned Rows</div>
            <div class="metric-value">{prep_stats['clean_shape'][0]:,}</div>
            <div class="metric-sub">{prep_stats['dropped_rows']} remaining null rows dropped</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Missing values breakdown
    col_miss_info, col_miss_chart = st.columns([1, 1])
    with col_miss_info:
        st.markdown("##### 🔍 Missing Value Handling Logic")
        st.markdown(f"""
        - **Original Missing Values:**
          - `Rating`: **1,474 missing values** (reduced to **1,465** after duplicate removal).
          - `Current Ver`: **8 missing**.
          - `Android Ver`: **3 missing**.
          - `Type`: **1 missing**.
          - `Content Rating`: **1 missing**.
          - `Reviews`: **0 missing** in raw file, but guarded in notebook.
        - **Imputation Strategy from Notebook:**
          1. **`df['Rating'] = df['Rating'].fillna(df['Rating'].median())`**: Rating is numeric but skewed, so median (**{prep_stats['rating_median']}**) preserves central tendency without being affected by extreme scores.
          2. **`df['Reviews'] = df['Reviews'].fillna(0)`**: Reviews null values safely defaulted to 0.
          3. **`df = df.dropna()`**: Dropped remaining **{prep_stats['dropped_rows']} rows** containing missing metadata (`Current Ver`, `Android Ver`, `Type`, `Content Rating`).
        - **Result:** **0 missing values remaining** across all columns!
        """)

    with col_miss_chart:
        missing_df = pd.DataFrame({
            "Column": prep_stats['missing_before'].index,
            "Missing Count": prep_stats['missing_before'].values
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=True)
        fig_miss = px.bar(
            missing_df,
            x="Missing Count",
            y="Column",
            orientation="h",
            text="Missing Count",
            title="<b>Missing Values Count (Before Cleaning)</b>",
            color="Missing Count",
            color_continuous_scale="Reds"
        )
        fig_miss.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20), plot_bgcolor="rgba(0,0,0,0)")
        fig_miss.update_traces(textposition='outside')
        st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. Data Transformation
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-title">🔄 2. Data Transformation (Notebook Cell 4)</div>', unsafe_allow_html=True)
    st.write(
        "In the raw dataset, `Installs` and `Price` are stored as string objects with symbols like commas, plus signs, and dollar signs. "
        "The notebook converted them to numeric formats and created a derived feature `App_Type`."
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        ##### 📥 Installs Conversion
        - **Original Format:** Text string with commas and plus signs (e.g. `"10,000+"`, `"500,000+"`).
        - **Transformation:**
          ```python
          df['Installs'] = pd.to_numeric(
              df['Installs'].str.replace(r'[,+]', '', regex=True),
              errors='coerce'
          ).fillna(0)
          ```
        - **Result:** Clean integers (e.g. `10000`, `500000`) enabling mathematical calculations.
        """)

        st.markdown("""
        ##### 💵 Price & App_Type Creation
        - **Original Format:** Text string with dollar symbols (e.g. `"$4.99"`, `"0"`).
        - **Transformation:**
          ```python
          df['Price'] = pd.to_numeric(
              df['Price'].str.replace('$', '', regex=False),
              errors='coerce'
          ).fillna(0)

          # Identify Free and Paid apps
          df['App_Type'] = np.where(df['Price'] == 0, 'Free', 'Paid')
          ```
        - **Result:** `Price` is a floating-point number, and `App_Type` classifies each app as **Free** or **Paid**.
        """)

    with t_col2:
        st.markdown("##### 📊 Real Dataset Before & After Transformation Sample")
        
        # Real before sample with mixed Free and Paid
        demo_before = pd.DataFrame({
            "App": ["Photo Editor & Candy Camera", "Coloring book moana", "TurboScan: scan documents", "Tiny Scanner Pro", "Puffin Browser Pro"],
            "Installs (Before)": ["10,000+", "500,000+", "100,000+", "100,000+", "100,000+"],
            "Price (Before)": ["0", "0", "$4.99", "$4.99", "$3.99"]
        })

        demo_after = pd.DataFrame({
            "App": ["Photo Editor & Candy Camera", "Coloring book moana", "TurboScan: scan documents", "Tiny Scanner Pro", "Puffin Browser Pro"],
            "Installs (Numeric)": [10000, 500000, 100000, 100000, 100000],
            "Price ($ Numeric)": [0.00, 0.00, 4.99, 4.99, 3.99],
            "App_Type (Derived)": ["Free", "Free", "Paid", "Paid", "Paid"]
        })

        st.caption("**Before Transformation (String format with symbols):**")
        st.dataframe(demo_before, use_container_width=True, hide_index=True)
        st.caption("**After Transformation (Clean numeric columns + App_Type):**")
        st.dataframe(demo_after, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3 & 4. Outlier Detection & Handling
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-title">🎯 3 & 4. Outlier Detection & Handling (Notebook Cell 5)</div>', unsafe_allow_html=True)
    st.write(
        "User reviews in the Google Play Store span across several orders of magnitude (from 1 review to tens of millions). "
        "The notebook utilized the standard statistical **Interquartile Range (IQR) method** to identify extreme outliers in `Reviews`."
    )

    out_c1, out_c2, out_c3, out_c4, out_c5 = st.columns(5)
    with out_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Q1 (25th %)</div>
            <div class="metric-value">{prep_stats['Q1']:,.0f}</div>
            <div class="metric-sub">25% apps have ≤ 32 reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with out_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Q3 (75th %)</div>
            <div class="metric-value">{prep_stats['Q3']:,.0f}</div>
            <div class="metric-sub">75% apps have ≤ 46,598 reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with out_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IQR (Q3 - Q1)</div>
            <div class="metric-value">{prep_stats['IQR']:,.0f}</div>
            <div class="metric-sub">Spread of middle 50%</div>
        </div>
        """, unsafe_allow_html=True)
    with out_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Upper Bound</div>
            <div class="metric-value">{prep_stats['upper_bound']:,.1f}</div>
            <div class="metric-sub">Q3 + 1.5 × IQR</div>
        </div>
        """, unsafe_allow_html=True)
    with out_c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Outliers Detected</div>
            <div class="metric-value">{prep_stats['outliers_count']:,}</div>
            <div class="metric-sub">{prep_stats['outliers_count'] / prep_stats['clean_shape'][0] * 100:.1f}% of total apps</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col_out_formula, col_out_explanation = st.columns([1, 1])
    with col_out_formula:
        st.markdown("##### 📐 Mathematical Formulas Used in Notebook")
        st.latex(r"\text{IQR} = Q_3 - Q_1 = 46,598.25 - 32.0 = 46,566.25")
        st.latex(r"\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR} = 32.0 - 1.5(46,566.25) = -69,817.38")
        st.latex(r"\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR} = 46,598.25 + 1.5(46,566.25) = 116,447.63")
        st.info(
            "Since review counts cannot be negative, all values below the upper bound (116,447.6) are within the normal range. "
            "Apps with reviews strictly exceeding 116,447.6 are flagged as statistical outliers."
        )

    with col_out_explanation:
        st.markdown("##### 🛡️ Capping Technique: `Reviews_Capped`")
        st.markdown(f"""
        - **Important Distinction:** The original `Reviews` column was **NOT overwritten or deleted**.
        - Instead, the notebook created a brand-new feature called **`Reviews_Capped`** using pandas `.clip()`:
          ```python
          df['Reviews_Capped'] = df['Reviews'].clip(lower, upper)
          ```
        - Any app with more than **{prep_stats['upper_bound']:,.1f}** reviews has its `Reviews_Capped` set to **{prep_stats['upper_bound']:,.1f}**.
        - This preserves the raw data integrity while providing a clean, non-skewed distribution for visual modeling.
        """)

    # Visual Comparison Before vs After Capping
    st.markdown("##### 📊 Visual Comparison: Original Reviews vs Capped Reviews")
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        fig_raw_rev = px.box(
            clean_df,
            y="Reviews",
            title="<b>Original Reviews (Extreme Skew up to 78M)</b>",
            color_discrete_sequence=["#ef4444"]
        )
        fig_raw_rev.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_raw_rev, use_container_width=True)
        st.caption("⚠️ The box is compressed completely flat near 0 due to viral apps with 78M+ reviews.")

    with v_col2:
        fig_cap_rev = px.box(
            clean_df,
            y="Reviews_Capped",
            title=f"<b>Reviews_Capped (Clipped at {prep_stats['upper_bound']:,.0f})</b>",
            color_discrete_sequence=["#10b981"]
        )
        fig_cap_rev.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cap_rev, use_container_width=True)
        st.caption("✅ Outliers are neatly capped at the upper fence, enabling clear visualization of the interquartile range.")

    st.markdown("---")


# ==============================================================================
# 5. VIEW ROUTING: ANALYTICS DASHBOARD (Preserved Exactly)
# ==============================================================================
else:
    # Filter dataset based on sidebar criteria
    filtered_df = clean_df.copy()

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

    # Header
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">📱 Google Play Store Data Analytics Dashboard</h1>
        <p class="header-subtitle">
            An interactive exploratory dashboard analyzing app categories, ratings, downloads, pricing models, and user reviews.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Metrics Cards
    total_apps = len(filtered_df)
    avg_rating = filtered_df['Rating'].mean() if total_apps > 0 else 0.0
    total_installs = filtered_df['Installs'].sum() if total_apps > 0 else 0
    free_apps_count = (filtered_df['App_Type'] == 'Free').sum() if total_apps > 0 else 0
    paid_apps_count = (filtered_df['App_Type'] == 'Paid').sum() if total_apps > 0 else 0
    free_pct = (free_apps_count / total_apps * 100) if total_apps > 0 else 0.0

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
            <div class="metric-sub">Filtered / {len(clean_df):,} total</div>
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

    # Visual Analysis & Distributions
    st.markdown('<div class="section-title">📊 Visual Analysis & Distributions</div>', unsafe_allow_html=True)

    # Row 1: Top 10 Categories & Ratings Distribution (Notebook Cells 6 & 7)
    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
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

    # Searchable Data Table
    st.markdown('<div class="section-title">📋 Searchable & Filterable Data Table</div>', unsafe_allow_html=True)

    display_cols = ['App', 'Category', 'Rating', 'Reviews', 'Installs', 'App_Type', 'Price', 'Content Rating', 'Genres']
    display_df = filtered_df[display_cols].sort_values(by=['Installs', 'Rating'], ascending=False)

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

# Common Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Google Play Store Data Analysis & Visualization • College Mini-Project • Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
