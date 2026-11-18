# Google Play Store Data Analysis Dashboard

🚀 **Live Dashboard:** https://google-play-store-analysis-bn1f.onrender.com/

A data analysis dashboard for exploring Google Play Store applications, ratings, installs, categories, reviews, and pricing.

# Google Play Store Data Analysis - Interactive Streamlit Dashboard

A modern, interactive, and professional college mini-project dashboard for **Google Play Store Data Analysis & Visualization**, built using **Python, Streamlit, Pandas, and Plotly**.

This project directly implements the data preprocessing, cleaning, outlier handling, and exploratory analysis developed in the college project notebook `DWP_mini_project.ipynb`.

---

## 🌟 Key Features

1. **KPI Metric Cards**:
   - **Total Apps**: Real-time count of apps matching current filters.
   - **Average Rating**: Mean star rating of filtered apps.
   - **Total Installs**: Total cumulative downloads across filtered apps.
   - **Free vs. Paid Apps**: Count and percentage breakdown of apps by pricing model.

2. **Interactive Visualizations (Plotly)**:
   - **Top 10 App Categories**: Horizontal bar chart of most popular categories by volume.
   - **Rating Distribution**: Interactive histogram with boxplot spread (Notebook Cell 7).
   - **Free vs. Paid Breakdown**: Donut chart displaying app monetization proportions.
   - **Installs by Category**: Total downloads aggregated across top categories.
   - **Average Rating by Category**: Top categories ranked by mean user rating score.
   - **Reviews Distribution**: Histogram showing user engagement using IQR outlier-capped reviews (Notebook Cell 5).

3. **Multi-Faceted Interactive Filters (Sidebar)**:
   - Multi-select **Category** filter.
   - Radio selector for **App Type** (`All`, `Free`, `Paid`).
   - Range slider for **Minimum / Maximum Rating** (1.0 to 5.0 stars).
   - Multi-select **Content Rating** (`Everyone`, `Teen`, `Mature 17+`, etc.).
   - Instant text search by **App Name**.
   - One-click **Reset All Filters** button.

4. **Searchable Data Table & CSV Export**:
   - Filterable, sortable interactive data table.
   - Instant **Download Filtered CSV** button for exporting custom slices.

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit dashboard application
├── googleplaystore.csv         # Google Play Store dataset
├── requirements.txt            # Python dependencies (streamlit, pandas, plotly, numpy)
└── README.md                   # Setup guide and college viva / presentation notes
```

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have **Python 3.9+** installed on your system.

### Step 1: Open Terminal or VS Code
Open the project directory in VS Code or your terminal:
```bash
cd c:\Users\rites\Downloads\NEEW
```

### Step 2: Install Required Dependencies
Run the following command in your terminal or command prompt:
```bash
pip install -r requirements.txt
```
*(Or install directly: `pip install streamlit pandas numpy plotly`)*

### Step 3: Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

Your default web browser will automatically open with the dashboard running at:
```text
http://localhost:8501
```

---

## 🎓 College Viva / Teacher Explanation Guide

When explaining this project to your college instructor or evaluator:

1. **Data Preprocessing & Cleaning (Notebook alignment)**:
   - *Duplicate Removal*: Used `df.drop_duplicates()` to eliminate repeated app entries.
   - *Missing Value Handling*: Imputed missing `Rating` values with the statistical **median** (`df['Rating'].median() = 4.3`) to prevent distortion from skewed data. Replaced missing `Reviews` with `0`. Dropped any remaining null records using `dropna()`.
   - *Data Type Conversion*: Cleaned string artifacts (e.g. `+`, `,` in `Installs` and `$` in `Price`) and converted both into numeric floats/integers.
   - *Feature Engineering*: Derived the `App_Type` feature (`Free` if `Price == 0`, else `Paid`).
   - *Outlier Capping (IQR Method)*: Calculated $IQR = Q3 - Q1$, determined upper/lower bounds ($Q \pm 1.5 \times IQR$), and capped extreme review counts with `.clip()` to ensure realistic distributions.

2. **Why Streamlit & Plotly?**:
   - Streamlit provides an immediate, reactive web interface where metric cards and plots update instantly whenever sidebar filters change.
   - Plotly adds interactive tooltips, zoom, and responsive sizing while keeping the underlying Python logic lightweight and clean.
