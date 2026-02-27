import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------- INR Formatting Function -------- #
def format_inr(number):
    return f"{number:,.0f}"

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("../advertising_model.pkl")

# Load dataset for performance & insights
df = pd.read_csv("Advertising.csv")
X = df.drop("Sales", axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- HEADER ---------------- #
st.title("📊 Advertising Sales Prediction System")
st.markdown("### Predict product sales based on advertising budgets")
st.divider()

# ---------------- SIDEBAR NAVIGATION ---------------- #
st.sidebar.title("📑 Menu")

page = st.sidebar.radio(
    "Go to",
    ["🔮 Sales Prediction", "📉 Data Analysis", "📊 Model Performance & Insights"]
)

# =====================================================
# 🔮 PAGE 1 — SALES PREDICTION
# =====================================================
if page == "🔮 Sales Prediction":

    st.header("🔮 Sales Prediction")

    mode = st.radio("Select Prediction Mode", ["Single Prediction", "Compare Scenarios", "Budget Optimizer"])

    if mode == "Single Prediction":
        st.sidebar.header("📥 Enter Advertising Budget")

        tv = st.sidebar.slider("📺 TV Budget (₹)", 0, 300000, 150000, step=5000, format="localized")
        radio = st.sidebar.slider("📻 Radio Budget (₹)", 0, 50000, 25000, step=1000, format="localized")
        newspaper = st.sidebar.slider("📰 Newspaper Budget (₹)", 0, 120000, 60000, step=2000, format="localized")

        predict_button = st.sidebar.button("Predict Sales")

        st.subheader("💰 Selected Advertising Budget")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("TV Budget", f"₹{format_inr(tv)}")

        with col2:
            st.metric("Radio Budget", f"₹{format_inr(radio)}")

        with col3:
            st.metric("Newspaper Budget", f"₹{format_inr(newspaper)}")

        st.divider()

        if predict_button:
            # Convert INR to dataset scale
            data = np.array([[tv/1000, radio/1000, newspaper/1000]])
            prediction = model.predict(data)[0]

            st.metric("📈 Predicted Sales", f"{prediction:.2f} Units")
            st.success("Model prediction generated successfully!")

            st.divider()
            st.subheader("💡 Budget Optimization Suggestion")
            total_budget = tv + radio + newspaper

            if total_budget > 0:
                tv_pct = (tv / total_budget) * 100
                news_pct = (newspaper / total_budget) * 100

                if tv_pct < 50:
                    st.warning(f"⚠️ **Low TV Allocation ({tv_pct:.1f}%):** TV advertising historically provides the highest return on investment. Consider shifting some budget from Radio or Newspaper to TV to maximize sales.")
                elif news_pct > 25:
                    st.warning(f"⚠️ **High Newspaper Allocation ({news_pct:.1f}%):** Newspaper advertising generally has the lowest impact on sales. Consider reallocating some of these funds to TV or Radio.")
                else:
                    st.success("✅ **Well Optimized:** Your budget allocation looks solid based on historical advertising performance trends.")

            st.divider()
            
            # Download prediction report
            report_df = pd.DataFrame([{
                "TV Budget (₹)": tv,
                "Radio Budget (₹)": radio,
                "Newspaper Budget (₹)": newspaper,
                "Predicted Sales (Units)": prediction
            }])
            
            st.download_button(
                label="📥 Download Prediction Report (CSV)",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="single_prediction_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    elif mode == "Compare Scenarios":
        st.sidebar.header("📥 Scenario A Budget")
        tv_a = st.sidebar.slider("📺 TV Budget A (₹)", 0, 300000, 100000, step=5000, format="localized")
        radio_a = st.sidebar.slider("📻 Radio Budget A (₹)", 0, 50000, 20000, step=1000, format="localized")
        news_a = st.sidebar.slider("📰 Newspaper Budget A (₹)", 0, 120000, 40000, step=2000, format="localized")

        st.sidebar.divider()

        st.sidebar.header("📥 Scenario B Budget")
        tv_b = st.sidebar.slider("📺 TV Budget B (₹)", 0, 300000, 200000, step=5000, format="localized")
        radio_b = st.sidebar.slider("📻 Radio Budget B (₹)", 0, 50000, 40000, step=1000, format="localized")
        news_b = st.sidebar.slider("📰 Newspaper Budget B (₹)", 0, 120000, 80000, step=2000, format="localized")

        compare_button = st.sidebar.button("Compare Scenarios")

        if compare_button:
            st.subheader("📊 Scenario Comparison")

            # Convert INR to dataset scale
            data_a = np.array([[tv_a/1000, radio_a/1000, news_a/1000]])
            data_b = np.array([[tv_b/1000, radio_b/1000, news_b/1000]])

            pred_a = model.predict(data_a)[0]
            pred_b = model.predict(data_b)[0]
            
            diff = pred_b - pred_a

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Scenario A Predicted Sales", f"{pred_a:.2f} Units", f"Total Budget: ₹{format_inr(tv_a + radio_a + news_a)}")
            
            with col2:
                st.metric("Scenario B Predicted Sales", f"{pred_b:.2f} Units", f"Total Budget: ₹{format_inr(tv_b + radio_b + news_b)}")

            with col3:
                st.metric("Difference (B - A)", f"{abs(diff):.2f} Units", f"{'+' if diff > 0 else '-'}{abs(diff):.2f} Units", delta_color="normal" if diff > 0 else "inverse")

            st.success("Scenarios compared successfully!")

            st.divider()

            # Download scenario comparison report
            scenario_df = pd.DataFrame([
                {"Scenario": "A", "TV Budget (₹)": tv_a, "Radio Budget (₹)": radio_a, "Newspaper Budget (₹)": news_a, "Total Budget (₹)": tv_a + radio_a + news_a, "Predicted Sales (Units)": pred_a},
                {"Scenario": "B", "TV Budget (₹)": tv_b, "Radio Budget (₹)": radio_b, "Newspaper Budget (₹)": news_b, "Total Budget (₹)": tv_b + radio_b + news_b, "Predicted Sales (Units)": pred_b},
            ])
            st.download_button(
                label="📥 Download Comparison Report (CSV)",
                data=scenario_df.to_csv(index=False).encode("utf-8"),
                file_name="scenario_comparison_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    elif mode == "Budget Optimizer":
        st.sidebar.header("📥 Enter Total Budget")
        total_budget = st.sidebar.number_input("Total Investment (₹)", min_value=10000, max_value=1000000, value=200000, step=5000)

        optimize_button = st.sidebar.button("Get Optimization Suggestion")

        if optimize_button:
            st.subheader("💡 Optimal Budget Allocation Suggestion")
            st.markdown("We ran thousands of simulations using the trained model to find the best way to distribute your budget for maximum sales.")

            # Random Search for Optimization
            np.random.seed(42)
            num_samples = 5000

            # Generate random allocations ensuring logical bounds for advertising
            tv_allocs = np.random.uniform(0.1, 0.9, num_samples)
            radio_allocs = np.random.uniform(0.01, 0.6, num_samples)
            news_allocs = np.random.uniform(0.0, 0.4, num_samples)

            totals = tv_allocs + radio_allocs + news_allocs
            tv_allocs /= totals
            radio_allocs /= totals
            news_allocs /= totals

            tv_budgets = tv_allocs * total_budget
            radio_budgets = radio_allocs * total_budget
            news_budgets = news_allocs * total_budget

            # Predict sales for all combinations (convert to dataset scale)
            combinations = np.column_stack((tv_budgets/1000, radio_budgets/1000, news_budgets/1000))
            predictions = model.predict(combinations)

            best_idx = np.argmax(predictions)
            best_tv = tv_budgets[best_idx]
            best_radio = radio_budgets[best_idx]
            best_news = news_budgets[best_idx]
            max_sales = predictions[best_idx]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📺 Suggested TV Budget", f"₹{format_inr(best_tv)}")
            with col2:
                st.metric("📻 Suggested Radio Budget", f"₹{format_inr(best_radio)}")
            with col3:
                st.metric("📰 Suggested News Budget", f"₹{format_inr(best_news)}")

            st.divider()
            
            st.metric("🚀 Expected Maximum Sales", f"{max_sales:.2f} Units")

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([best_tv, best_radio, best_news], labels=["TV", "Radio", "Newspaper"], 
                   autopct='%1.1f%%', colors=["#ff9999", "#66b3ff", "#99ff99"], startangle=90)
            ax.set_title("Recommended Budget Distribution")
            st.pyplot(fig)
            
            st.success("Optimization complete! The suggested allocation provides the absolute highest predicted return on investment based on historical data.")

            st.divider()

            # Download suggested allocation report
            opt_df = pd.DataFrame([{
                "Total Budget (₹)": total_budget,
                "Suggested TV Budget (₹)": best_tv,
                "Suggested Radio Budget (₹)": best_radio,
                "Suggested Newspaper Budget (₹)": best_news,
                "Expected Max Sales (Units)": max_sales
            }])
            
            st.download_button(
                label="📥 Download Optimization Report (CSV)",
                data=opt_df.to_csv(index=False).encode("utf-8"),
                file_name="budget_optimization_report.csv",
                mime="text/csv",
                use_container_width=True
            )

# =====================================================
# 📉 PAGE 2 — DATA ANALYSIS
# =====================================================
elif page == "📉 Data Analysis":

    st.header("📉 Data Analysis")

    st.markdown("### Correlation Between Advertising Budgets & Sales")
    
    # Calculate Correlation Matrix
    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    
    st.pyplot(fig)

    st.info("💡 **Insight:** Values closer to 1 indicate a strong positive correlation. For example, a high correlation between TV and Sales means higher TV budget leads to more sales.")

# =====================================================
# 📊 PAGE 3 — MODEL PERFORMANCE & INSIGHTS
# =====================================================
elif page == "📊 Model Performance & Insights":

    st.header("📊 Model Performance & Insights")

    st.subheader("KPI Summary Cards")

    y_pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Average Sales", f"{df['Sales'].mean():.2f}")

    with col2:
        st.metric("📈 Max Sales", f"{df['Sales'].max():.2f}")

    with col3:
        st.metric("📉 Min Sales", f"{df['Sales'].min():.2f}")

    with col4:
        st.metric("🎯 Model Accuracy (R²)", f"{r2:.3f}")

    st.divider()

    st.subheader("Model Error Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", f"{mae:.2f}")

    with col2:
        st.metric("RMSE", f"{rmse:.2f}")

    st.caption("Metrics calculated on 30% test dataset.")

    st.divider()

    st.subheader("🤖 Model Comparison")
    st.markdown("We trained and evaluated multiple machine learning models. The Random Forest Regressor was selected for the final prediction app because of its high R² Score and low error margin.")
    
    try:
        model_comp_df = pd.read_csv("model_comparison.csv")
        st.dataframe(model_comp_df, use_container_width=True, hide_index=True)
        
        # Download model comparison report
        st.download_button(
            label="📥 Download Model Comparison Report (CSV)",
            data=model_comp_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_report.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.warning("Model comparison data not found. Please run the training script.")

    st.divider()

    st.subheader("Model Insights")

    col1, col2 = st.columns(2)

    # -------- Feature Importance -------- #
    with col1:
        st.markdown("**Feature Importance**")

        importance = model.feature_importances_
        features = ["TV", "Radio", "Newspaper"]

        fig, ax = plt.subplots()
        ax.bar(features, importance)
        ax.set_title("Feature Importance")
        ax.set_ylabel("Importance Score")

        st.pyplot(fig)

        st.info("TV advertising usually has the highest influence on sales.")

    # -------- Residual Plot -------- #
    with col2:
        st.markdown("**Residual Plot**")

        residuals = y_test - y_pred_test

        fig, ax = plt.subplots()
        ax.scatter(y_pred_test, residuals)
        ax.axhline(y=0, linestyle="--")
        ax.set_xlabel("Predicted Sales")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")

        st.pyplot(fig)

        st.info("Residuals randomly scattered around zero indicate a good fit.")

# ---------------- FOOTER ---------------- #
st.markdown(
    """
    ---
    📌 **Project:** Advertising Sales Prediction  
    🤖 **Model:** Tuned Random Forest Regressor  
    🎓 Built using Machine Learning & Streamlit
    """

)
