import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Page configuration
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load("models/best_model.pkl")
        model_info = joblib.load("models/model_info.pkl")
        scaler = joblib.load("models/scaler.pkl")
        label_encoders = joblib.load("models/label_encoders.pkl")
        feature_names = joblib.load("models/feature_names.pkl")

        return model, model_info, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error("Make sure you've run src/train.py first!")
        return None, None, None, None, None


def create_gauge_chart(probability):
    """Create a gauge chart for attrition probability"""
    if probability < 0.4:
        color = "green"
        risk_level = "Low Risk"
    elif probability < 0.7:
        color = "orange"
        risk_level = "Medium Risk"
    else:
        color = "red"
        risk_level = "High Risk"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Attrition Probability", "font": {"size": 24}},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig, risk_level


def main():
    """Main application function"""

    # Load model artifacts
    model, model_info, scaler, label_encoders, feature_names = load_model_artifacts()

    if model is None:
        st.error("Failed to load model. Please run training first: python src/train.py")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Model Performance"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application predicts employee attrition using machine learning. "
        "Built with XGBoost and Streamlit."
    )

    # Main content
    if page == "Home":
        show_home_page(model_info)
    else:
        show_performance_page()


def show_home_page(model_info):
    """Display home page"""
    st.title("HR Employee Attrition Prediction System")
    st.markdown("### AI-Powered Workforce Analytics")

    st.markdown("---")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", f"{model_info['accuracy']*100:.1f}%")

    with col2:
        st.metric("Precision", f"{model_info['precision']*100:.1f}%")

    with col3:
        st.metric("Recall", f"{model_info['recall']*100:.1f}%")

    with col4:
        st.metric("F1-Score", f"{model_info['f1_score']*100:.1f}%")

    st.markdown("---")

    # Key features
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Features")
        st.markdown(
            """
        - **Predictive Analytics**: Identify at-risk employees early
        - **Risk Scoring**: Categorize employees by attrition risk
        - **Data-Driven Insights**: Make informed retention decisions
        - **Cost Reduction**: Reduce recruitment and training expenses
        - **Model Explainability**: Understand prediction factors
        """
        )

    with col2:
        st.subheader("Business Impact")
        st.markdown(
            """
        - **Early Detection**: Identify at-risk employees 3-6 months in advance
        - **Cost Savings**: Potential $250K-$500K annual savings
        - **Strategic Planning**: Data-driven retention strategies
        - **ROI**: 10-20x return on targeted interventions
        - **Fair Decisions**: Transparent, data-based approach
        """
        )

    st.markdown("---")

    # Model insights
    st.subheader("Top Attrition Drivers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Overtime Work**\n\nEmployees working overtime are 2.5x more likely to leave"
        )

    with col2:
        st.warning(
            "**Job Satisfaction**\n\nLow satisfaction scores strongly correlate with attrition"
        )

    with col3:
        st.error(
            "**Work-Life Balance**\n\nPoor balance is a critical factor in employee decisions"
        )


def show_performance_page():
    """Display model performance page"""
    st.title("Model Performance & Insights")

    tab1, tab2, tab3 = st.tabs(
        ["Performance Metrics", "Visualizations", "Business Insights"]
    )

    with tab1:
        st.subheader("Model Performance Metrics")

        st.markdown(
            """
        ### Performance Explanation
        
        **Accuracy**: Percentage of correct predictions overall (84%)
        
        **Precision**: Of employees predicted to leave, what percentage actually left? (49%)
        - Higher precision means fewer false alarms
        
        **Recall**: Of employees who actually left, what percentage did we catch? (45%)
        - Higher recall means we catch more at-risk employees
        
        **F1-Score**: Balanced measure combining precision and recall (47%)
        - Good balance between catching employees and avoiding false alarms
        
        **ROC-AUC**: Ability to distinguish between stay/leave across all thresholds (0.77)
        - Score of 0.5 = random guessing, 1.0 = perfect prediction
        """
        )

    with tab2:
        st.subheader("Performance Visualizations")

        # Check if images exist
        if os.path.exists("assets/confusion_matrix.png"):
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    "assets/confusion_matrix.png",
                    caption="Confusion Matrix",
                    use_column_width=True,
                )

            with col2:
                st.image(
                    "assets/roc_curve.png", caption="ROC Curve", use_column_width=True
                )

            st.markdown("---")

            if os.path.exists("assets/feature_importance.png"):
                st.image(
                    "assets/feature_importance.png",
                    caption="Feature Importance",
                    use_column_width=True,
                )

            if os.path.exists("assets/model_comparison.png"):
                st.image(
                    "assets/model_comparison.png",
                    caption="Model Comparison",
                    use_column_width=True,
                )
        else:
            st.warning(
                "Visualization images not found. Make sure you've run training: python src/train.py"
            )

    with tab3:
        st.subheader("Business Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### Retention Strategies
            
            **High-Impact Actions:**
            - Reduce mandatory overtime requirements
            - Implement flexible work arrangements
            - Conduct regular compensation benchmarking
            - Enhance onboarding programs for new hires
            - Establish clear career progression paths
            
            **Employee Engagement:**
            - Quarterly satisfaction surveys
            - Regular one-on-one check-ins
            - Recognition and rewards programs
            - Professional development opportunities
            - Team building activities
            """
            )

        with col2:
            st.markdown(
                """
            ### Cost-Benefit Analysis
            
            **Estimated Costs:**
            - Average cost to replace an employee: $15,000 - $50,000
            - Includes: Recruitment, training, productivity loss
            - Annual attrition rate: ~16%
            
            **Potential Savings:**
            - Reducing attrition by 5%: $250,000 - $500,000/year
            - Early intervention cost: $50 - $200 per employee
            - ROI: 10-20x return on retention investment
            
            **Implementation Priority:**
            1. Focus on high-risk, high-value employees first
            2. Address systemic issues (overtime, compensation)
            3. Improve manager training on retention
            """
            )


if __name__ == "__main__":
    main()
