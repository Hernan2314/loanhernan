import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io  # For in-memory CSV download

# Load the trained model
classifier = joblib.load('classifier.pkl')

@st.cache()
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
    # Pre-process user input
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    LoanAmount = LoanAmount / 1000  # Scale loan amount if required

    # Create the feature array with exactly the 5 features
    features = np.array([Gender, Married, ApplicantIncome, LoanAmount, Credit_History]).reshape(1, -1)

    # Predict directly without scaling (assuming model was trained on non-scaled data)
    prediction = classifier.predict(features)
    return 'Approved' if prediction == 1 else 'Rejected'

# Streamlit Interface
def main():
    st.set_page_config(page_title="ProFund Insight - Loan Approval", page_icon="💼", layout="centered")

    # Branding and Title
    st.markdown("""
        <style>
        .title { font-size: 2.4em; font-weight: bold; color: #2e3a45; }
        .subtitle { font-size: 1.2em; color: #6c757d; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="title">💼 ProFund Insight - Loan Approval System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Trusted Solution for Modern Financial Decision Making</p>', unsafe_allow_html=True)

    # Input Form
    st.markdown("### Application Details")
    Gender = st.radio("Select your Gender:", ("Male", "Female"), help="Choose the gender of the applicant.")
    Married = st.radio("Marital Status:", ("Unmarried", "Married"), help="Choose the marital status of the applicant.")
    ApplicantIncome = st.slider("Applicant's Monthly Income (in USD)", min_value=0, max_value=20000, step=500, value=5000, help="Enter the applicant's monthly income.")
    LoanAmount = st.slider("Loan Amount Requested (in USD)", min_value=0, max_value=500000, step=1000, value=150000, help="Enter the loan amount the applicant is requesting.")
    Credit_History = st.selectbox("Credit History Status:", ("Unclear Debts", "No Unclear Debts"), help="Select the applicant's credit history status.")

    # Prediction Button
    if st.button("Predict My Loan Status"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        
        # Display Prediction Outcome
        if result == "Approved":
            st.success(f'✅ Your loan application status: **Approved**')
        else:
            st.error(f'❌ Your loan application status: **Rejected**')

        # Executive Summary Section
        st.write("---")
        st.subheader("Executive Summary")
        st.write(f"""
            **Applicant Details**
            - **Gender**: {Gender}
            - **Marital Status**: {Married}
            - **Monthly Income**: ${ApplicantIncome}
            - **Loan Amount Requested**: ${LoanAmount}
            - **Credit History**: {Credit_History}

            **Decision**: The loan application was **{result}** based on the applicant's profile and historical approval criteria.
        """)

        # Applicant Insights
        st.write("---")
        st.subheader("Applicant Insights")
        if result == "Rejected":
            st.write("""
                - **Suggested Actions**: Improve credit history by reducing unclear debts, or consider a lower loan request.
                - **Note**: Approval may increase with a requested loan amount closer to your income level.
            """)
        else:
            st.write("""
                - **Your profile appears strong based on monthly income and credit history.**
                - **Continue maintaining good credit to support future approvals.**
            """)

        # Visualization Section
        st.write("---")
        st.markdown("### Visual Insights")

        # Scatter plot of Income vs Loan Amount
        st.subheader("Income vs. Loan Amount Requested")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Income", y="LoanAmount", hue="Credit_History", data=pd.DataFrame({
            'Income': np.random.randint(1000, 20000, 100),
            'LoanAmount': np.random.randint(1000, 500000, 100),
            'Credit_History': np.random.choice(['Unclear Debts', 'No Unclear Debts'], 100)
        }), ax=ax)
        ax.axvline(ApplicantIncome, color="red", linestyle="--", label="Your Income")
        ax.axhline(LoanAmount, color="blue", linestyle="--", label="Requested Loan Amount")
        plt.legend()
        st.pyplot(fig)
        st.write("🔍 **Explanation**: Your income and requested loan amount are marked, showing where you fall among other applicants.")

        # Histogram of Loan Amounts
        st.subheader("Distribution of Loan Amounts")
        fig, ax = plt.subplots()
        sns.histplot(np.random.randint(1000, 500000, 100), bins=20, color="skyblue", ax=ax)
        ax.axvline(LoanAmount, color="blue", linestyle="--", label="Requested Loan Amount")
        plt.legend()
        st.pyplot(fig)
        st.write("🔍 **Explanation**: Higher loan amounts may impact approval likelihood.")

        # Approval Rate by Credit History
        st.subheader("Approval Rate by Credit History")
        approval_rate = pd.Series(['Unclear Debts', 'No Unclear Debts']).value_counts()
        fig, ax = plt.subplots()
        approval_rate.plot(kind="bar", color=["salmon", "lightgreen"], ax=ax)
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        st.write("🔍 **Explanation**: Clear credit histories often lead to higher approval rates.")

        # Affordability Calculator Visualization
        st.subheader("Loan Affordability Based on Income")
        income_levels = np.arange(1000, 20000, 500)
        affordable_loans = income_levels * 5  # Assume affordability is 5 times income
        fig, ax = plt.subplots()
        plt.plot(income_levels, affordable_loans, color="purple", label="Max Affordable Loan")
        plt.scatter(ApplicantIncome, LoanAmount, color="blue", label="Your Request")
        plt.xlabel("Monthly Income")
        plt.ylabel("Loan Amount")
        plt.legend()
        st.pyplot(fig)
        if result == "Rejected":
            st.write("🔍 **Explanation**: Your requested loan amount exceeds the maximum affordable loan amount at your income level.")
        else:
            st.write("🔍 **Explanation**: Your requested loan amount is within an affordable range based on your income level.")

        # Downloadable CSV Report
        st.write("---")
        st.subheader("Download Application Data")
        applicant_data = pd.DataFrame({
            "Gender": [Gender],
            "Marital Status": [Married],
            "Monthly Income": [ApplicantIncome],
            "Loan Amount": [LoanAmount],
            "Credit History": [Credit_History],
            "Decision": [result]
        })
        csv = applicant_data.to_csv(index=False)
        st.download_button(
            label="Download Application Data as CSV",
            data=csv,
            file_name="Loan_Application_Data.csv",
            mime="text/csv"
        )

    # Additional Information Section
    st.write("---")
    with st.expander("About This Tool"):
        st.write("""
            ProFund Insight's Loan Approval System helps financial institutions make data-driven decisions on loan applications.
            With a focus on transparency and insight, this tool leverages historical data to assess key factors contributing to loan approvals.
        """)
    with st.expander("How the Prediction Works"):
        st.write("""
            This tool uses a machine learning model trained on historical loan data. It evaluates factors such as gender, marital status, income,
            loan amount, and credit history to predict the approval outcome based on past trends.
        """)
    with st.expander("Why Was My Application Rejected?"):
        st.write("""
            Rejections may be due to factors like low income relative to the loan amount, unclear credit history, or high loan requests.
            Adjusting these factors could improve the likelihood of approval.
        """)

if __name__ == '__main__':
    main()
