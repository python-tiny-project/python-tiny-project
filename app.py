import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏦 Loan Eligibility Checker")

# user inputs
income = st.number_input("Income")
dependents = st.number_input("Dependents", step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
credit = st.selectbox("Credit History", [1, 0])

# convert input
education = 1 if education == "Graduate" else 0

# prediction
if st.button("Check Eligibility"):
    data = np.array([[income, dependents, education, credit]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("✅ Loan Approved")
        st.write("Suggested Loan Amount:", income * 5)
    else:
        st.error("❌ Loan Rejected")

        # basic reason logic
        if income < 3000:
            st.write("Reason: Low Income")
        elif credit == 0:
            st.write("Reason: Poor Credit History")