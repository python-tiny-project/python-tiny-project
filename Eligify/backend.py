import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


FEATURE_NAMES = [
    "Income (log)",
    "Loan amount (log)",
    "Loan-to-income ratio",
    "EMI-to-income ratio",
    "CIBIL score",
    "Credit history",
    "Graduate",
    "Salaried",
    "Urban area",
    "Semi-urban area",
    "Dependents",
    "Age",
]


# ──────────────────────────────────────────────────────────────
#  Synthetic data generation + model training
# ──────────────────────────────────────────────────────────────
def train_model() -> tuple[LogisticRegression, StandardScaler]:
    """
    Generate ~1 000 synthetic loan records and fit a Logistic Regression.

    Returns
    -------
    model   : fitted LogisticRegression
    scaler  : fitted StandardScaler (must be used at inference time)
    """
    rng = np.random.default_rng(42)
    n = 1000

    # ── Raw feature simulation ────────────────────────────────
    income  = rng.integers(100_000,   2_000_001, n)
    loan    = rng.integers( 50_000,   5_000_001, n)
    dep     = rng.integers(0, 7, n)
    age     = rng.integers(18, 66, n)
    edu     = rng.integers(0, 2, n)          # 1 = graduate
    emp     = rng.integers(0, 3, n)          # 0 salaried, 1 self-emp, 2 business
    prop    = rng.integers(0, 3, n)          # 0 urban, 1 semi, 2 rural
    credit  = rng.choice([0, 1], n, p=[0.2, 0.8])
    cibil   = rng.integers(300, 901, n)
    emi     = rng.integers(0, 100_001, n)

    lti       = loan / income
    emi_ratio = emi / (income / 12)

    # ── Ground-truth logit (domain-knowledge weights) ─────────
    logit = (
        -3.0
        + 2.5 * credit
        + 0.004 * (cibil - 600)
        - 0.25 * lti
        - 1.2 * emi_ratio
        + 0.4 * edu
        + 0.3 * (emp == 0).astype(float)
        + 0.2 * (prop == 1).astype(float)
        - 0.15 * dep
        + 0.6 * np.log1p(income / 100_000)
        - 0.1 * np.abs(age - 38) / 10
        + rng.normal(0, 0.3, n)             # noise
    )
    prob = 1 / (1 + np.exp(-logit))
    y    = (prob > 0.5).astype(int)

    # ── Feature matrix ────────────────────────────────────────
    X = np.column_stack([
        np.log1p(income),                   # income_log
        np.log1p(loan),                     # loan_log
        lti,                                # loan-to-income
        emi_ratio,                          # emi burden
        (cibil - 600) / 300,               # cibil normalised
        credit,                             # credit history flag
        edu,                                # graduate flag
        (emp == 0).astype(float),           # salaried flag
        (prop == 0).astype(float),          # urban flag
        (prop == 1).astype(float),          # semi-urban flag
        dep,                                # dependents
        (age - 40) / 20,                    # age normalised
    ])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(X_sc, y)

    return model, scaler


# ── Module-level singletons (loaded once per process) ─────────
_model, _scaler = train_model()


# ──────────────────────────────────────────────────────────────
#  Public helpers
# ──────────────────────────────────────────────────────────────
def fmt_inr(v: int) -> str:
    """Format an integer rupee amount as e.g. ₹5L or ₹50,000."""
    if v >= 100_000:
        lakh = v / 100_000
        return f"₹{lakh:.1f}L" if lakh % 1 else f"₹{int(lakh)}L"
    return f"₹{v:,}"


def predict(
    income: int,
    loan: int,
    dep: int,
    age: int,
    edu: str,               # "Graduate" | "Not graduate"
    emp_type: str,          # "Salaried" | "Self-employed" | "Business owner"
    prop_area: str,         # "Urban" | "Semi-urban" | "Rural"
    credit: int,            # 1 = clean, 0 = has defaults
    cibil: int,
    emi: int,
) -> dict:
    """
    Run the logistic-regression model and return a result dict.

    Returns
    -------
    {
        "approved"   : bool,
        "score"      : int,          # 0-100
        "reasons"    : list[str],
        "max_loan"   : int,
        "rate"       : float,        # % p.a.
        "tenure"     : int,          # years
        "feature_df" : pd.DataFrame  # for the weights expander
    }
    """
    lti        = loan / income
    emi_ratio  = emi / (income / 12)
    is_sal     = 1 if emp_type == "Salaried" else 0
    is_grad    = 1 if edu == "Graduate" else 0
    is_urban   = 1 if prop_area == "Urban" else 0
    is_semi    = 1 if prop_area == "Semi-urban" else 0

    x = np.array([[
        np.log1p(income),
        np.log1p(loan),
        lti,
        emi_ratio,
        (cibil - 600) / 300,
        credit,
        is_grad,
        is_sal,
        is_urban,
        is_semi,
        dep,
        (age - 40) / 20,
    ]])
    x_sc     = _scaler.transform(x)
    prob     = _model.predict_proba(x_sc)[0][1]
    approved = prob >= 0.5

    # ── Rejection reasons ──────────────────────────────────────
    reasons = []
    if credit == 0:
        reasons.append("credit defaults on record")
    if cibil < 600:
        reasons.append(f"low CIBIL score ({cibil})")
    if lti > 6:
        reasons.append("loan-to-income ratio too high")
    if emi_ratio > 0.5:
        reasons.append("high existing EMI burden")
    if income < 200_000:
        reasons.append("low annual income")
    if dep >= 4:
        reasons.append(f"high number of dependents ({dep})")

    # ── Loan offer ─────────────────────────────────────────────
    multiplier = 5.0 if cibil >= 700 else 3.5
    max_loan   = (
        min(loan, int(income * multiplier / 100_000) * 100_000)
        if approved else 0
    )
    rate       = 8.5 if cibil >= 750 else (10.5 if cibil >= 650 else 12.5)
    max_tenure = 30 if age < 40 else (20 if age < 50 else 15)

    # ── Feature-weight dataframe ───────────────────────────────
    coefs = _scaler.scale_ * _model.coef_[0]


    return {
        "approved"   : approved,
        "score"      : round(prob * 100),
        "reasons"    : reasons,
        "max_loan"   : max_loan,
        "rate"       : rate,
        "tenure"     : max_tenure
    }

