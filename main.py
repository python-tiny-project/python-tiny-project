import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv("loan_data.csv")

# clean data
df = df.dropna()

df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(int)

# convert categorical to numeric
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Credit_History'] = df['Credit_History'].astype(int)
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# features + target
X = df[['ApplicantIncome', 'Dependents', 'Education', 'Credit_History']]
y = df['Loan_Status']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# save model
import pickle
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")
