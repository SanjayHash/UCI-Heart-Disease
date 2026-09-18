import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("./Heart_disease_cleveland_new.csv")

def summary(df):
    print(df.head())
    print(df.shape)
    print(df.info())

def target_distribution(df):
    sns.countplot(x='target',data=df)
    plt.title("Target Distribution")
    plt.show()

def numeric_plot():
    numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
    for col in numeric_cols:
        sns.boxplot(x='target',y=col,data=df)
        plt.title(f'{col} distribution across target ')
        plt.show()

# numeric_plot()

numeric_cols = df.select_dtypes(include='number').columns

def correlation_heatmap():
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    plt.title('Feature Correlation Map')
    plt.show()

def distribution():
    df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.show()


# Model Selection and Training

X= df.drop('target',axis=1)
Y = df['target']

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42,stratify=Y
    )

# Decision Tree model

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train,Y_train)

DT_prediction  = DT.predict(X_test)

print("Decision Tree",accuracy_score(Y_test,DT_prediction))

# Random Forest Model

RF = RandomForestClassifier(n_estimators=100,random_state=42)
RF.fit(X_train,Y_train)

RF_prediction = RF.predict(X_test)

print("Random Forest : ",accuracy_score(Y_test,RF_prediction))

# Checking for Overfitting

print("RF model train accuracy",accuracy_score(Y_train,RF.predict(X_train)))

RF_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

RF_tuned.fit(X_train,Y_train)

print("Rf tuned Train Acuuracy = ",accuracy_score(Y_train,RF_tuned.predict(X_train)))
print("RF tuned Test Accuracy  = ",accuracy_score(Y_test,RF_tuned.predict(X_test)))


# Cross Validation

cv_scores = cross_val_score(RF_tuned,X,Y,cv=5)
print("CV Scores :",cv_scores)
print("mean : ",cv_scores.mean())

# Feature Importance 
importances = pd.Series(RF_tuned.feature_importances_,index=X.columns)
print(importances.sort_values(ascending=False).head(8))

joblib.dump(RF_tuned,'Heart_Disease_RF_model.joblib')