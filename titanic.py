import joblib
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import random


@st.cache_data
def generate_random_value(x):
    return random.uniform(0, x)


st.set_page_config(
    page_title="Streamlit Titanic // Use case: Titanic",
    page_icon="🚢",
)

df = pd.read_csv("train.csv")

st.title("Titanic: Binary classification project")
st.sidebar.title("Table of contents")
pages = ["Exploration", "Data visualization", "Modelling", "Cache"]
page = st.sidebar.radio("Go to", pages)


# First page: Data exploration
if page == pages[0]:
    st.write("### Presentation of data")

    st.dataframe(df.head())

    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())


# Second page: Data visualization
if page == pages[1]:
    st.write("### Data visualization")

    fig = plt.figure()
    sns.countplot(x="Survived", data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Sex", data=df)
    plt.title("Distribution of the passengers' gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Pclass", data=df)
    plt.title("Distribution of the passengers' class")
    st.pyplot(fig)

    fig = sns.displot(x="Age", data=df)
    plt.title("Distribution of the passengers' age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Survived", hue="Sex", data=df)
    st.pyplot(fig)

    fig = sns.catplot(x="Pclass", y="Survived", data=df, kind="point")
    st.pyplot(fig)

    fig = sns.lmplot(x="Age", y="Survived", hue="Pclass", data=df)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)

if page == pages[2]:
    st.write("### Modelling")

    df_model = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Target variable
    # 0 = not survived, 1 = survived
    y = df_model["Survived"]

    # Categorical explanatory variables
    X_cat = df_model[["Pclass", "Sex", "Embarked"]]

    # Numerical explanatory variables
    X_num = df_model[["Age", "Fare", "SibSp", "Parch"]]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])

    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    X_cat_enc = pd.get_dummies(X_cat, columns=X_cat.columns)

    X = pd.concat([X_cat_enc, X_num], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    def prediction(classifier):
        if classifier == "Random Forest":
            clf = RandomForestClassifier()
        elif classifier == "SVC":
            clf = SVC()
        elif classifier == "Logistic Regression":
            clf = LogisticRegression()

        clf.fit(X_train, y_train)
        joblib.dump(clf, "model")
        pickle.dump(clf, open("model", "wb"))

        return clf

    def scores(clf, choice):
        if choice == "Accuracy":
            return clf.score(X_test, y_test)
        elif choice == "Confusion matrix":
            return confusion_matrix(y_test, clf.predict(X_test))

    choice = ["Random Forest", "SVC", "Logistic Regression"]
    option = st.selectbox("Choice of the model", choice)
    st.write("The chosen model is :", option)

    clf = prediction(option)

    display = st.radio("What do you want to show?", ("Accuracy", "Confusion matrix"))

    if display == "Accuracy":
        st.write(scores(clf, display))
    elif display == "Confusion matrix":
        st.dataframe(scores(clf, display))

if page == pages[3]:

    st.write("### Cache test")
    x = st.slider("Choose a number", 0, 100, 50)
    st.write(generate_random_value(x))

    a = generate_random_value(10)
    b = generate_random_value(20)
    st.write(a)
    st.write(b)
