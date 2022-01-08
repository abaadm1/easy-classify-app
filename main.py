import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

st.set_page_config(page_title='Easy Classify',layout='wide')

st.title("Easy Classification On The Go!")

st.write("""
Explore Your Dataset For Different Classification Models
""")

with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV/Excel file", type=['csv','xlsx'])

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest","Logistic Regression"))




def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM' or clf_name == 'Logistic Regression':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf
clf = get_classifier(classifier_name, params)

def build_model(df,clf):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last column as Y

    #Classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write('### Classifier')
    st.write(classifier_name)

    st.write('### Accuracy')
    st.write(np.round(acc*100,2))



    st.write('### Confusion Matrix')
    st.write(confusion_matrix(y_test, y_pred))

    st.write('### Classifcation Report')
    st.write(classification_report(y_test, y_pred))



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('### Glimpse Of Dataset')
    st.write(df.head(10))
    build_model(df,clf)
else:

    st.info('Awaiting for CSV file to be uploaded.')
    dataset_name = st.selectbox('Select Preloaded Dataset',('Iris', 'Breast Cancer', 'Wine'))
    if dataset_name == 'Iris':
        ex_data = datasets.load_iris()
    elif dataset_name == 'Wine':
        ex_data = datasets.load_wine()
    else:
        ex_data = datasets.load_breast_cancer()
    X = pd.DataFrame(ex_data.data, columns=ex_data.feature_names)
    Y = pd.Series(ex_data.target, name='Target')
    df = pd.concat([X, Y], axis=1)

    st.markdown('### Glimpse Of Dataset')
    st.write(df.head(10))

    build_model(df, clf)



