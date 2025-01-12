import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import random
import time
from sklearn.utils import shuffle
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from tensorflow.python.ops.linalg.linalg_impl import sqrtm
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, RandomUnderSampler, AllKNN
from random_forest import RandomForest

def create_labels_key(labels):
    labels_key = {}
    for i in range(len(labels)):
        labels_key[labels[i]] = i
    return labels_key

def get_value_for_key(labels_key:dict, value):
    for k,v in labels_key.items():
        if v == value:
            return k
    return None

def show_statistics(test_data, pred_data, alg_name:str):
    accuracy = accuracy_score(test_data, pred_data)
    precision = precision_score(test_data, pred_data, average='weighted')
    recall = recall_score(test_data, pred_data, average='weighted')
    f1 = f1_score(test_data, pred_data, average='weighted')

    print("----------------------------------------------------")
    print(f"{alg_name} Accuracy training on : {accuracy:.4f}")
    print(f"{alg_name} Precision training on : {precision:.4f}")
    print(f"{alg_name} Recall training on : {recall:.4f}")
    print(f"{alg_name} F1-Score training on : {f1:.4f}")

def show_confusion_matrix(test_data, pred_data, alg_name:str, labels: dict):
    label_encoder = preprocessing.LabelEncoder()
    y_true = label_encoder.fit_transform([get_value_for_key(labels, el) for el in test_data])  # Encode test labels
    y_pred = label_encoder.fit_transform([get_value_for_key(labels, el) for el in pred_data]) # Your ensemble predictions
    cm = confusion_matrix(y_true, y_pred)

    # Create a Seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix of {alg_name}')
    plt.show()

def apply_scaler(train_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data

# undersampling algorithms

def apply_EditedNearestNeighbours(X_train, y_train):
    enn = EditedNearestNeighbours()
    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_NearMiss(X_train, y_train):
    nm1 = NearMiss(version=1)
    X_resampled, y_resampled = nm1.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_RandomUnderSampler(X_train, y_train):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_AllKNN(X_train, y_train):
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# classification algorithms 

def apply_GaussianNB(X_train, y_train, X_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    return y_pred

def apply_MLPClassifier(X_train, y_train, X_test):
    nnc=MLPClassifier(hidden_layer_sizes=(10), activation="relu", max_iter=10000)
    nnc.fit(X_train, y_train)
    y_pred = nnc.predict(X_test)
    return y_pred

def apply_KNeighborsClassifier(X_train, y_train, X_test):
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf3.fit(X_train, y_train)
    y_pred = clf3.predict(X_test)
    return y_pred

def apply_XGBClassifier(X_train, y_train, X_test):
    xgboost_model = XGBClassifier(n_estimators=500, random_state=42)
    xgboost_model.fit(X_train, y_train)
    y_pred = xgboost_model.predict(X_test)
    return y_pred

def apply_RandomForestClassifier(X_train, y_train, X_test):
    random_forest_model = RandomForestClassifier(n_estimators=500, random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    return y_pred

def apply_customRandomForestClassifier(X_train, y_train, X_test):
    # using 50 trees without a limitation for the max depth; random state as for the other classifiers
    random_forest_model = RandomForest(num_trees=50, random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred = random_forest_model.predict(X_test)
    return y_pred

# oversampling algorithms 

def apply_ADASYN(X, y):
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    return X_adasyn, y_adasyn

# study cases

def apply_clasification(X_train, X_test, y_train, y_test, labels, with_statistics = True, with_confusion_matrix = True):

    # y_pred = apply_GaussianNB(X_train, y_train, X_test)
    # if with_statistics: show_statistics(y_test, y_pred, "Naive Bayes Classifier")
    # if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "Naive Bayes Classifier", labels)
    #
    # y_pred = apply_MLPClassifier(X_train, y_train, X_test)
    # if with_statistics: show_statistics(y_test, y_pred, "MLP Classifier")
    # if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "MLP Classifier", labels)
    #
    # y_pred = apply_KNeighborsClassifier(X_train, y_train, X_test)
    # if with_statistics: show_statistics(y_test, y_pred, "KNeighbors Classifier")
    # if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "KNeighbors Classifier", labels)
    #
    # y_pred = apply_XGBClassifier(X_train, y_train, X_test)
    # if with_statistics: show_statistics(y_test, y_pred, "XGBoost Classifier")
    # if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "XGBoost Classifier", labels)
    #
    # y_pred = apply_RandomForestClassifier(X_train, y_train, X_test)
    # if with_statistics: show_statistics(y_test, y_pred, "Random Forest Classifier")
    # if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "Random Forest Classifier", labels)

    y_pred = apply_customRandomForestClassifier(X_train, y_train, X_test)
    if with_statistics: show_statistics(y_test, y_pred, "Custom Random Forest Classifier")
    if with_confusion_matrix: show_confusion_matrix(y_test, y_pred, "Custom Random Forest Classifier", labels)


def clasification_by_class():
    df = pd.read_csv("./Obfuscated-MalMem2022.csv")

    X = df.drop(columns=["Category","Class"])
    labels = list(df["Class"].unique())
    labels_key = create_labels_key(labels)
    df['Class'] = df['Class'].replace(labels_key)
    y = df["Class"]

    print("Classification by Class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)
    X_train, X_test = apply_scaler(X_train, X_test)

    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)

def clasification_by_category():
    df = pd.read_csv("./Obfuscated-MalMem2022.csv")
    
    df['Category'] = [label.split('-')[0] for label in df['Category']]
    labels = list(df["Category"].unique())
    labels_key = create_labels_key(labels)
    df['Category'] = df['Category'].replace(labels_key)
    X = df.drop(columns=["Category","Class"])
    y = df['Category']

    print("Classification by Category")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)
    X_train, X_test = apply_scaler(X_train, X_test)

    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)


def clasification_by_category_with_undersampling():
    df = pd.read_csv("./Obfuscated-MalMem2022.csv")
    
    df['Category'] = [label.split('-')[0] for label in df['Category']]
    labels = list(df["Category"].unique())
    labels_key = create_labels_key(labels)
    df['Category'] = df['Category'].replace(labels_key)
    X = df.drop(columns=["Category","Class"])
    y = df['Category']

    print("Classification by Category with undersampling")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)
    X_train, X_test = apply_scaler(X_train, X_test)

    print("For EditedNearestNeighbours")
    X_train, y_train = apply_EditedNearestNeighbours(X_train,y_train)
    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)

    print("For NearMiss")
    X_train, y_train = apply_NearMiss(X_train,y_train)
    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)

    print("For RandomUnderSampler")
    X_train, y_train = apply_RandomUnderSampler(X_train,y_train)
    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)

    print("For AllKNN")
    X_train, y_train = apply_AllKNN(X_train,y_train)
    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)

def clasification_by_category_with_oversampling_only_malware():
    df = pd.read_csv("./Obfuscated-MalMem2022.csv")
    
    df['Category'] = [label.split('-')[0] for label in df['Category']]
    label_bening = "Benign"
    labels_malware = [el for el in list(df["Category"].unique()) if el != label_bening]

    adasyn_df_categories = []
    for label_malware in labels_malware:
        df_b = df[df["Category"] == label_bening]
        df_m = df[df["Category"] == label_malware]
        combined_df = pd.concat([df_b, df_m], ignore_index=True)

        labels = list(combined_df["Category"].unique())
        labels_key = create_labels_key(labels)
        combined_df['Category'] = combined_df['Category'].replace(labels_key)
        X_categ = combined_df.drop(columns=["Category","Class"])
        y_categ = combined_df['Category']

        X_adasyn, y_adasyn = apply_ADASYN(X_categ, y_categ)
        
        adasyn_df_categ = pd.DataFrame(X_adasyn, columns=X_categ.columns)
        adasyn_df_categ['Category'] = y_adasyn
        adasyn_df_categ = adasyn_df_categ[adasyn_df_categ['Category'] == labels_key[label_malware]]
        adasyn_df_categ['Category'] = adasyn_df_categ['Category'].replace({labels_key[label_malware]:label_malware})
        adasyn_df_categories.append(adasyn_df_categ)

    combined_df = pd.concat(adasyn_df_categories, ignore_index=True)
    labels = list(combined_df["Category"].unique())
    labels_key = create_labels_key(labels)
    combined_df['Category'] = combined_df['Category'].replace(labels_key)
    X = combined_df.drop(columns=["Category"])
    y = combined_df['Category']

    print("Classification by Category with oversampling only malware data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state=42)
    X_train, X_test = apply_scaler(X_train, X_test)

    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)


def clasification_by_category_with_oversampling_with_all_data():
    df = pd.read_csv("./Obfuscated-MalMem2022.csv")
    
    df['Category'] = [label.split('-')[0] for label in df['Category']]
    label_bening = "Benign"
    labels_malware = [el for el in list(df["Category"].unique()) if el != label_bening]

    adasyn_df_categories = []
    for label_malware in labels_malware:
        df_b = df[df["Category"] == label_bening]
        df_m = df[df["Category"] == label_malware]
        combined_df = pd.concat([df_b, df_m], ignore_index=True)

        labels = list(combined_df["Category"].unique())
        labels_key = create_labels_key(labels)
        combined_df['Category'] = combined_df['Category'].replace(labels_key)
        X_categ = combined_df.drop(columns=["Category","Class"])
        y_categ = combined_df['Category']

        X_adasyn, y_adasyn = apply_ADASYN(X_categ, y_categ)
        
        adasyn_df_categ = pd.DataFrame(X_adasyn, columns=X_categ.columns)
        adasyn_df_categ['Category'] = y_adasyn
        adasyn_df_categ = adasyn_df_categ[adasyn_df_categ['Category'] == labels_key[label_malware]]
        adasyn_df_categ['Category'] = adasyn_df_categ['Category'].replace({labels_key[label_malware]:label_malware})
        adasyn_df_categories.append(adasyn_df_categ)

    benign_df = df[df["Category"] == label_bening]
    benign_df = benign_df.drop(["Class"],axis=1)
    adasyn_df_categories.append(benign_df)
    
    combined_df = pd.concat(adasyn_df_categories, ignore_index=True)
    labels = list(combined_df["Category"].unique())
    labels_key = create_labels_key(labels)
    combined_df['Category'] = combined_df['Category'].replace(labels_key)
    X = combined_df.drop(columns=["Category"])
    y = combined_df['Category']

    print("Classification by Category with oversampling only malware data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state=42)
    X_train, X_test = apply_scaler(X_train, X_test)

    apply_clasification(X_train, X_test, y_train, y_test, labels_key, True, True)


if __name__ == "__main__":
    # clasification_by_class()
    # clasification_by_category()
    # clasification_by_category_with_undersampling()
    # clasification_by_category_with_oversampling_only_malware()
    clasification_by_category_with_oversampling_with_all_data()
