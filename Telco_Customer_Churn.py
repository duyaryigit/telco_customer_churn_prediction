##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem: Developing a machine learning model to predict customers who are likely to churn from the company.
# Prior to building the model, it is expected to perform the necessary data analysis and feature engineering steps.

# Telco customer churn data contains information about a fictitious telecom company that provides home phone and Internet services to 7043 customers in California
# during the third quarter. It includes which customers left the service, stayed, or signed up for the service.

# 21 Variables 7043 Observations

# CustomerId : Customer İd
# Gender : Gender
# SeniorCitizen : Whether the customer is elderly or not (1, 0)
# Partner : Whether the customer has a partner (Yes, No) - Marital status.
# Dependents : Whether the customer has dependents (Yes, No) (Child, mother, father, grandparent)
# tenure : The number of months the customer has stayed with the company.
# PhoneService : Whether the customer has a phone service (Yes, No)
# MultipleLines : Whether the customer has multiple lines (Yes, No, No phone service)
# InternetService : The customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup : Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, No internet service)
# TechSupport : Whether the customer has technical support (Yes, No, No internet service)
# StreamingTV : Whether the customer has TV streaming (Yes, No, No internet service)
# StreamingMovies : Whether the customer has streaming movies (Yes, No, No internet service)
# Contract : The customer's contract term (Month-to-month, One year, Two years)
# PaperlessBilling : Whether the customer has paperless billing (Yes, No)
# PaymentMethod : The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card)
# MonthlyCharges : The amount collected monthly from the customer.
# TotalCharges : The total amount collected from the customer.
# Churn : Whether the customer used it (Yes or No) - Churned customers in the last month or quarter.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##################################
# CAPTURE OF NUMERICAL AND CATEGORICAL VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optional
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col,plot=True)


# Approximately half of the customers in our dataset are male, and the other half are female.
# About 50% of customers have a partner (married).
# Only 30% of the total customers have dependents.
# 90% of the customers have phone service.
# Among the 90% with phone service, 58% do not have multiple lines.
# There is a 21% segment of customers who do not have an internet service provider.
# Most customers have month-to-month contracts, with a similar number on 1-year and 2-year contracts.
# 60% of customers have paperless billing.
# Approximately 26% of customers churned from the platform last month.
# The dataset comprises 16% elderly customers, with the majority being young.

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# ANALYSIS OF NUMERICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


##################################
# ANALYSIS OF CATEGORICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# The churn percentage is almost equal between men and women.
# Customers with partners and dependents have lower churn rates.
# There is no significant difference in churn rates for PhoneService and MultipleLines.
# Churn rates are much higher for Fiber Optic Internet Services.
# Customers without services like OnlineSecurity, OnlineBackup, and TechSupport have higher churn rates.
# A higher percentage of customers with a monthly subscription churn compared to those with one or two-year contracts.
# Customers with paperless billing have a higher churn rate.
# Customers with Electronic Check as the PaymentMethod tend to churn more than other options.
# Churn percentage is higher for elderly customers.

##################################
# CORRELATION ANALYSIS
##################################

df[num_cols].corr()

# Correlation Matrix

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# It is observed that TotalCharges have a high correlation with monthly charges and tenure.

df[num_cols].corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# 1. method
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 2. method
df["TotalCharges"].fillna(df["MonthlyCharges"]*df["tenure"], inplace=True)

# 3. method
df["TotalCharges"].dropna(inplace=True)

#df.drop(df[df["TotalCharges"].isnull()].index, axis=0)

df.isnull().sum()

##################################
# OUTLIER ANALYSIS
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

##################################
# BASE MODEL
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345,force_row_wise=True,verbose=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.8031
# Auc: 0.8423
# Recall: 0.5404
# Precision: 0.6568
# F1: 0.5926

# ########## KNN ##########
# Accuracy: 0.7627
# Auc: 0.7463
# Recall: 0.4478
# Precision: 0.5681
# F1: 0.5003

# ########## CART ##########
# Accuracy: 0.728
# Auc: 0.6586
# Recall: 0.5077
# Precision: 0.4886
# F1: 0.4977

# ########## RF ##########
# Accuracy: 0.792
# Auc: 0.8252
# Recall: 0.4842
# Precision: 0.6448
# F1: 0.5529

# ########## XGB ##########
# Accuracy: 0.7886
# Auc: 0.827
# Recall: 0.5131
# Precision: 0.6263
# F1: 0.5631

# ########## LightGBM ##########
# Accuracy: 0.7982
# Auc: 0.8373
# Recall: 0.5281
# Precision: 0.6482
# F1: 0.5816

# ########## CatBoost ##########
# Accuracy: 0.797
# Auc: 0.8401
# Recall: 0.5051
# Precision: 0.6531
# F1: 0.5691



##################################
# FEATURE ENGINEERING
##################################

# Creating an annual categorical variable from the Tenure variable.
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Labeling customers with 1 or 2-year contracts as "Engaged."

df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# People who do not have any support, backup, or protection.

df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Customers with monthly contracts who are young.

df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# The total number of services a person has subscribed to.

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# People who have any streaming service.

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Does the person have automatic payment?

df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Average monthly payment.

df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase in current price compared to the average price.

df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Cost per service.

df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()
df.shape

##################################
# ENCODING
##################################

# Variable separation by their dtypes.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

##################################
# MODELLING
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345,verbose=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.7999
# Auc: 0.84
# Recall: 0.5003
# Precision: 0.6645
# F1: 0.5699
# ########## KNN ##########
# Accuracy: 0.7701
# Auc: 0.7535
# Recall: 0.4666
# Precision: 0.5851
# F1: 0.5182
# ########## CART ##########
# Accuracy: 0.7302
# Auc: 0.6602
# Recall: 0.5067
# Precision: 0.4922
# F1: 0.4992
# ########## RF ##########
# Accuracy: 0.7934
# Auc: 0.8269
# Recall: 0.5072
# Precision: 0.6404
# F1: 0.5659
# ########## XGB ##########
# Accuracy: 0.7907
# Auc: 0.8256
# Recall: 0.5153
# Precision: 0.6296
# F1: 0.5664
# ########## LightGBM ##########
# Accuracy: 0.794
# Auc: 0.8358
# Recall: 0.5222
# Precision: 0.6374
# F1: 0.5738
# ########## CatBoost ##########
# Accuracy: 0.7975
# Auc: 0.841
# Recall: 0.5179
# Precision: 0.6493
# F1: 0.576

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None], # Ağacın maksimum derinliği
             "max_features": [3, 7, "auto"], # En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
             "min_samples_split": [5, 8, 15], # Bir node'u bölmek için gereken minimum örnek sayısı
             "n_estimators": [100, 500]} # Ağaç sayısı

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_ # {'max_depth': 8, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 100}
#rf_final = rf_model.set_params(rf_best_grid.best_params_, random_state=17).fit(X, y)

rf_final = RandomForestClassifier(max_depth=8, max_features=7,min_samples_split=8,n_estimators=100,random_state=17).fit(X, y)
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1","recall","precision","roc_auc"])
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")



# ########## RF ##########
# Base Model
# Accuracy: 0.792
# Auc: 0.8252
# Recall: 0.4842
# Precision: 0.6448
# F1: 0.5529

# after Feature engineering
# Accuracy: 0.7934
# Auc: 0.8269
# Recall: 0.5072
# Precision: 0.6404
# F1: 0.5659

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 15],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_best_grid.best_params_
#{'colsample_bytree': 0.5,'learning_rate': 0.01,'max_depth': 5,'n_estimators': 1000}
xgboost_final = xgboost_model.set_params(max_depth=5, colsample_bytree=0.5,learning_rate=0.01,n_estimators=1000, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
#{'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}
lgbm_final = lgbm_model.set_params(colsample_bytree=0.5,learning_rate=0.01,n_estimators=500, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
#{'depth': 6, 'iterations': 500, 'learning_rate': 0.01}
catboost_final = catboost_model.set_params(depth=6, learning_rate=0.01,iterations=500, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)