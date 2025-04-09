import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def nan_replace_t(t:pd.DataFrame):
    """
    Aceasta functie analizeaza coloanele unui DataFrame
    si inlocuieste valorile lipsa in functie de tipul de data
    al coloanei.
    :param t:
    :return:
    """
    for v in t.columns:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t.fillna({v: t[v].mean()}, inplace=True)
            else:
                t.fillna({v: t[v].mode()[0]}, inplace=True)


def heatmap(t:pd.DataFrame):
    """
    Aceasta functie genereaza un heatmap pentru
    un DataFrame introdus ca parametru.
    :param t:
    :return:
    """
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(t.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    return fig

def find_outliers_iqr(df, col):
    """
    Funcție care calculează limitele inferioare și superioare folosind metoda IQR pentru o coloană numerică
    și returnează valorile limite și un DataFrame cu outlierii respectivi.
    """
    Q1 = df[col].quantile(0.25)       # Calculăm Quartila 1
    Q3 = df[col].quantile(0.75)       # Calculăm Quartila 3
    IQR = Q3 - Q1                   # Intervalul intercuartilic
    lower_bound = Q1 - 1.5 * IQR      # Limita inferioară
    upper_bound = Q3 + 1.5 * IQR      # Limita superioară
    outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]  # Selectăm valorile care ies din interval
    return lower_bound, upper_bound, outliers_df

def plot_pairplot_numeric(t:pd.DataFrame, numeric_cols):
    """
    Creează un pairplot pentru variabilele numerice.
    - diag_kind='kde' -> pe diagonală se afișează grafic de densitate
    - corner=True -> afișează doar jumătate din matrice (opțional)
    """
    fig = sns.pairplot(t[numeric_cols], diag_kind='kde')
    plt.suptitle("Pairplot pentru variabilele numerice", y=1.02)
    return fig

def boxplot_numeric(t:pd.DataFrame, col):
    """
    Aceasta functie creeaza un boxplot pentru o variabila numerica.
    """
    fig = plt.figure(figsize=(6, 4))
    sns.boxplot(x=t[col])
    plt.title(f"Boxplot pentru '{col}'")
    plt.xlabel(col)
    plt.tight_layout()
    return fig

def standardizare_col(t:pd.DataFrame, col):
    """
    Aceasta functie va standardiza coloana primita ca parametru utilizand
    StandardScaler din modulul scikit-learn.
    """
    scaler = StandardScaler()
    t[col] = scaler.fit_transform(t[[col]])

def calcul_regresie_liniara(t:pd.DataFrame):
    """
    Aceasta functie antreneaza si ruleaza un model de regresie liniara
    utilizand LinearRegression din modulul scikit-learn.
    :param t:
    :return:
    """
    target = 'Life_expectancy'

    # Eliminăm coloana țintă din setul de caracteristici
    X = t.drop([target], axis=1)
    y = t[target]

    # Împărțim datele în seturi de antrenare (80%) și test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inițializăm și antrenăm modelul de regresie liniară
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Facem predicții pe setul de test
    y_pred = lr_model.predict(X_test)

    # Valorile reale
    y_test_original = y_test

    # Calculăm metrici de evaluare pe scara originală
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    return mae,mse,r2
