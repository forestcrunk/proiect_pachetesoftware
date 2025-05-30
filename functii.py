import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import geopandas as gpd
import statsmodels.api as sm

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

def map_life_expectancy_by_country(df):
    """
    Creează o hartă cu speranța de viață per țară folosind GeoPandas.
    Presupune că în df există o coloană 'Country' cu denumiri standardizate.
    """
    # Încărcăm geometria lumii
    world = gpd.read_file("date_in/map_units/ne_110m_admin_0_map_units.shp")

    # Facem merge pe țară
    df_country = df[["Country", "Life_expectancy"]]
    merged = world.merge(df_country, how="left", left_on="NAME", right_on="Country")

    # Cream harta
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    merged.plot(column='Life_expectancy', cmap='RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title('Speranța de viață per țară', fontdict={'fontsize': 15})
    ax.axis('off')
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

def kmeans_clusterizare(df, numeric_cols, n_clusters=3):
    """
    Aceasta functie aplica KMeans pe coloanele numerice si returneaza dataframe-ul cu clusterul asignat.
    """
    df_copy = df.copy(deep=True)

    # Scalare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_copy[numeric_cols])

    # KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_copy['Cluster'] = model.fit_predict(X_scaled)

    return df_copy, model

def regresie_multipla_statsmodels(t: pd.DataFrame):
    """
    Aceasta functie antrenează un model de regresie multiplă folosind pachetul statsmodels.
    """
    target = 'Life_expectancy'
    X = t.drop(columns=[target])
    y = t[target]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model
