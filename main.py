import pandas as pd
import streamlit as st
import numpy as np
from functii import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

set_date = pd.read_csv("date_in/set de date.csv")
set2 = pd.read_csv("date_in/set de date 2.csv")
df = set_date.copy(deep=True)
variabile_numerice = ['Life_expectancy','CHE_GDP_ratio','GDP_per_capita']

st.markdown(
    """
    <style>
    .custom-title {
        color: #F39C12;
        font-size: 40px;
        text-align: center;
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .custom-header {
        color: #E38C02;
        font-size: 40px;
        text-align: center;
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="custom-title">Proiect Pachete Software</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="custom-header">Realizat de Jinga Ștefan și Matei Ștefan</h1>', unsafe_allow_html=True)




section = st.sidebar.radio("Navigați la:",
                           ["Vizualizare si analiza date", "Informatii date", "Grafice"])
# ----
# Vizualizare date
# ----
if section == "Vizualizare si analiza date":
    st.header("Setul de date")
    st.dataframe(set_date)

    exista_NaN = df.isna().any().any()
    # st.write(exista_NaN)
    if not exista_NaN:
        st.write("Setul de date nu contine valori lipsa.")
    else:
        st.write("Setul de date contine valori lipsa.")
        nan_replace_t(df)
        st.write("Valorile lipsa au fost inlocuite.")

    st.write("Dimensiunea setului de date este: " + str(df.shape))


    st.subheader("Statistici descriptive ale setului de date")
    st.write(df.describe())

    st.subheader("Corelatii intre variabile")
    st.pyplot(heatmap(df))

    st.write("Observam ca intre variabilele independente exista o corelatie mica, iar intre acestea si variabila dependenta (speranta de viata) exista legaturi moderate.")

    st.subheader("Detectare outlieri in coloanele numerice")
    for col in variabile_numerice:
        if col in df.columns:
            # Calculăm limitele și outlierii pentru coloana curentă
            lower_bound, upper_bound, outliers_df = find_outliers_iqr(df, col)
            st.write(f"Coloana: {col}")
            st.write(f"Limita inferioară: {lower_bound:.2f}, Limita superioară: {upper_bound:.2f}")
            st.write(f"Număr de outlieri: {len(outliers_df)}")
            st.write("-" * 50)

    st.subheader("Standardizare")
    st.write("Vom standardiza coloana 'Life_expectancy' utilizand modulul scikit-learn, apoi vom verifica media si deviatia standard pentru a ne asigura ca a fost standardizata corect.")
    df_standardized = df.copy(deep=True)
    standardizare_col(df_standardized, 'Life_expectancy')

    st.write("Media coloanei inainte de standardizare: ", df['Life_expectancy'].mean())
    st.write("Deviatia standard a coloanei inainte de standardizare: ", df['Life_expectancy'].std())
    st.write('-' * 50)
    st.write("Media coloanei dupa standardizare: ", df_standardized['Life_expectancy'].mean())
    st.write("Deviatia standard a coloanei dupa standardizare: ", df_standardized['Life_expectancy'].std())

    st.subheader("Rezultatele regresiei liniare")
    lr_mae,lr_mse,lr_r2 = calcul_regresie_liniara(df_standardized[variabile_numerice])
    st.write("In urma antrenarii unui model de regresie liniara pe setul de date cu coloana 'Life_expectancy' standardizata, am obtinut urmatoarele rezultate:")
    st.write("  Mean Absolute Error (MAE):", lr_mae)
    st.write("  Mean Squared Error (MSE):", lr_mse)
    st.write("  R2 Score:", lr_r2)


# ----
# Informatii date
# ----
elif section == "Informatii date":
    st.header("Informatii despre datele alese")
    st.markdown(r"""
        Progresele din domeniul tehnologic și medical din ultimele sute de ani stau la baza extinderii duratei vieții oamenilor, ceea ce a condus la transformarea fundamentală a sănătății publice și a calității vieții. Fiind un subiect de interes la nivel global, cât și pentru a înțelege mai bine relația dintre sănătatea populației și dezvoltare economică, ne-am propus să determinăm în ce măsura există o legătură între **speranța de viată** a oamenilor din diferite țări de factori economici si sociali locali.
        
        Astfel, am ales să analizăm această ipoteză pentru **30 de țări europene**, iar factorii pe care am hotărât să ii luăm în calcul sunt **cheltuielile medicale ca procent din PIB-ul țării** (notat `CHE_GDP_ratio` în setul de date) și **PIB-ul pe cap de locuitor**, exprimat în **euro pe locuitor** (notat `GDP_per_capita`).
        
        **Speranța de viață** este definită ca numărul estimat de ani rămași de trăit pentru o anumită persoană sau grup de persoane, fiind un concept matematic aplicat supraviețuirii.
    """
    )

elif section == "Grafice":
    st.subheader("Relatia intre cheltuielile medicale si speranta de viata")
    fig = px.scatter(set_date,
                     x="CHE_GDP_ratio",
                     y="Life_expectancy",
                     color="Country",
                     size="GDP_per_capita",
                     hover_name="Country",
                     title="Cheltuieli medicale vs. Speranță de viață",
                     labels={"CHE_GDP_ratio": "CHE % din PIB", "Life_expectancy": "Speranța de viață"})

    st.plotly_chart(fig)
    st.write("In acest scatter plot, observam relatia intre cheltuielile medicale ca procent din PIB si speranta de viata a tarii respective. Dimensiunea fiecarui punct din grafic este data de PIB-ul pe cap de locuitor (GDP_per_capita).")

    st.subheader("Boxplot-uri pentru variabile")
    for col in variabile_numerice:
        if col in df.columns:
            st.pyplot(boxplot_numeric(df, col))

    st.subheader("Pairplot pentru variabile")
    st.pyplot(plot_pairplot_numeric(df, variabile_numerice))