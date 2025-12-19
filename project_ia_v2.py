### Document produit par le groupe de Harry Boisselot et Benoit Catry S5A2
### Etudiants à l'IUT NFC 2025-2026 

### Pour installer les modules nécessaires :
### pip install --upgrade pip langchain langchain-mistralai langchain-openai python-dotenv streamlit pandas numpy matplotlib seaborn xgboost statsmodels joblib

### Importation des modules

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import functions_project_ia
import ast

### Pour lancer le front : streamlit run project_ia_v2.py

######################### PARTIE GENERATIVE #########################

def promptIAGen(texte) -> str:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=token
    )

    prompt = ChatPromptTemplate.from_template("""
    SI ET SEULEMENT SI l'on te demande de prédire un nombre de patient pour une date donnée dans la question, répond STRICTEMENT la date sous format YYYY-MM-DD en str.
    SINON SI il n'y a pas de date MAIS qu'on te demande le nombre de patient, répond 'Veuillez préciser une date avec cette prédiction. §'. SI on te répond une date suite à ta réponse, répond STRICTEMENT la date sous format YYYY-MM-DD en str.
    ----
    SINON dans le cas où l'on te demande pas de prédire un nombre de patients, répond normalement à la question posée et ajoute un "§" à la fin de la réponse.
    ----
    Voici la question :
    ----
    {texte}""")

    llm = prompt | llm

    return llm.invoke({
        "texte": texte
    }).content

## Import du env
load_dotenv()
token = os.environ["MISTRAL_API_KEY"]

## Import du llm
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=token
)


######################### PARTIE PREDICTIVE #########################

df = pd.read_csv("services_weekly.csv")
BASE_DATE = pd.to_datetime("2024-01-01")
df["date"] = BASE_DATE + pd.to_timedelta(df["week"] - 1, unit="W")

# Série hebdomadaire globale = somme sur tous les services
serie = (
    df.groupby("date")["patients_admitted"]
      .sum()
      .sort_index()
)

# On impose une fréquence hebdomadaire (lundi)
serie = serie.asfreq("W-MON")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

df_features = serie.to_frame(name="y")

# Features calendaires
df_features["week"] = df_features.index.isocalendar().week.astype(int)
df_features["month"] = df_features.index.month
df_features["year"] = df_features.index.year

# Feature événement (flu vs none) agrégée à la semaine
weekly_event = (
    df.groupby("date")["event"]
      .agg(lambda x: "flu" if "flu" in x.values else "none")
      .reindex(df_features.index)
)

df_features["flu"] = (weekly_event == "flu").astype(int)

# Features retardées (lags) de la cible
for lag in [1, 2, 3, 4, 8, 12]:
    df_features[f"lag_{lag}"] = df_features["y"].shift(lag)

# Moyennes mobiles (sur les valeurs passées)
df_features["roll_mean_4"] = df_features["y"].shift(1).rolling(4).mean()
df_features["roll_std_4"]  = df_features["y"].shift(1).rolling(4).std()

# Nettoyage des NA (dus aux lags / rolling)
df_features = df_features.dropna()

# Matrices finales
X = df_features.drop(columns=["y"])
y = df_features["y"]

class_func = functions_project_ia.Functions_AI(np,pd,joblib,df,serie,df_features,X)


######################### FRONT #########################
rep = ""

st.title("Bienvenue sur le chatbot de l'hopital !") 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour. Comment puis-je vous aider ?", "df":""}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if (isinstance(message["df"], str)) :
            st.markdown(message["content"])
        else :
            st.dataframe(message["df"])

# Accept user input
if input := st.chat_input("Envoyer votre meilleur inspiration !"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input, "df":""})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        rep = promptIAGen(str(input))
        print(rep)
        if (rep[rep.__len__()-1] == "§") :
            rep = rep.strip("§")
            message_placeholder.markdown(rep)
            st.session_state.messages.append({"role": "assistant", "content": rep, "df":""})
        else :
            assistant_response = class_func.predict_weeks_from_date(rep)
            st.dataframe(assistant_response)
            st.session_state.messages.append({"role": "assistant", "df": assistant_response})
    # Add assistant response to chat history


# Test :
# Je veux prédire le nombre de patients pour la date du 02/02/2025 
# C'est bien, tu as pris en compte mes paramètres
# Je veux aussi prédire le nombre de lits qui sera disponible le 02/02/2022
# Génère-moi visuellement un schéma fictif simplifié d'un hôpital avec un nombre de lits de {nbPlaces}, 
# génère aussi visuellement un graphique du nombre de lits par rapport à {nbPatients} patients et {nbAssistants} assistants. 
# Le graphique a pour titre : 'Nombre de lits par rapport aux patients et par rapport aux assistants'; et a pour abscisse le nombre de lits et pour ordonnées le nombre d'assistants et de patients.