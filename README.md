# Chatbot IA – Prédiction de l’occupation hospitalière

Projet réalisé dans le cadre du **semestre 5 du BUT Informatique**  
**IUT Nord Franche-Comté – Année universitaire 2025–2026**  
Auteurs : **Harry Boisselot, Benoît Catry (S5)**

---

## 1. Présentation du projet

Ce projet vise à développer une application permettant de **prédire le nombre de patients admis à l’hôpital** à partir de données historiques hebdomadaires.  
Il combine un **modèle d’apprentissage automatique** (XGBoost) avec une **interface conversationnelle intelligente**, accessible via une application web Streamlit.

L’utilisateur peut formuler ses demandes en langage naturel et obtenir automatiquement des **prévisions chiffrées sur plusieurs semaines**.

### Installation des dépendances

```bash
pip install --upgrade pip
pip install langchain langchain-mistralai langchain-openai python-dotenv \
            streamlit pandas numpy matplotlib seaborn \
            xgboost statsmodels joblib scikit-learn
```


###Configuration de l’API Mistral

```env
MISTRAL_API_KEY=VOTRE_CLE_API
```