from idlelib import tooltip
import numpy as np
import requests
import pandas as pd
import requests, json
import streamlit as st
from PIL import Image
import pickle as pc
import streamlit.components.v1 as components
from altair.examples.scatter_with_minimap import detail
from matplotlib import pyplot as plt
from pandas.core import series
from sklearn.model_selection import train_test_split
from streamlit_echarts import st_echarts
import shap
shap.initjs()
from pyecharts import options #to show progress

#import dataframe and model

df_api = pd.read_csv('../model/df_api.csv') #data

model_pip = pc.load(open('../model/model_pip.pkl', 'rb')) #model

#st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>Prêt à dépenser : Demande de crédit</h1>", unsafe_allow_html=True)

#import the logo of startup
image = Image.open('../image/pad.png')
st.sidebar.image(image, width = 250)

seuil = 55
st.write("Seuil définit par l'institution :", seuil)
st.header('Score attribué au client :' )

#left side get id
def get_id():
    st.sidebar.header("Renseignements Client")
    id_curr = st.sidebar.text_input("ID client")
    Base = 'http://127.0.0.1:5000'
    #st.sidebar.button('Entrer')
    result = id_curr.title()
    reponse= requests.get(Base + "/score/" + result)

    if reponse.status_code != 200:
        return('Aucun client trouver')
    else:
        return(reponse)

response = get_id()
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

#exception : If the client doesn't exist
try:
    response.json()
except AttributeError:
    st.header('MESSAGE:')
    st.write('Le numéro client renseigné est inexistant')

#get proba
score = int(response.json()['proba']*100)

distance = score - seuil
if distance < -15 :
    couleur = 'tomato'
elif -15 < distance < 0 :
    couleur = 'darkorange'
elif distance == 0:
   couleur = 'cornflowerblue'
elif 0 < distance < 15:
    couleur = 'gold'
elif 15 < distance:
    couleur = 'seagreen'


options = {

        "series": [
            {
                "name": "Pressure",
                "type": "gauge",
                "axisLine": {
                    "lineStyle": {
                        "width": 10
                    },

                },
                "itemStyle" : {
                    'valueAnimation': 'true',
                    'color': couleur
                },
                "progress": {"show": "true", "width": 10},
                "detail": {"valueAnimation": "true"},
                "data": [{"value": score, "name": "Score"}]
            }

        ]
}


st_echarts(options=options, width="100%", key=0)


#Taking information about the client selected
df_client = df_api[df_api['SK_ID_CURR'] == response.json()["SK_ID_CURR"]]




cols = ['NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']
inf = []
for colonne in cols:
    df_client_filter = df_client.filter(regex=colonne)
    for col, item in df_client_filter.iteritems():
        if item.iat[0] == 1:
            inf.append(col.split('_')[-1])

info = pd.DataFrame(
    columns=["Niveau d'étude", 'Type de logement', 'Statut marital', 'Poste occupé', 'Poste occupé détaillé'])
info = pd.DataFrame([inf], columns=["Niveau d'étude", 'Type de logement', 'Statut marital',
                                    'Poste occupé','Poste occupé détaillé'])


info_client = pd.DataFrame(df_client[['Age',"Nombre d'enfant",'Nombre dans la famille',
                                    'Revenu Annuel', 'Montant Crédit','Montant Patrimoine immo']])

info_client["Ratio endettement"] = (info_client['Montant Crédit']/info_client['Revenu Annuel'])
index = info_client.index[0]
info_client.rename(index = {index:0}, inplace = True)
info_client = pd.concat([info_client,info], axis = 1)
info_client.rename(index={0:response.json()["SK_ID_CURR"]}, inplace=True)

st.subheader("Information client :")
st.dataframe(info_client)

# split data

X = df_api.iloc[:,2:]

Y = df_api['TARGET']

X_train, X_test, y_train , y_test = train_test_split(X, Y, test_size = 0.3)


#sampling for api deployement
# transorm data
X_test_std = model_pip.named_steps['process'].transform(X_test)
X_train_std = model_pip.named_steps['process'].transform(X_train)

### SHAP

st.header("Variables explicatives:")

#features names
feat_names = df_client.iloc[:, 3:].columns

#get index rows of clients
id_index =df_client['Unnamed: 0']

explainer = shap.LinearExplainer(model_pip.named_steps['model'], X_train_std, feature_names=feat_names)
shap_values = explainer(X_test_std)

st.set_option('deprecation.showPyplotGlobalUse', False)



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
st.subheader("Local :")
st_shap(shap.force_plot(explainer.expected_value,
                        explainer.shap_values(X_test_std[id_index].reshape(-1,1))[0],
                        feature_names=feat_names,
                        out_names="TARGET"
                        )
         )
st.subheader("Global :")
shap.summary_plot(explainer.shap_values(X_test_std),
                  features = X_test_std,
                  feature_names=feat_names)
st.pyplot()


### graph


st.subheader("Graphique :")

col1, col2, col3 = st.columns(3)
with col1:
    feat1 = st.selectbox("Select the Job", pd.unique(feat_names), key= 0)
with col2:
    feat2 = st.selectbox("Select the Job", pd.unique(feat_names), key=1)
with col3:
    feat3 = st.selectbox("Select the Job", pd.unique(feat_names), key=2)

import streamlit as st
import plotly.express as px
import seaborn as sns

#1er graph
yL = [0]
fig, ax = plt.subplots()
sns.histplot(data = df_api, x= feat1,ax= ax,  kde = True)
ax2 = ax.twinx()
sns.scatterplot(data = df_client, x=feat1,y= yL, ax=ax2, color='r')
st.pyplot()

#2eme graph
fig, ax = plt.subplots()
sns.histplot(data = df_api, x= feat2,ax= ax,  kde = True)
ax2 = ax.twinx()
sns.scatterplot(data = df_client, x=feat2,y= yL, ax=ax2, color='r')
st.pyplot()

#3eme graph analyse bivarié
sns.scatterplot(data= df_api, x=feat1, y= feat2, hue = feat3)
st.pyplot()

#git add .git
#git commit -m "ton message"
#git push origin main
###############
et inverser la proba car il sagitdune proba de defaut
calculer la proba pour defaut pour tous les clients et hue = proba pour avoir un dégradé de couleur
ajouter le client