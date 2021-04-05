import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# import folium

# from streamlit_folium import folium_static
st.title('Démo Pompiers')

#MAP
# st.write(pdk.Deck(map_style='mapbox://styles/cedric60128/ckn0g3q6u13pz17pbnp9vmn6n',
#                   initial_view_state=pdk.ViewState(latitude=51.495, longitude=-0.060, zoom=9.30, pitch=50)))

# st.write(pdk.Deck(map_style='mapbox://styles/cedric60128/ckn0g3q6u13pz17pbnp9vmn6n',
#                   initial_view_state=pdk.ViewState(latitude=51.45, longitude=-0.06,zoom=8.5, pith=50)))

# stations = pd.read_csv("Stations_GPS.csv")
# df = pd.read_csv('dfw.csv')
london_borough = ("london_boroughs.json")


# layer_station = pdk.Layer("ColumnLayer", 
#                           data=stations,
#                           id="stations",
#                           get_position="[lng, lat]",
#                           auto_highlight=True,
#                           elevation_scale=50,
#                           pickable=True,
#                           elevation_range=[0, 3000],
#                           extruded=True,
#                           coverage=1)

layer = pdk.Layer(
                    "HexagonLayer",
                    london_borough,
                    get_position="[lon, Lat]",
                    auto_highlight=True,
                    elevation_scale=50,
                    pickable=True,
                    elevation_range=[0, 3000],
                    extruded=True,
                    coverage=1,
                    id='id'
)




# Set the viewport location
view_state = pdk.ViewState(
    longitude=-0.06, latitude=51.42, zoom=8.7, min_zoom=5, max_zoom=15, pitch=40.5, bearing=-27.36)



# Combined all of it and render a viewport
r = pdk.Deck(
    map_style="mapbox://styles/cedric60128/ckn0g3q6u13pz17pbnp9vmn6n",
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>Elevation Value:</b> {elevationValue}", "style": {"color": "white"}},
)
# r.to_html("test.html", open_browser=True, notebook_display=False)

" Afficher l'API"
st.pydeck_chart(r)



def run_status():
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Percent complete {i+1}')
        bar.progress(i+1)
        time.sleep(0.001)
        st.empty()
run_status()


st.sidebar.header('Choix des paramètres')

# chargement des casernes des quartiers
df_cas_quart = pd.read_csv('df_cas_quart_2019.csv', index_col=0)
# tri par nom des quartiers
df_cas_quart = df_cas_quart.sort_values('ProperCase', ascending = True)

# chargement de la liste des quartiers pour la sidebar
liste_qu = []
for quartier in df_cas_quart.ProperCase.unique():
    liste_qu.append(quartier)


# chargement des quartiers et sous quartiers
df_qsq = pd.read_csv('df_qsq_2019.csv', index_col=0)
# tri par nom des quartiers et des sous quartiers
df_qsq = df_qsq.sort_values(['ProperCase','IncGeo_WardName'], ascending = True)

# chargement des incident group
df_ig = pd.read_csv('df_ig_2019.csv', index_col=0)


# chargement des codeId avec leur description
df_codeid = pd.read_csv('df_codeid_2019.csv', index_col=0)
# chargement des codeId avec leur description pour la sidebar
liste_delaycodeid = []
for description in df_codeid.DelayCode_Description.unique():
    liste_delaycodeid.append(description)


# chargement des servicespecialtype avec leur description
df_sst = pd.read_csv('df_sst_2019.csv', index_col=0)
# chargement des servicespecialtype avec leur description pour la sidebar
liste_sst = []
for sst in df_sst.SpecialServiceType.unique():
    liste_sst.append(sst)

# chargement des num_data_2020 prétraités
num_data = pd.read_csv('num_data_2020.csv', index_col=0)
num_data2 = num_data

# chargement des sous quartiers + incident group + weekday
df_sqigwd = pd.read_csv('df_sqigwd_2019.csv', index_col=0)

# chargement des quartiers + servicespecial + hourofcall
df_quart = pd.read_csv('df_quartiers_2019.csv', index_col=0)

# chargement des sous quartiers + codeId
df_sqid = pd.read_csv('df_sqid_2019.csv', index_col=0)

#sub2 = st.button('Entrainement du Modèle')
#if sub2:
#    from sklearn import preprocessing
#    from sklearn.model_selection import train_test_split
#    from sklearn.model_selection import cross_val_score
#    from sklearn.metrics import mean_squared_error
#    
#    scaler = preprocessing.StandardScaler().fit(num_data)
#    num_data[num_data.columns] = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns, index= num_data.index)
#    
#    st.write(num_data.head())
#    
#    dfw = num_data
#    target = dfw.FirstPumpArriving_AttendanceTime
#    feats = dfw.drop(['FirstPumpArriving_AttendanceTime'], axis=1)
#    
#    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2)
#    
#    from sklearn.ensemble import GradientBoostingRegressor
#    
#    gb_reg = GradientBoostingRegressor()
#    gb_reg.fit(X_train, y_train)
#    
#    st.write(gb_reg.score(X_train, y_train))


# définition de la fonction qui renvoie les choix faits dans la sidebar
def user_input():
    hourofcall = st.sidebar.slider('Heure d\'appel', 0, 23, 12)
    weekday = st.sidebar.select_slider('Jour de la semaine', options=['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    incidentgroup = st.sidebar.radio('Type d\'incident', ('Fire', 'False Alarm', 'Special Service'))
    sst_sel= st.sidebar.selectbox('Service Spécial', liste_sst)
    codeid_description= st.sidebar.selectbox('Conditions particulières', liste_delaycodeid)
    quartier= st.sidebar.selectbox('Quartier', liste_qu)
    if incidentgroup == 'Fire' or incidentgroup == 'False Alarm':
        sst_sel = 'None'   # quand incidentgroup est selectionné à Fire ou False Alarm alors on force le servicespecialtype à None
    # définition du dictionnaire résultat des choix de l'utilisateur
    data = {'hourofcall': hourofcall,
    'weekday': weekday,
    'incidentgroup': incidentgroup,
    'specialservicetype': sst_sel,
    'codeid_description': codeid_description,
    'quartier': quartier
    }
    # transformation des choix en DataFrame
    parametres = pd.DataFrame(data, index=[0])
    # la fonction renvoie le DataFrame des choix
    return parametres

# Appel de la fonction user_input pour récupération du DataFrame des choix
df = user_input()

# Affichage du DataFrame des choix
#st.write(df)

#liste_sq = []
#for sousquartier in df_qsq[df_qsq.ProperCase == df.quartier[0]].IncGeo_WardName.unique():
#    liste_sq.append(sousquartier)
#    df = user_input()


# chargement des sous quartiers correspondant au quartier choisi pour la sidebar
liste_sq = []
for sousquartier in df_qsq[df_qsq.ProperCase == df.quartier[0]].IncGeo_WardName.unique():
    liste_sq.append(sousquartier)
# valorisation des sous quartiers de la sidebar
sous_quartier= st.sidebar.selectbox('Sous-quartier', liste_sq)

#st.write(liste_sq)
num_quartier = df_cas_quart[df_cas_quart.ProperCase == df.quartier[0]].ProperCase_num
for nq in num_quartier:
    val_quartier = nq

num_sst = df_sst[df_sst.SpecialServiceType == df.specialservicetype[0]].SpecialServiceType_num
for sst in num_sst:
    val_sst = sst

num_ig = df_ig[df_ig.IncidentGroup == df.incidentgroup[0]].IncidentGroup_num
for ig in num_ig:
    val_ig = ig

num_codeid = df_codeid[df_codeid.DelayCode_Description == df.codeid_description[0]].DelayCodeId
for cid in num_codeid:
    val_codeid = cid


num_sous_quartier = df_qsq[(df_qsq.ProperCase == df.quartier[0])&(df_qsq.IncGeo_WardName == sous_quartier)].IncGeo_WardName_num
for nsq in num_sous_quartier:
    val_sous_quartier = nsq

num_heure = df.hourofcall[0]

#num_wd = df.weekday[0].replace(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],[0,1,2,3,4,5,6])
num_wd = df.weekday[0]
num_wd = num_wd.replace('Lundi','0')
num_wd = num_wd.replace('Mardi','1')
num_wd = num_wd.replace('Mercredi','2')
num_wd = num_wd.replace('Jeudi','3')
num_wd = num_wd.replace('Vendredi','4')
num_wd = num_wd.replace('Samedi','5')
num_wd = num_wd.replace('Dimanche','6')

num_wd = int(num_wd)

#st.write(val_quartier)
#st.write(val_sous_quartier)
#st.write(num_heure)
#st.write(num_wd)
#st.write(val_sst)
#st.write(val_ig)
#st.write(val_codeid)
#
#st.write(num_data[(num_data.ProperCase_num == num_quartier)&(num_data.IncGeo_WardName_num == num_sous_quartier)])

#num_data_filtre = num_data2[(num_data2.ProperCase_num == val_quartier)&(num_data2.IncGeo_WardName_num == val_sous_quartier)&(num_data2.HourOfCall == num_heure)&(num_data2.CalWeekDay == num_wd)&(num_data2.DelayCodeId == val_codeid)&(num_data2.IncidentGroup_num == val_ig)]

#st.write(num_data_filtre)

# Bouton de prédiction du temps d'arrivée
submit = st.button('Prédiction du temps d\'arrivée des pompiers')
if submit:
    # Construction du dictionnaire en vue de créer un dataframe avec les features retenues
    data_pred = {'HourOfCall': num_heure,
    'DelayCodeId': val_codeid,
    'ProperCase_num': val_quartier,
    'casernes_quartier': 0,
    'tmoyq': 0.0,
    'tmedq': 0.0,
    'CalWeekDay': num_wd,
    'IncGeo_WardName_num': val_sous_quartier,
    'casernes_sous_quartier': 0,
    'tmoysqigwd': 0.0,
    'tmedsqigwd': 0.0,
    'tstdsqigwd': 0.0,
    'tminsqigwd': 0.0,
    'tmaxsqigwd': 0.0,
    'tmoyqussho': 0.0,
    'tmedqussho': 0.0,
    'tstdqussho': 0.0,
    'tminqussho': 0.0,
    'tmaxqussho': 0.0,
    'IncidentGroup_num': val_ig,
    'SpecialServiceType_num': val_sst,
    'tmoyCodeId': 0.0,
    'tmedCodeId': 0.0,
    'tstdCodeId': 0.0,
    'tminCodeId': 0.0,
    'tmaxCodeId': 0.0,
    'FirstPumpArriving_AttendanceTime' : 302
    }
    # transformation en DataFrame
    num_data_pred = pd.DataFrame(data_pred, index=[0])
    # Récupération des statistiques correspondantes
    num_data_pred['casernes_quartier'] = df_cas_quart[df_cas_quart.ProperCase_num == val_quartier][['casernes_quartier']].values
    num_data_pred['tmoyq'] = df_cas_quart[df_cas_quart.ProperCase_num == val_quartier][['tmoyq']].values
    num_data_pred['tmedq'] = df_cas_quart[df_cas_quart.ProperCase_num == val_quartier][['tmedq']].values
    num_data_pred['casernes_sous_quartier'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['casernes_sous_quartier']].values
    num_data_pred['tmoysqigwd'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['tmoysqigwd']].values
    num_data_pred['tmedsqigwd'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['tmedsqigwd']].values
    num_data_pred['tstdsqigwd'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['tstdsqigwd']].values
    num_data_pred['tminsqigwd'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['tminsqigwd']].values
    num_data_pred['tmaxsqigwd'] = df_sqigwd[(df_sqigwd.IncGeo_WardName_num == val_sous_quartier)&(df_sqigwd.IncidentGroup == df.incidentgroup[0])&(df_sqigwd.CalWeekDay == num_wd)][['tmaxsqigwd']].values
    num_data_pred['tmoyqussho'] = df_quart[(df_quart.ProperCase == df.quartier[0])&(df_quart.SpecialServiceType == df.specialservicetype[0])&(df_quart.HourOfCall == num_heure)][['tmoyqussho']].values
    num_data_pred['tmedqussho'] = df_quart[(df_quart.ProperCase == df.quartier[0])&(df_quart.SpecialServiceType == df.specialservicetype[0])&(df_quart.HourOfCall == num_heure)][['tmedqussho']].values
    num_data_pred['tstdqussho'] = df_quart[(df_quart.ProperCase == df.quartier[0])&(df_quart.SpecialServiceType == df.specialservicetype[0])&(df_quart.HourOfCall == num_heure)][['tstdqussho']].values
    num_data_pred['tminqussho'] = df_quart[(df_quart.ProperCase == df.quartier[0])&(df_quart.SpecialServiceType == df.specialservicetype[0])&(df_quart.HourOfCall == num_heure)][['tminqussho']].values
    num_data_pred['tmaxqussho'] = df_quart[(df_quart.ProperCase == df.quartier[0])&(df_quart.SpecialServiceType == df.specialservicetype[0])&(df_quart.HourOfCall == num_heure)][['tmaxqussho']].values
    num_data_pred['tmoyCodeId'] = df_sqid[(df_sqid.IncGeo_WardName == sous_quartier)&(df_sqid.DelayCodeId == val_codeid)][['tmoyCodeId']].values
    num_data_pred['tmedCodeId'] = df_sqid[(df_sqid.IncGeo_WardName == sous_quartier)&(df_sqid.DelayCodeId == val_codeid)][['tmedCodeId']].values
    num_data_pred['tstdCodeId'] = df_sqid[(df_sqid.IncGeo_WardName == sous_quartier)&(df_sqid.DelayCodeId == val_codeid)][['tstdCodeId']].values
    num_data_pred['tminCodeId'] = df_sqid[(df_sqid.IncGeo_WardName == sous_quartier)&(df_sqid.DelayCodeId == val_codeid)][['tminCodeId']].values
    num_data_pred['tmaxCodeId'] = df_sqid[(df_sqid.IncGeo_WardName == sous_quartier)&(df_sqid.DelayCodeId == val_codeid)][['tmaxCodeId']].values
    # Affichage de la ligne des features
    st.write(num_data_pred)
    # Normalisation des num_data 2020 et du dataframe avec les features retenues
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    
    scaler = preprocessing.StandardScaler().fit(num_data)
    num_data[num_data.columns] = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns, index= num_data.index)
    num_data_pred[num_data_pred.columns] = pd.DataFrame(scaler.transform(num_data_pred), columns=num_data_pred.columns, index= num_data_pred.index)
    
    #st.write(num_data.head())
    #st.write(num_data_pred.head())
    
    # découpage feats et target de num_data
    dfw = num_data
    target = dfw.FirstPumpArriving_AttendanceTime
    feats = dfw.drop(['FirstPumpArriving_AttendanceTime'], axis=1)
    # Echantillons 2020
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2)
    
    #from sklearn.ensemble import GradientBoostingRegressor
    #
    #gb_reg = GradientBoostingRegressor()
    #gb_reg.fit(X_train, y_train)
    #
    #st.write(gb_reg.score(X_train, y_train))
    
    # création d'un regressor
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # suppression de la colonne target et stockage dans feats_demo
    feats_demo = num_data_pred.drop(['FirstPumpArriving_AttendanceTime'], axis=1)
    # Calcul de la prédiction avec le modèle entrainé
    pred_test = lr.predict(feats_demo)
    
    # reconstitution du DataFrame avec les features retenues
    num_X1 = feats_demo[['HourOfCall']]
    num_X2 = feats_demo[['DelayCodeId', 'ProperCase_num',
        'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
        'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
        'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
        'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
        'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
        'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
    num_X1 = num_X1.join(num_X2)
    # ajout de la colonne cible prédite
    num_X_pred = num_X1.assign(FirstPumpArriving_AttendTime_pred=pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred']).values)
    # Dénormalisation pour récupérer le temps prédit en secondes
    num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)
    # Affichage du DataFrame dénormalisé
    st.write(num_X_pred_inverse)
    # Récupération du temps estimé en secondes converti en entier
    tps_estimé = int(np.round(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred[0], 0))
    # Préparation du message de résultat
    success_message = 'Temps estimé d\'arrivée des pompiers : ' + str(tps_estimé) + ' secondes.'
    # Affichage du temps prédit
    st.success(success_message)
    