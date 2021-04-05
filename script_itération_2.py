# %matplotlib inline
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge, LassoCV
from datetime import datetime

t0 = datetime.today()


# Chargement du dataset des Incidents
dfi = pd.read_excel('LFB Incident data from January 2017.xlsx')


# Chargement du dataset des Mobilisations
dfm = pd.read_excel('LFB Mobilisation data from January 2017.xlsx')


# Fusion des datasets sur la colonne commune IncidentNumber (on ne garde que les incidents en commun avec inner)
fusion = dfi.merge(right = dfm, on = 'IncidentNumber', how = 'inner') 
fusion.info()

# chargement du dataset de travail à partir de la fusion 
# On ne retient que les lignes qui correspondent au premier camion (PumpOrder = 1) de la première équipe de pompiers arrivée sur les lieux
dfw = fusion[(fusion.FirstPumpArriving_DeployedFromStation == fusion.DeployedFromStation_Name)&(fusion.PumpOrder == 1)]

#On travaille sur l'année précédente pour calculer des statistiques
dfw = dfw[dfw.CalYear == 2019]

dfw.to_csv('dfwCalYear2019.csv')
dfw = pd.read_csv('dfwCalYear2019.csv')
dfw.head()
# # on souhaite ajouter au dataset le nombre de casernes du quartier de l'incident (ex. dans le quartier de Westminster il y a 8 casernes)
# # dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
# # on ajoute également les temps moyens  et médians par casernes
# df_cas_quart = pd.DataFrame(columns=('ProperCase', 'ProperCase_num' ,'casernes_quartier', 'tmoyq', 'tmedq'))

# for indice, quartier in enumerate(dfw.ProperCase.unique()):
#     #print(indice)
#     #print(quartier)
#     df_cas_quart.loc[indice,'ProperCase'] = quartier
#     df_cas_quart.loc[indice,'ProperCase_num'] = indice
#     nb_cas_pot = dfw[dfw.ProperCase == quartier].IncidentStationGround.nunique()
#     #print("casernes potentielles ",nb_cas_pot)
#     df_cas_quart.loc[indice,'casernes_quartier'] = nb_cas_pot
#     tps_moy = dfw[dfw.ProperCase == quartier].FirstPumpArriving_AttendanceTime.mean()
#     df_cas_quart.loc[indice,'tmoyq'] = tps_moy
#     tps_med = dfw[dfw.ProperCase == quartier].FirstPumpArriving_AttendanceTime.median()
#     df_cas_quart.loc[indice,'tmedq'] = tps_med
#     #print("temps moyen ",tps_moy)

# #display(df_cas_quart)
# # on sauvegarde ces statistiques dans un csv
# df_cas_quart.to_csv('df_cas_quart_2019.csv')
df_cas_quart = pd.read_csv('df_cas_quart_2019.csv')

# on ajoute une colonne jour de la semaine
dfw['CalWeekDay'] = pd.to_datetime(dfw['DateOfCall']).dt.weekday


# on souhaite ajouter au dataset le nombre de casernes du sous quartier de l'incident 
# dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
# on ajoute également les temps moyens, médians, écart type, min et max par sous quartier, type d'incident et jour de la semaine
# df_sqigwd = pd.DataFrame(columns=('IncGeo_WardName', 'IncGeo_WardName_num','casernes_sous_quartier','IncidentGroup','CalWeekDay','tmoysqigwd','tmedsqigwd','tstdsqigwd','tminsqigwd','tmaxsqigwd'))

# num_sq = 0
# indice = 0

# for sous_quartier in dfw.IncGeo_WardName.unique():
#     nb_cas_pot = dfw[dfw.IncGeo_WardName == sous_quartier].IncidentStationGround.nunique()
#     #print(indice)
#     for ig in dfw[dfw.IncGeo_WardName == sous_quartier].IncidentGroup.unique():
#         for k, wd in enumerate(dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)].CalWeekDay.unique()):
#             #print("casernes potentielles ",nb_cas_pot)
#             df_sqigwd.loc[indice+k,'IncGeo_WardName'] = sous_quartier
#             df_sqigwd.loc[indice+k,'IncGeo_WardName_num'] = num_sq
#             df_sqigwd.loc[indice+k,'casernes_sous_quartier'] = nb_cas_pot
#             df_sqigwd.loc[indice+k,'IncidentGroup'] = ig
#             df_sqigwd.loc[indice+k,'CalWeekDay'] = wd
#             tps_moy = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)&(dfw.CalWeekDay == wd)].FirstPumpArriving_AttendanceTime.mean()
#             df_sqigwd.loc[indice+k,'tmoysqigwd'] = tps_moy
#             tps_med = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)&(dfw.CalWeekDay == wd)].FirstPumpArriving_AttendanceTime.median()
#             df_sqigwd.loc[indice+k,'tmedsqigwd'] = tps_med
#             tps_std = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)&(dfw.CalWeekDay == wd)].FirstPumpArriving_AttendanceTime.std()
#             df_sqigwd.loc[indice+k,'tstdsqigwd'] = tps_std
#             tps_min = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)&(dfw.CalWeekDay == wd)].FirstPumpArriving_AttendanceTime.min()
#             df_sqigwd.loc[indice+k,'tminsqigwd'] = tps_min
#             tps_max = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.IncidentGroup == ig)&(dfw.CalWeekDay == wd)].FirstPumpArriving_AttendanceTime.max()
#             df_sqigwd.loc[indice+k,'tmaxsqigwd'] = tps_max
#             #tps_moy = dfw[dfw.ProperCase == quartier].FirstPumpArriving_AttendanceTime.mean()
#             #print("temps moyen ",tps_moy)
#         indice +=k+1
#     num_sq += 1
# df_sqigwd.to_csv('df_sqigwd_2019')
df_sqigwd = pd.read_csv('df_sqigwd_2019.csv')
#display(df_sqigwd.head(10))

# on remplace les écarts types null par 0
df_sqigwd.tstdsqigwd = df_sqigwd.tstdsqigwd.fillna(0, inplace=True)
# on sauvegarde ces statistiques dans un csv
# df_sqigwd.to_csv('df_sqigwd_2019.csv')

df_sqigwd = pd.read_csv('df_sqigwd_2019.csv')

# on souhaite ajouter au dataset les temps moyens, médians, écart type, min et max par quartier, heure de l'incident et service spécial
# dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
# df_quartiers = pd.DataFrame(columns=('ProperCase','SpecialServiceType', 'HourOfCall','tmoyqussho', 'tmedqussho', 'tstdqussho','tminqussho', 'tmaxqussho'))

# ind = 0

dfw.SpecialServiceType.fillna('None', inplace=True)

# for quartier in dfw.ProperCase.unique():
#     #nb_cas_pot = dfw[dfw.ProperCase == quartier].IncidentStationGround.nunique()
#     #print(quartier)
#     for heure in range(0,24):
#         for i, spst in enumerate(dfw[(dfw.ProperCase == quartier)&(dfw.HourOfCall == heure)].SpecialServiceType.unique()):
#             df_quartiers.loc[ind+i,'ProperCase'] = quartier
#             df_quartiers.loc[ind+i,'SpecialServiceType'] = spst
#             df_quartiers.loc[ind+i,'HourOfCall'] = heure
#             #print("casernes potentielles ",nb_cas_pot)
#             #df_quartiers.loc[indice,'casernes_quartier'] = nb_cas_pot
#             tps_moy = dfw[(dfw.ProperCase == quartier)&(dfw.SpecialServiceType == spst)&(dfw.HourOfCall == heure)].FirstPumpArriving_AttendanceTime.mean()
#             df_quartiers.loc[ind+i,'tmoyqussho'] = tps_moy
#             tps_med = dfw[(dfw.ProperCase == quartier)&(dfw.SpecialServiceType == spst)&(dfw.HourOfCall == heure)].FirstPumpArriving_AttendanceTime.median()
#             df_quartiers.loc[ind+i,'tmedqussho'] = tps_med
#             tps_std = dfw[(dfw.ProperCase == quartier)&(dfw.SpecialServiceType == spst)&(dfw.HourOfCall == heure)].FirstPumpArriving_AttendanceTime.std()
#             df_quartiers.loc[ind+i,'tstdqussho'] = tps_std
#             tps_min = dfw[(dfw.ProperCase == quartier)&(dfw.SpecialServiceType == spst)&(dfw.HourOfCall == heure)].FirstPumpArriving_AttendanceTime.min()
#             df_quartiers.loc[ind+i,'tminqussho'] = tps_min
#             tps_max = dfw[(dfw.ProperCase == quartier)&(dfw.SpecialServiceType == spst)&(dfw.HourOfCall == heure)].FirstPumpArriving_AttendanceTime.max()
#             df_quartiers.loc[ind+i,'tmaxqussho'] = tps_max
#             #print("temps moyen ",tps_moy)
#         ind +=i+1

# #display(df_quartiers)
# # on remplace les écarts types null par 0
# df_quartiers.tstdqussho.fillna(0, inplace=True)
# # on sauvegarde ces statistiques dans un csv
# df_quartiers.to_csv('df_quartiers_2019.csv')

df_quartiers = pd.read_csv('df_quartiers_2019.csv')

time = datetime.today()-t0

print("df_quartiers_2019",time)


# on souhaite ajouter au dataset les temps moyens, médians, écart type, min et max par sous quartier et delaycodeId
# dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
# df_sqid = pd.DataFrame(columns=('IncGeo_WardName','DelayCodeId','tmoyCodeId', 'tmedCodeId', 'tstdCodeId','tminCodeId', 'tmaxCodeId'))
# # on remplace les DelayCodeId null par 0
import numpy as np
dfw.DelayCodeId = dfw.DelayCodeId.replace(np.nan,0)
dfw.DelayCodeId.isna().sum()
# indice = 0

# for sous_quartier in dfw.IncGeo_WardName.unique():
#     for n, codeid in enumerate(dfw[(dfw.IncGeo_WardName == sous_quartier)].DelayCodeId.unique()):
#         #print("casernes potentielles ",nb_cas_pot)
#         df_sqid.loc[indice+n,'IncGeo_WardName'] = sous_quartier
#         df_sqid.loc[indice+n,'DelayCodeId'] = codeid
#         tps_moy = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.DelayCodeId == codeid)].FirstPumpArriving_AttendanceTime.mean()
#         df_sqid.loc[indice+n,'tmoyCodeId'] = tps_moy
#         tps_med = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.DelayCodeId == codeid)].FirstPumpArriving_AttendanceTime.median()
#         df_sqid.loc[indice+n,'tmedCodeId'] = tps_med
#         tps_std = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.DelayCodeId == codeid)].FirstPumpArriving_AttendanceTime.std()
#         df_sqid.loc[indice+n,'tstdCodeId'] = tps_std
#         tps_min = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.DelayCodeId == codeid)].FirstPumpArriving_AttendanceTime.min()
#         df_sqid.loc[indice+n,'tminCodeId'] = tps_min
#         tps_max = dfw[(dfw.IncGeo_WardName == sous_quartier)&(dfw.DelayCodeId == codeid)].FirstPumpArriving_AttendanceTime.max()
#         df_sqid.loc[indice+n,'tmaxCodeId'] = tps_max
#         #tps_moy = dfw[dfw.ProperCase == quartier].FirstPumpArriving_AttendanceTime.mean()
#         #print("temps moyen ",tps_moy)
#     indice +=n+1

# #display(df_sqid.head(10))
# # on remplace les écarts types null par 0
# df_sqid.tstdCodeId.fillna(0, inplace=True)
# # on sauvegarde ces statistiques dans un csv
# df_sqid.to_csv('df_sqid_2019.csv')

df_sqid = pd.read_csv('df_sqid_2019.csv')

time = datetime.today()-t0

print("df_sqid_2019",time)



# on numérote les différents types d'incident dans un DataFrame
df_ig = pd.DataFrame(columns=('IncidentGroup', 'IncidentGroup_num' ))


for j, ig in enumerate(dfw.IncidentGroup.unique()):
    df_ig.loc[j,'IncidentGroup'] = ig
    df_ig.loc[j,'IncidentGroup_num'] = j

#display(df_ig)

# on numérote les différents services spéciaux dans un DataFrame
df_sst = pd.DataFrame(columns=('SpecialServiceType', 'SpecialServiceType_num' ))


for m, sst in enumerate(dfw.SpecialServiceType.unique()):
    df_sst.loc[m,'SpecialServiceType'] = sst
    df_sst.loc[m,'SpecialServiceType_num'] = m
df_ig = df_ig.to_csv('df_ig_2019')
df_ig = pd.read_csv('df_ig_2019')
#display(df_sst)




# on repart de la fusion d'origine
dfw = fusion[(fusion.FirstPumpArriving_DeployedFromStation == fusion.DeployedFromStation_Name)&(fusion.PumpOrder == 1)]

# cette fois ci on charge les incidents de 2020
dfw = dfw[dfw.CalYear == 2020]
#dfw.shape


# on ajoute la colonne casernes_quartier, temps moyens et médians au dataframe de travail par un merge sur la colonne commune ProperCase
dfw = dfw.merge(right = df_cas_quart, on = 'ProperCase' , how = 'inner') 

#dfw.shape

# on ajoute le jour de la semaine
dfw['CalWeekDay'] = pd.to_datetime(dfw['DateOfCall']).dt.weekday

# on ajoute la colonne casernes_sous_quartier, temps moyens, médians, écart type, min, max au dataframe de travail par un merge sur les colonnes communes 
# IncGeo_WardName, IncidentGroup et CalWeekDay
dfw = dfw.merge(right = df_sqigwd, on=["IncGeo_WardName", "IncidentGroup","CalWeekDay"] , how = 'inner') 

#dfw.shape


# on remplace les services spéciaux nulls par 0
dfw.SpecialServiceType.fillna('None', inplace=True)
# on ajoute les colonnes temps moyens, médians, écart type, min, max au dataframe de travail par un merge sur les colonnes communes 
# ProperCase, SpecialServiceType, HourOfCall
dfw = dfw.merge(right = df_quartiers, on=["ProperCase", "SpecialServiceType","HourOfCall"] , how = 'inner') 

#dfw.shape

# on ajoute la colonne IncidentGroup_num au dataframe de travail par un merge sur la colonne commune IncidentGroup
dfw = dfw.merge(right = df_ig, on=["IncidentGroup"] , how = 'inner') 

#dfw.shape

# on ajoute la colonne SpecialServiceType_num au dataframe de travail par un merge sur la colonne commune SpecialServiceType
dfw = dfw.merge(right = df_sst, on=["SpecialServiceType"] , how = 'inner') 

dfw.shape



dfw.DelayCodeId.unique()
print(datetime.today()-t0,"secondes")
# on remplace les DelayCodeId à null par 0
# dfw.DelayCodeId.fillna(0, inplace=True)

dfw.DelayCodeId = dfw.DelayCodeId.replace(np.nan,0)


dfw.DelayCodeId.unique()


dfw.shape
# on ajoute les colonnes temps moyens, médians, écart type, min, max au dataframe de travail par un merge sur les colonnes communes 
# IncGeo_WardName, DelayCodeId
dfw = dfw.merge(right = df_sqid, on=["IncGeo_WardName", "DelayCodeId"] , how = 'inner') 
dfw.to_csv('dfw.csv')
#dfw.shape

# on supprime les colonnes devenues inutiles
dfw = dfw.drop(['IncidentNumber','DateOfCall','TimeOfCall','IncidentGroup','StopCodeDescription','SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier','Postcode_full', 
                'Postcode_district','UPRN', 'USRN', 'IncGeo_BoroughCode','IncGeo_BoroughName','ProperCase','IncGeo_WardCode',
                'IncGeo_WardName','IncGeo_WardNameNew','Easting_m','Northing_m','Latitude','Longitude','FRS','IncidentStationGround','FirstPumpArriving_DeployedFromStation',
                'SecondPumpArriving_AttendanceTime','SecondPumpArriving_DeployedFromStation',
                'NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount','PumpHoursRoundUp','Notional Cost (£)',
                'ResourceMobilisationId','Resource_Code','PerformanceReporting','DateAndTimeMobilised',
                'DateAndTimeMobile', 'TimeMobileTimezoneId', 'DateAndTimeArrived',
                'TimeArrivedTimezoneId', 'AttendanceTimeSeconds', 'DateAndTimeLeft',
                'TimeLeftTimezoneId', 'DateAndTimeReturned', 'TimeReturnedTimezoneId',
                'DeployedFromStation_Code', 'DeployedFromStation_Name','DeployedFromLocation','PumpOrder', 'PlusCode_Code','PlusCode_Description', 'DelayCode_Description']
               , axis=1)

# on transforme les nouvelles colonnes en numérique
dfw.casernes_quartier = dfw.casernes_quartier.astype('int64')
dfw.casernes_sous_quartier = dfw.casernes_sous_quartier.astype('int64')


# on transforme les nouvelles colonnes en numérique
dfw.HourOfCall = dfw.HourOfCall.astype('int64')
dfw.ProperCase_num = dfw.ProperCase_num.astype('int64')
dfw.CalWeekDay = dfw.CalWeekDay.astype('int64')
dfw.IncGeo_WardName_num = dfw.IncGeo_WardName_num.astype('int64')
dfw.IncidentGroup_num = dfw.IncidentGroup_num.astype('int64')
dfw.SpecialServiceType_num = dfw.SpecialServiceType_num.astype('int64')
# on transforme les nouvelles colonnes en numérique
dfw.tmoyq = dfw.tmoyq.astype('float64')
dfw.tmedq = dfw.tmedq.astype('float64')
dfw.tmoysqigwd = dfw.tmoysqigwd.astype('float64')
dfw.tmedsqigwd = dfw.tmedsqigwd.astype('float64')
dfw.tstdsqigwd = dfw.tstdsqigwd.astype('float64')
dfw.tminsqigwd = dfw.tminsqigwd.astype('float64')
dfw.tmaxsqigwd = dfw.tmaxsqigwd.astype('float64')
dfw.tmoyqussho = dfw.tmoyqussho.astype('float64')
dfw.tmedqussho = dfw.tmedqussho.astype('float64')
dfw.tstdqussho = dfw.tstdqussho.astype('float64')
dfw.tminqussho = dfw.tminqussho.astype('float64')
dfw.tmaxqussho = dfw.tmaxqussho.astype('float64')
dfw.tmoyCodeId = dfw.tmoyCodeId.astype('float64')
dfw.tmedCodeId = dfw.tmedCodeId.astype('float64')
dfw.tstdCodeId = dfw.tstdCodeId.astype('float64')
dfw.tminCodeId = dfw.tminCodeId.astype('float64')
dfw.tmaxCodeId = dfw.tmaxCodeId.astype('float64')
time = datetime.today()-t0

print("step",time)
dfw.head()
num_data = dfw.select_dtypes(include=['float64','int64'])
#print(len(num_data.columns),"colonnes numériques")
# contrôle du nombre de NA égal à 0
num_data.isna().sum()
# pour chaque colonne numérique

num_data = num_data.drop('Unnamed: 0_x',axis=1)
num_data = num_data.drop('Unnamed: 0_y',axis=1)
num_data = num_data.drop('Unnamed: 0',axis=1)

num_data.isna().sum()

for col in num_data:
    # on supprime la colonne initiale de dfw pour plus tard joindre toutes les colonnes numériques 
    # centrées réduites de num_data à dfw
    dfw = dfw.drop(col, axis=1)
dfw.head()
num_data.columns

# reconstruction de num_data en mettant la target en dernière colonne
num_d1 = num_data[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
num_d2 = num_data[['DelayCodeId', 'ProperCase_num',
       'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
       'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
       'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
num_d3 = num_data[['FirstPumpArriving_AttendanceTime']]


num_data = num_d1.join(num_d2.join(num_d3))

#num_data.head()

#num_data.shape

# Matrice de corrélation
cor = num_data.corr()

fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(cor, annot= True, ax= ax, cmap="coolwarm");





# Normalisation de num_data
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge, LassoCV


scaler = preprocessing.StandardScaler().fit(num_data)
num_data[num_data.columns] = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns, index= num_data.index)


# Récupération de num_data dans dfw
dfw = dfw.join(num_data)
#dfw.head(10)

# découpage target et features
target = dfw.FirstPumpArriving_AttendanceTime
feats = dfw.drop(['FirstPumpArriving_AttendanceTime'], axis=1)

# on fabrique des échantillons d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2)

# Création du modèle LinearRegression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# entrainement du modèle
lr.fit(X_train, y_train)


# R**2  + proche de 1 possible
print("Coefficient de détermination du modèle :", lr.score(X_train, y_train))

# affichage du score
lr.score(X_test,y_test)

# on stocke les prédictions du modèle pour X_test dans pred_test 
pred_test = lr.predict(X_test)

# reconstitution des dataframes normalisés X_test + y_test et X_test + y_pred avec la target en dernière colonnes
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
num_X2 = X_test[['DelayCodeId', 'ProperCase_num',
       'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
       'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
       'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
num_X1 = num_X1.join(num_X2)
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

num_X_test = num_X1.join(num_X3)



num_X_pred = num_X1.assign(FirstPumpArriving_AttendTime_pred=pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred']).values)


# dénormalisation de num_X_test et num_X_pred pour récupérer les temps réels et prédits
num_X_test_inverse = pd.DataFrame(scaler.inverse_transform(num_X_test), columns=num_X_test.columns, index= num_X_test.index)
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)


# concaténation du temps prédit à num_x_test_inverse pour avoir les 2 colonnes dans le même dataframe
num_X_test_inverse = num_X_test_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)



# affichage de quelques lignes pour les mettre dans un graphique
print(num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)].sort_values(['CalWeekDay','HourOfCall']).head(20))

# récupératiob des mêmes données dans res
res = num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)][['CalWeekDay','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalWeekDay','HourOfCall']).head(30)



# Affichage des comparaisons Temps réels / Temps prédits par le modèle
plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle LinearRegression')
plt.legend();





# Création du modèle SGDRegressor
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()
# entrainement du modèle
sgd_reg.fit(X_train, y_train) 


#print( "alpha sélectionné par c-v :" ,ridge_reg.alpha_)
print("score train :", sgd_reg.score(X_train, y_train))
print("score test :", sgd_reg.score(X_test, y_test))

sgd_reg_train = sgd_reg.predict(X_train)
sgd_reg_test = sgd_reg.predict(X_test)

print("mse train:", mean_squared_error(sgd_reg_train, y_train))
print("mse test:", mean_squared_error(sgd_reg_test, y_test))

# on stocke les prédictions du modèle pour X_test dans pred_test 
pred_test = sgd_reg.predict(X_test)

# reconstitution des dataframes normalisés X_test + y_test et X_test + y_pred avec la target en dernière colonnes
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
num_X2 = X_test[['DelayCodeId', 'ProperCase_num',
       'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
       'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
       'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
num_X1 = num_X1.join(num_X2)
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

num_X_test = num_X1.join(num_X3)

num_X_pred = num_X1.assign(FirstPumpArriving_AttendTime_pred=pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred']).values)

# dénormalisation de num_X_test et num_X_pred pour récupérer les temps réels et prédits
num_X_test_inverse = pd.DataFrame(scaler.inverse_transform(num_X_test), columns=num_X_test.columns, index= num_X_test.index)
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)

# concaténation du temps prédit à num_x_test_inverse pour avoir les 2 colonnes dans le même dataframe
num_X_test_inverse = num_X_test_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)


# affichage de quelques lignes pour les mettre dans un graphique
print(num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)].sort_values(['CalWeekDay','HourOfCall']).head(20))
# récupératiob des mêmes données dans res
res = num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)][['CalWeekDay','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalWeekDay','HourOfCall']).head(30)



# Affichage des comparaisons Temps réels / Temps prédits par le modèle
plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle SGDRegressor')
plt.legend();


# Création du modèle GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor()
# entrainement du modèle
gb_reg.fit(X_train, y_train) 


#print( "alpha sélectionné par c-v :" ,ridge_reg.alpha_)
print("score train :", gb_reg.score(X_train, y_train))
print("score test :", gb_reg.score(X_test, y_test))

##score train : 0.6156006822570951
##score test : 0.6110778763441289


gb_reg_train = gb_reg.predict(X_train)
gb_reg_test = gb_reg.predict(X_test)

print("mse train:", mean_squared_error(gb_reg_train, y_train))
print("mse test:", mean_squared_error(gb_reg_test, y_test))

##mse train: 0.3868851870979195
##mse test: 0.3788480879452526

# on stocke les prédictions du modèle pour X_test dans pred_test 
pred_test = gb_reg.predict(X_test)

# reconstitution des dataframes normalisés X_test + y_test et X_test + y_pred avec la target en dernière colonnes
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
num_X2 = X_test[['DelayCodeId', 'ProperCase_num',
       'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
       'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
       'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
num_X1 = num_X1.join(num_X2)
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

num_X_test = num_X1.join(num_X3)
num_X_test.shape

num_X_pred = num_X1.assign(FirstPumpArriving_AttendTime_pred=pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred']).values)

num_X_pred.shape


# dénormalisation de num_X_test et num_X_pred pour récupérer les temps réels et prédits
num_X_test_inverse = pd.DataFrame(scaler.inverse_transform(num_X_test), columns=num_X_test.columns, index= num_X_test.index)
num_X_test_inverse.shape
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)
num_X_pred_inverse.shape

# concaténation du temps prédit à num_x_test_inverse pour avoir les 2 colonnes dans le même dataframe
num_X_test_inverse = num_X_test_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)
num_X_test_inverse.shape
# affichage de quelques lignes pour les mettre dans un graphique
print(num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)].sort_values(['CalWeekDay','HourOfCall']).head(20))
# récupératiob des mêmes données dans res
res = num_X_test_inverse[(num_X_test_inverse.ProperCase_num == 11)&(num_X_test_inverse.IncidentGroup_num == 2)&(num_X_test_inverse.CalWeekDay == 0)][['CalWeekDay','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalWeekDay','HourOfCall']).head(30)
res.to_csv('data_full.csv')
# Affichage des comparaisons Temps réels / Temps prédits par le modèle
plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle GradientBoostingRegressor')
plt.legend();
time = datetime.today()-t0

print(time,"temps de chargement")
X_train.shape
