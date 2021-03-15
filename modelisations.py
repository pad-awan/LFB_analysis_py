
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LassoCV

# Chargement du dataset des Incidents
dfi = pd.read_excel('LFB Incident data from January 2017.xlsx')

# Chargement du dataset des Mobilisations
dfm = pd.read_excel('LFB Mobilisation data from January 2017.xlsx')

# Fusion des datasets sur la colonne commune IncidentNumber (on ne garde que les incidents en commun avec inner)
fusion = dfi.merge(right = dfm, on = 'IncidentNumber', how = 'inner') 

# chargement du dataset de travail à partir de la fusion 
# On ne retient que les lignes qui correspondent au premier camion (PumpOrder = 1) de la première équipe de pompiers arrivée sur les lieux
dfw = fusion[(fusion.FirstPumpArriving_DeployedFromStation == fusion.DeployedFromStation_Name)&(fusion.PumpOrder == 1)]

# on souhaite ajouter au dataset le nombre de casernes du quartier de l'incident (ex. dans le quartier de Westminster il y a 8 casernes)
# dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
df_quartiers = pd.DataFrame(columns=('ProperCase','casernes_quartier'))

for indice, quartier in enumerate(dfw.ProperCase.unique()):
    #pour chaque quartier on récupère son nom
    df_quartiers.loc[indice,'ProperCase'] = quartier
    #on calcule le nombre de casernes différentes du quartier
    nb_cas_pot = dfw[dfw.ProperCase == quartier].IncidentStationGround.nunique()
    #on l'enregistre dans le nouveau dataframe
    df_quartiers.loc[indice,'casernes_quartier'] = nb_cas_pot
    
# on ajoute la colonne casernes_quartier au dataframe de travail par un merge sur la colonne commune ProperCase
dfw = dfw.merge(right = df_quartiers, on = 'ProperCase' , how = 'inner') 


# on souhaite ajouter au dataset le nombre de casernes du sous quartier de l'incident (ex. dans le sous quartier de Barkingside il y a 2 casernes)
# dans le but d'enrichir notre dataset afin d'aider les modèles de prédiction
df_sous_quartiers = pd.DataFrame(columns=('IncGeo_WardName','casernes_sous_quartier'))

for indice, sous_quartier in enumerate(dfw.IncGeo_WardName.unique()):
    #pour chaque sous quartier on récupère son nom
    df_sous_quartiers.loc[indice,'IncGeo_WardName'] = sous_quartier
    #on calcule le nombre de casernes différentes du sous quartier
    nb_cas_pot = dfw[dfw.IncGeo_WardName == sous_quartier].IncidentStationGround.nunique()
    #on l'enregistre dans le nouveau dataframe
    df_sous_quartiers.loc[indice,'casernes_sous_quartier'] = nb_cas_pot

# on ajoute la colonne casernes_sous_quartier au dataframe de travail par un merge sur la colonne commune IncGeo_WardName
dfw = dfw.merge(right = df_sous_quartiers, on = 'IncGeo_WardName' , how = 'inner') 


# on supprime les colonnes inutiles
dfw = dfw.drop(['IncidentNumber','TimeOfCall','PropertyType','Postcode_full', 'Postcode_district','UPRN', 'USRN', 'IncGeo_BoroughCode','IncGeo_BoroughName',
                'IncGeo_WardCode',
                'IncGeo_WardName','IncGeo_WardNameNew','Easting_m','Northing_m','Latitude','Longitude','FRS',
                'SecondPumpArriving_AttendanceTime','SecondPumpArriving_DeployedFromStation',
                'NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount','PumpHoursRoundUp','Notional Cost (£)',
                'ResourceMobilisationId','Resource_Code','PerformanceReporting','DateAndTimeMobilised',
                'DateAndTimeMobile', 'TimeMobileTimezoneId', 'DateAndTimeArrived',
                'TimeArrivedTimezoneId', 'AttendanceTimeSeconds', 'DateAndTimeLeft',
                'TimeLeftTimezoneId', 'DateAndTimeReturned', 'TimeReturnedTimezoneId',
                'DeployedFromStation_Code', 'DeployedFromStation_Name','PumpOrder', 'PlusCode_Code','PlusCode_Description', 'DelayCode_Description']
               , axis=1)


# on transforme le type des variables numériques object en entiers
dfw.casernes_quartier = dfw.casernes_quartier.astype('int64')
dfw.casernes_sous_quartier = dfw.casernes_sous_quartier.astype('int64')

# on rajoute le jour de la semaine, le jour et le mois de l'appel avant de supprimer les variables de type datetime
dfw['CalWeekDay'] = pd.to_datetime(dfw['DateOfCall']).dt.weekday
dfw['CalDay'] = pd.to_datetime(dfw['DateOfCall']).dt.day
dfw['CalMonth'] = pd.to_datetime(dfw['DateOfCall']).dt.month

# La variable DelayCodeId est une catégorielle donc on change son type en chaine de caractères 
dfw.DelayCodeId = dfw.DelayCodeId.astype('str')


# on récupère dans num_data toutes les colonnes numériques
num_data = dfw.select_dtypes(include=['float64','int64'])
# print(len(num_data.columns),"colonnes numériques")
# pour chaque colonne numérique
for col in num_data:
    # on la supprime du dataset de travail
    dfw = dfw.drop(col, axis=1)

# on réorganise l'ordre des colonnes de num_data pour avoir la variable cible en dernier    
num_d1 = num_data[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
num_d2 = num_data[['casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_d3 = num_data[['FirstPumpArriving_AttendanceTime']]

num_data = num_d1.join(num_d2.join(num_d3))


# on affiche la heatmap des corrélations
cor = num_data.corr()

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(cor, annot= True, ax= ax, cmap="coolwarm");


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LassoCV

# on normalise les variables numériques avec StandardScaler
scaler = preprocessing.StandardScaler().fit(num_data)
num_data[num_data.columns] = pd.DataFrame(scaler.transform(num_data), columns=num_data.columns, index= num_data.index)

# on rajoute à dfw les colonnes numériques normalisées
dfw = dfw.join(num_data)


# on récupère dans cat_data toutes les colonnes catégorielles
cat_data = dfw.select_dtypes(include=['O'])
#print(len(cat_data.columns),"colonnes catégorielles")


# pour chaque colonne catégorielles
for col in cat_data:
    # on crée des variables indicatrices autant que de valeurs différentes et on nomme les nouvelles colonnes 
    # avec comme préfixe le nom de la colonne d'origine
    dfw = dfw.join(pd.get_dummies(dfw[col], prefix=col))
    # on supprime la colonne initiale de dfw pour plus tard joindre toutes les variables indicatrices de cat_data à dfw
    dfw = dfw.drop(col, axis=1)
    
# on sépare la cible dans target et les autres colonnes de dfw dans feats (on en profite pour supprimer la colonne DateOfCall)   
target = dfw.FirstPumpArriving_AttendanceTime
feats = dfw.drop(['DateOfCall',  'FirstPumpArriving_AttendanceTime'], axis=1)

# on découpe les données de dfw en échantillons d'entrainement (80%) et de test final (20%)
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2)


####################### MODELISATIONS  ==> LinearRegression

from sklearn.linear_model import LinearRegression

#création du modèle
lr = LinearRegression()
lr.fit(X_train, y_train)


# R**2 entrainement + proche de 1 possible
print("Coefficient de détermination du modèle :", lr.score(X_train, y_train))

##Coefficient de détermination du modèle : 0.5839818857279384

# R**2 test + proche de 1 possible
lr.score(X_test,y_test)

##0.5786472154140925


# on stocke les prédictions du modèle pour X_test dans pred_test 
pred_test = lr.predict(X_test)
#plt.scatter(pred_test, y_test)
#plt.plot((y_test.min(),y_test.max()), (y_test.min(),y_test.max()))

# Reconstitution de l'échantillon de test avec la colonne cible dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

# num_X_test avec la cible réelle FirstPumpArriving_AttendanceTime
num_X_test = num_X1.join(num_X3)

# Reconstitution de l'échantillon de test avec la colonne prédite dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X4 = pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred'])

# num_X_pred avec la cible prédite FirstPumpArriving_AttendTime_pred
num_X_pred = num_X1.join(num_X4)

# on dénormalise les deux dataframes
num_X_test_inverse = pd.DataFrame(scaler.inverse_transform(num_X_test), columns=num_X_test.columns, index= num_X_test.index)
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)

# on rajoute la colonne cible prédite à num_X_test_inverse
num_X_test_inverse = num_X_test_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)


# On extrait une journée du test avec la cible et la prédiction
res = num_X_test_inverse[(-pd.isnull(num_X_test_inverse.FirstPumpArriving_AttendTime_pred))&(num_X_test_inverse.CalYear == 2020)&(num_X_test_inverse.CalMonth == 11)&(num_X_test_inverse.CalDay == 14)][['CalDay','CalMonth','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalDay','HourOfCall']).head(30)

# on affiche les écarts dans un graphique pour cette journée
plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle LinearRegression')
plt.legend();



####################### MODELISATIONS  ==> RidgeCV


from sklearn.linear_model import RidgeCV

#création du modèle
ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
ridge_reg.fit(X_train, y_train) 

print( "alpha sélectionné par c-v :" ,ridge_reg.alpha_)
print("score train :", ridge_reg.score(X_train, y_train))
print("score test :", ridge_reg.score(X_test, y_test))

##alpha sélectionné par c-v : 10.0
##score train : 0.5839438555837259
##score test : 0.5787123523986961


ridge_pred_train = ridge_reg.predict(X_train)
ridge_pred_test = ridge_reg.predict(X_test)

print("mse train:", mean_squared_error(ridge_pred_train, y_train))
print("mse test:", mean_squared_error(ridge_pred_test, y_test))


##mse train: 0.4150893945887835
##mse test: 0.4252015806664894

# on stocke les prédictions du modèle pour X_test dans pred_test 
pred_test = ridge_reg.predict(X_test)
# plt.scatter(pred_test, y_test)
# plt.plot((y_test.min(),y_test.max()), (y_test.min(),y_test.max()))


# Reconstitution de l'échantillon de test avec la colonne cible dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

# num_X_test avec la cible réelle FirstPumpArriving_AttendanceTime
num_X = num_X1.join(num_X3)

# Reconstitution de l'échantillon de test avec la colonne prédite dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X4 = pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred'])

# num_X_pred avec la cible prédite FirstPumpArriving_AttendTime_pred
num_X_pred = num_X1.join(num_X4)


# on dénormalise les deux dataframes
num_X_inverse = pd.DataFrame(scaler.inverse_transform(num_X), columns=num_X.columns, index= num_X.index)
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)


# on rajoute la colonne cible prédite à num_X_test_inverse
num_X_inverse = num_X_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)

#display(num_X_inverse[(-pd.isnull(num_X_inverse.FirstPumpArriving_AttendTime_pred))&(num_X_inverse.CalYear == 2020)&(num_X_inverse.CalMonth == 11)&(num_X_inverse.CalDay ==14)][['CalDay','CalMonth','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalDay','HourOfCall']).head(30))
res = num_X_inverse[(-pd.isnull(num_X_inverse.FirstPumpArriving_AttendTime_pred))&(num_X_inverse.CalYear == 2020)&(num_X_inverse.CalMonth == 11)&(num_X_inverse.CalDay ==14)][['CalDay','CalMonth','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalDay','HourOfCall']).head(30)

# on affiche les écarts dans un graphique pour cette journée
plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle RidgeCV')
plt.legend();





####################### MODELISATIONS  ==> LassoCV

# Création du modèle de régression LassoCV
model_lasso = LassoCV().fit(X_train, y_train)

# Prédictions à partir de X_test
pred_test = model_lasso.predict(X_test)
# Prédictions à partir de X_train
pred_train = model_lasso.predict(X_train)

# Affichage des performances du modèle score R carré + mean_squared_error
print("score train:", model_lasso.score(X_train, y_train))
print("mse train:", mean_squared_error(pred_train, y_train))
print("score test:", model_lasso.score(X_test, y_test))
print("mse test:", mean_squared_error(pred_test, y_test))


##score train: 0.5805036551202245
##mse train: 0.41852160138784983
##score test: 0.5762807159318282
##mse test: 0.4276558080220872


# Reconstitution de l'échantillon de test avec la colonne cible dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]

# num_X_test avec la cible réelle FirstPumpArriving_AttendanceTime
num_X_test = num_X1.join(num_X3)

# Reconstitution de l'échantillon de test avec la colonne prédite dans le but de dénormaliser pour visualiser les écarts entrainement / test
num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded','casernes_quartier','casernes_sous_quartier','CalWeekDay','CalDay','CalMonth']]
num_X4 = pd.DataFrame(pred_test, columns=['FirstPumpArriving_AttendTime_pred'])

# num_X_pred avec la cible prédite FirstPumpArriving_AttendTime_pred
num_X_pred = num_X1.join(num_X4)


# on dénormalise les deux dataframes
num_X_test_inverse = pd.DataFrame(scaler.inverse_transform(num_X_test), columns=num_X_test.columns, index= num_X_test.index)
num_X_pred_inverse = pd.DataFrame(scaler.inverse_transform(num_X_pred), columns=num_X_pred.columns, index= num_X_pred.index)


# on rajoute la colonne cible prédite à num_X_test_inverse
num_X_test_inverse = num_X_test_inverse.join(num_X_pred_inverse.FirstPumpArriving_AttendTime_pred)


# display(num_X_test_inverse[(-pd.isnull(num_X_test_inverse.FirstPumpArriving_AttendTime_pred))&(num_X_test_inverse.CalYear == 2020)&(num_X_test_inverse.CalMonth == 11)&(num_X_test_inverse.CalDay ==14)][['CalDay','CalMonth','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalDay','HourOfCall']).head(30))
res = num_X_test_inverse[(-pd.isnull(num_X_test_inverse.FirstPumpArriving_AttendTime_pred))&(num_X_test_inverse.CalYear == 2020)&(num_X_test_inverse.CalMonth == 11)&(num_X_test_inverse.CalDay ==14)][['CalDay','CalMonth','HourOfCall', 'FirstPumpArriving_AttendanceTime','FirstPumpArriving_AttendTime_pred' ]].sort_values(['CalDay','HourOfCall']).head(30)

# on affiche les écarts dans un graphique pour cette journéePropertyCategory

plt.figure(figsize=(15,8))
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendTime_pred, c='r', label='Temps Prédit')
plt.scatter(res.HourOfCall, res.FirstPumpArriving_AttendanceTime, c='b', label='Temps Réel')
plt.xlabel('Heures d\'appel')
plt.ylabel('Temps d\'arrivée en minutes')
plt.yticks([180,240,300,360,420,480],[3,4,5,6,7,8])
plt.title('Comparaison Temps réels / Temps prédits par le modèle LassoCV')
plt.legend();


##################################################################################################
#
#Pour la partie Machine Learning, nous allons essayer d'estimer le temps d'arrivée des pompiers à partir de l'heure d'appel à leurs services.
#Pour cela nous avons utilisé des méthodes de Régression Linéaires : LinearRegression, RidgeCV  et LassoCV.
#
#La variable cible identifiée est :
#* FirstPumpArriving_AttendanceTime
#
#Les variables numériques de nos datasets que nous avons retenues sont :
#* DateOfCall retravaillée en CalYear, CalMonth, CalDay, CalWeekDay
#* HourOfCall
#* Easting_rounded
#* Northing_rounded
#
#Les variables catégorielles de nos datasets que nous avons retenues sont :
#* IncidentGroup
#* StopCodeDescription
#* SpecialServiceType
#* AddressQualifier
#* ProperCase
#* IncidentStationGround
#* FirstPumpArriving_DeployedFromStation
#* DeployedFromLocation
#* DelayCodeId
#
#De plus nous avons enrichi nos données en rajoutant des variables calculées :
#* casernes_quartier : le nombre de casernes du quartier de l'incident (ex. dans le quartier de Westminster il y a 8 casernes)
#* casernes_sous_quartier : le nombre de casernes du sous quartier de l'incident (ex. dans le sous quartier de Barkingside il y a 2 casernes)
#
#
#
#Modèles de Machine Learning expérimentés
#
#La heatmap de corrélation des variables numériques ne nous a pas permis de trouver une variable fortement corrélée avec notre cible.
#La corrélation la plus élevée concerne la variable casernes_sous_quartier avec un coefficient de 0.094 ce qui est trop faible pour faire un modèle de Régression Linéaire simple.
#
#(Insérer le graphique Heatmap)
#
#
#
#
#Modèle LinearRegression
#
#Le coefficient de détermination du modèle sur l'échantillon d'entrainement est de 0.5839818857279384
#Le coefficient de détermination du modèle sur l'échantillon de test est de 0.5786472154140925
#
#Les scores sont faibles mais relativement proches entre nos 2 échantillons.
#
#Le graphique suivant nous montre sur une journée de l'échantillon de test les écarts entre les temps réels d'arrivée des pompiers et les temps prédits 
#par le modèle LinearRegression à différentes heurs de la journée.
#
#(Insérer le graphique modélisation_LinearRegression.jpg)
#
#
#
#
#
#Modèle RidgeCV
#
#Le score du modèle sur l'échantillon d'entrainement est de 0.5839438555837259
#Le score du modèle sur l'échantillon de test est de 0.5787123523986961
#
#Le mean_squared_error sur l'échantillon d'entrainement est de 0.4150893945887835
#Le mean_squared_error sur l'échantillon de test est de 0.4252015806664894
#
#Les scores sont faibles mais relativement proches entre nos 2 échantillons.
#
#Le graphique suivant nous montre sur une journée de l'échantillon de test les écarts entre les temps réels d'arrivée des pompiers et les temps prédits 
#par le modèle RidgeCV à différentes heurs de la journée.
#
#(Insérer le graphique modélisation_RidgeCV.jpg)
#
#
#
#Modèle LassoCV
#
#Le score du modèle sur l'échantillon d'entrainement est de 0.5805036551202245
#Le score du modèle sur l'échantillon de test est de 0.5762807159318282
#
#Le mean_squared_error sur l'échantillon d'entrainement est de 0.41852160138784983
#Le mean_squared_error sur l'échantillon de test est de 0.4276558080220872
#
#Les scores sont faibles mais relativement proches entre nos 2 échantillons.
#
#Le graphique suivant nous montre sur une journée de l'échantillon de test les écarts entre les temps réels d'arrivée des pompiers et les temps prédits 
#par le modèle LassoCV à différentes heurs de la journée.
#
#(Insérer le graphique modélisation_LassoCV.jpg)
#
#
#
#
#
#
##################################################################################################