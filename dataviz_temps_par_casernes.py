%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Chargement du dataset des incidents
df = pd.read_excel('LFB Incident data from January 2017.xlsx')


# Création nouveau dataframe de travail dfw  en retirant les colonnes inutiles pour la dataviz
dfw = df.drop(['Postcode_full', 'UPRN', 'USRN', 'IncGeo_BoroughCode', 'ProperCase', 'IncGeo_WardCode', 'Easting_m', 'Northing_m',
       'Easting_rounded', 'Northing_rounded', 'FRS', 'NumStationsWithPumpsAttending', 'NumPumpsAttending', 'PumpCount',
       'IncGeo_WardNameNew', 'Latitude', 'Longitude', 'PumpHoursRoundUp', 'Notional Cost (£)'], axis=1)

# utilisation du script pour détecter les colonnes avec des valeurs manquantes
# def valeur_manquante(df):
#     flag=0
#     for col in df.columns:
#             if df[col].isna().sum() > 0:
#                 flag=1
#                 print(f'"{col}": {df[col].isna().sum()} valeurs manquantes')
#     if flag==0:
#         print("Le dataset ne contient plus de valeurs manquantes, bien joué.")
# valeur_manquante(dfw)

# suppression des lignes avec temps d'arrivée manquant et des lignes avec caserne d'origine manquante
dfw.dropna(subset = ['FirstPumpArriving_AttendanceTime'], axis=0, inplace = True)
dfw.dropna(subset = ['FirstPumpArriving_DeployedFromStation'], axis=0, inplace = True)

# Récupération des noms des casernes les plus rapides et les plus lentes
# on ne prend que les incidents pour lesquels la caserne qui est intervenue est celle du secteur de l'incident pour ne pas défavoriser les casernes intervenues ailleurs
res = pd.DataFrame(dfw[dfw.IncidentStationGround == dfw.FirstPumpArriving_DeployedFromStation].groupby(['FirstPumpArriving_DeployedFromStation'], as_index = False).agg({'FirstPumpArriving_AttendanceTime' : 'median'}) )
print("Récupération des 3 casernes les plus rapides :\n")
display(res.sort_values('FirstPumpArriving_AttendanceTime', ascending=True).head(3))
print("Récupération des 3 casernes les plus lentes :\n")
display(res.sort_values('FirstPumpArriving_AttendanceTime', ascending=False).head(3))


# récupération des infos pour la dataviz dans res_h
res_h = pd.DataFrame(dfw[dfw.IncidentStationGround == dfw.FirstPumpArriving_DeployedFromStation].groupby(['FirstPumpArriving_DeployedFromStation', 'HourOfCall'], as_index = False).agg({'FirstPumpArriving_AttendanceTime' : 'mean'}) )

# Dataviz Temps d'arrivée moyen selon l'heure de la journée par caserne (3 plus rapides vs 3 plus lentes)
plt.figure(figsize=(6,20))
sns.relplot(x = 'HourOfCall', y = 'FirstPumpArriving_AttendanceTime',kind='line', 
            palette = 'viridis', height=8,
            data = res_h[res_h.FirstPumpArriving_DeployedFromStation.isin(['Orpington', 'Wennington', 'Ruislip', 'Lewisham', 'Brixton', 'Whitechapel'])], 
            hue = 'FirstPumpArriving_DeployedFromStation' )
plt.title('Temps d\'arrivée moyen selon l\'heure de la journée par caserne (3 plus rapides vs 3 plus lentes)')
plt.xlabel('Heure d\'appel')
plt.ylabel('Temps d\'arrivée moyen (minutes)')
plt.yticks([240, 300, 360, 420, 480, 540], [4, 5, 6, 7, 8, 9])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], ['0','2','4','6','8','10','Midi','14','16','18','20','22','Minuit'])
;


# Etude de la corrélation entre le temps d'arrivée des pompiers et la caserne du lieu de l'incident par le test ANOVA

import statsmodels.api

#                                      variable_continue                 variable_categorielle 
result = statsmodels.formula.api.ols('FirstPumpArriving_AttendanceTime ~ IncidentStationGround', data = dfw).fit()  
table = statsmodels.api.stats.anova_lm(result)

print(table)
##                             df        sum_sq       mean_sq           F  PR(>F)\
##IncidentStationGround     101.0  3.744967e+08  3.707889e+06  224.890618     0.0 
##Residual               389079.0  6.414948e+09  1.648752e+04         NaN     NaN 
##


# la p-value (PR(>F)) est inférieur à 5 % elle vaut 0.0 donc on rejette l'hypothèse 
# selon laquelle IncidentStationGround n'influe pas sur FirstPumpArriving_AttendanceTime




