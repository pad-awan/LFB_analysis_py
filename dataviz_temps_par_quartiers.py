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

# Création nouvelle colonne quartier initialisée à 'aucun'
dfw['quartier'] = 'aucun'

# Valorisation de quartier à partir des cartes London_boroughs et de la carte de découpage des quartiers
dfw.loc[dfw.IncGeo_BoroughName.isin(['BEXLEY','BROMLEY','GREENWICH','LEWISHAM','SOUTHWARK']), 'quartier'] = 'SOUTH EAST'
dfw.loc[dfw.IncGeo_BoroughName.isin(['HAVERING','REDBRIDGE','NEWHAM','BARKING AND DAGENHAM','WALTHAM FOREST', 'TOWER HAMLETS']), 'quartier'] = 'NORTH EAST'
dfw.loc[dfw.IncGeo_BoroughName.isin(['ENFIELD','BARNET','HARINGEY','HACKNEY','ISLINGTON', 'CAMDEN','WESTMINSTER']), 'quartier'] = 'NORTH'
dfw.loc[dfw.IncGeo_BoroughName.isin(['HARROW','HILLINGDON','BRENT','EALING','HOUNSLOW', 'HAMMERSMITH AND FULHAM','KENSINGTON AND CHELSEA']), 'quartier'] = 'WEST'
dfw.loc[dfw.IncGeo_BoroughName.isin(['RICHMOND UPON THAMES','KINGSTON UPON THAMES','WANDSWORTH','MERTON','LAMBETH', 'SUTTON','CROYDON']), 'quartier'] = 'SOUTH WEST'
dfw.loc[dfw.IncGeo_BoroughName.isin(['CITY OF LONDON']), 'quartier'] = 'CITY'


# Dataviz Evolution 2017-2020 du Temps d'arrivée des pompiers par Quartiers
sns.catplot(x='quartier', y='FirstPumpArriving_AttendanceTime',
            kind='boxen', height=9, hue='CalYear',
            data=dfw)
plt.title('Evolution 2017-2020 du Temps d\'arrivée des pompiers par Quartiers')
plt.xlabel('Quartiers')
plt.ylabel('Temps d\'arrivée des pompiers (minutes)')
plt.yticks([180, 240, 300, 360, 420, 480, 540, 600, 900, 1200], [3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
;
