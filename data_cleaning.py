# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:13:18 2021

@author: Home
"""

import pandas as pd
from datetime import datetime as dt


debut = dt.today()
incident = pd.read_excel("LFB_Incident_January 2017_2021.xlsx",sheet_name='Sheet1',)
incident.head()

fin = dt.today()
print(fin - debut)

mobilisation = pd.read_excel("LFB Mobilisation data from January 2017.xlsx")
df_merged = incident.merge(mobilisation)
# Informations sur df_merged
df_merged.info()


# test si inc Geo_WardName même valeur que IncGeo_WardNameNew
sorted(df_merged['IncGeo_WardName']) == sorted(df_merged['IncGeo_WardNameNew'])

#### Avant suppression des variables

to_drop = ['ResourceMobilisationId','Resource_Code','TimeMobileTimezoneId',
           'TimeArrivedTimezoneId','TimeLeftTimezoneId','DateAndTimeReturned',
           'TimeReturnedTimezoneId','DeployedFromLocation','PlusCode_Code',
           'PlusCode_Description','CalYear','Postcode_full','UPRN','USRN',
           'IncGeo_BoroughName','IncGeo_WardName','Easting_m','Northing_m',
           'Easting_rounded','Northing_rounded','FRS']

new_df_merged = df_merged.drop(to_drop,axis=1)

def valeur_manquante(df):
    flag=0
    for col in df.columns:
            if df[col].isna().sum() > 0:
                flag=1
                print(f'"{col}": {df[col].isna().sum()} valeurs manquantes')
    if flag==0:
        print("Le dataset ne contient plus de valeurs manquantes, bien joué.")
valeur_manquante(new_df_merged)

# Attention filtre sur les notnull supprime environ 150000 lignes dans un new df
new_df_merged = new_df_merged[new_df_merged['Latitude'].notnull()]
# colonnes en doublons après la fusion à supprimer
to_drop = ['PerformanceReporting' ,'DateAndTimeMobilised' ,'AttendanceTimeSeconds' ,'PumpOrder' ,'IncidentStationGround']
new_df_merged = new_df_merged.drop(to_drop,axis=1)

# set index
new_df_merged = new_df_merged.set_index('IncidentNumber')
new_df_merged['SpecialServiceType'] = new_df_merged['SpecialServiceType'].fillna('RAS')

# categoriser et typer les variables
to_category = ['IncidentGroup','StopCodeDescription','PropertyCategory' ,'PropertyType' ,'AddressQualifier' ,
               'Postcode_district' ,'IncGeo_BoroughCode' ,'ProperCase' ,
               'IncGeo_WardCode' ,'IncGeo_WardNameNew' ,'SpecialServiceType']
new_df_merged[to_category] = new_df_merged[to_category].astype('category')

valeur_manquante(new_df_merged)


new_df_merged['DateAndTimeMobile'] = pd.to_datetime(new_df_merged['DateAndTimeMobile'], errors='coerce')
new_df_merged['DateAndTimeArrived'] = pd.to_datetime(new_df_merged['DateAndTimeArrived'], errors='coerce')
new_df_merged['DateAndTimeLeft'] = pd.to_datetime(new_df_merged['DateAndTimeLeft'], errors='coerce')

valeur_manquante(new_df_merged)

# fillna 0
to_fillna = ['PumpCount','PumpHoursRoundUp' ,'Notional Cost (£)' ,'DelayCodeId',
             'SecondPumpArriving_AttendanceTime','FirstPumpArriving_AttendanceTime',
             'DateAndTimeMobile','DateAndTimeLeft' ,'DeployedFromStation_Code' ,'DeployedFromStation_Name']
new_df_merged[to_fillna] = new_df_merged[to_fillna].fillna(0)

valeur_manquante(new_df_merged)


# Remplacer les valeurs par 'no'

to_fill_no = ['FirstPumpArriving_DeployedFromStation','SecondPumpArriving_DeployedFromStation','DelayCode_Description']
new_df_merged[to_fill_no] = new_df_merged[to_fill_no].fillna('no')

new_df_merged['FirstPumpArriving_DeployedFromStation','SecondPumpArriving_DeployedFromStation'].astype('category')

cat_data = new_df_merged.select_dtypes(include=['O','category'])
print(len(cat_data.columns),"colonnes de type objets")
cat_data.columns


num_data = new_df_merged.select_dtypes(include=['float64' ,'int64'])
print(len(num_data.columns),"colonnes numériques")
num_data.columns

date_data = new_df_merged.select_dtypes(include=['datetime','datetime64'])
print(len(date_data.columns),"colonnes date")
date_data.columns


valeur_manquante(new_df_merged)
# timing fin
fin = dt.today()
print(fin - debut)