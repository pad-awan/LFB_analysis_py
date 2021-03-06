# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:13:18 2021

@author: Home
"""

import pandas as pd
from datetime import datetime as dt

debut = dt.today()

usecols1= ['IncidentNumber','DateOfCall','TimeOfCall','IncidentGroup','StopCodeDescription','SpecialServiceType','PropertyCategory' ,'PropertyType',
          'AddressQualifier','IncGeo_BoroughCode','IncGeo_WardNameNew', 'IncGeo_WardCode','ProperCase','Latitude','Longitude','IncidentStationGround',
          'FirstPumpArriving_DeployedFromStation','SecondPumpArriving_DeployedFromStation','FirstPumpArriving_AttendanceTime',
          'SecondPumpArriving_AttendanceTime' ,'NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount',
          'PumpHoursRoundUp','Notional Cost (£)']
file1 = 'LFB Incident data from January 2017.xlsx'
incident = pd.read_excel(io=file1,usecols=usecols1 , sheet_name=0, index_col=0 ,date_parser='DateOfCall')
# incident = pd.read_excel("LFB_Incident_January 2017_2021.xlsx",sheet_name='Sheet1',index_col=0)
file2 = "LFB Mobilisation data from January 2017.xlsx"
usecols2=['IncidentNumber','DateAndTimeMobilised','DateAndTimeMobile','DateAndTimeArrived','DateAndTimeLeft',
         'DeployedFromStation_Code','DelayCodeId','DeployedFromStation_Name']
mobilisation = pd.read_excel(io=file2,index_col=0,usecols = usecols2)
df_merged = incident.merge(mobilisation,how='left',on='IncidentNumber')

def valeur_manquante(df):
    flag=0
    for col in df.columns:
            if df[col].isna().sum() > 0:
                flag=1
                print(f'"{col}": {df[col].isna().sum()} valeurs manquantes')
    if flag==0:
        print("Le dataset ne contient plus de valeurs manquantes, bien joué.")


df_merged = df_merged[df_merged['Latitude'].notnull()]
df_merged['SpecialServiceType'] = df_merged['SpecialServiceType'].fillna('RAS')

# categoriser et typer les variables
to_category = ['IncidentGroup','StopCodeDescription','PropertyCategory' ,'PropertyType' ,'AddressQualifier' ,
               'IncGeo_BoroughCode' ,'ProperCase' ,
               'IncGeo_WardCode' ,'IncGeo_WardNameNew' ,'SpecialServiceType']
df_merged[to_category] = df_merged[to_category].astype('category')

df_merged['DateAndTimeMobile'] = pd.to_datetime(df_merged['DateAndTimeMobile'], errors='coerce')
df_merged['DateAndTimeArrived'] = pd.to_datetime(df_merged['DateAndTimeArrived'], errors='coerce')
df_merged['DateAndTimeLeft'] = pd.to_datetime(df_merged['DateAndTimeLeft'], errors='coerce')

# fillna 0
to_fillna = ['PumpCount','PumpHoursRoundUp' ,'Notional Cost (£)' ,'DelayCodeId',
             'SecondPumpArriving_AttendanceTime','FirstPumpArriving_AttendanceTime',
             'DateAndTimeMobile','DateAndTimeLeft' ,'DeployedFromStation_Code' ,
             'DeployedFromStation_Name','NumStationsWithPumpsAttending','NumPumpsAttending']
df_merged[to_fillna] = df_merged[to_fillna].fillna(0)

# Remplacer les valeurs par 'no'

to_fill_no = ['FirstPumpArriving_DeployedFromStation','SecondPumpArriving_DeployedFromStation']
df_merged[to_fill_no] = df_merged[to_fill_no].fillna('no')


df_merged[to_fill_no] = df_merged[to_fill_no].astype('category')
df_merged.info()
cat_data = df_merged.select_dtypes(include=['O','category'])
print(len(cat_data.columns),"colonnes de type objets")
cat_data.columns


num_data = df_merged.select_dtypes(include=['float64' ,'int64'])
print(len(num_data.columns),"colonnes numériques")
num_data.columns

date_data = df_merged.select_dtypes(include=['datetime','datetime64'])
print(len(date_data.columns),"colonnes date")
date_data.columns


valeur_manquante(df_merged)
df_merged.info()
# timing fin
fin = dt.today()
print(fin - debut)
