# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:37:34 2021

@author: arnau
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:38:29 2021

@author: arnau
"""
import pandas as pd
import numpy as np
from math import cos, sin, acos, pi
from datetime import datetime as dt
import seaborn as sns

# https://www.gladir.com/CODER/PYTHON/coordtodelta.htm
# fonction de calcul du delta de 2 coordonnées convertis en kilomètre
def CoordToDeltaKm (
    Q1Latitude,Q1LatiDeg,Q1LatiDirection,Q1Longitude,Q1LongDeg,Q1LongDirection,
    Q2Latitude,Q2LatiDeg,Q2LatiDirection,Q2Longitude,Q2LongDeg,Q2LongDirection
    ):
    "Calcul la distance entre deux coordonnees Latitude et Longitude en Km"
    a1=(Q1Latitude+(Q1LatiDeg/60.0))*pi/180
    if Q1LatiDirection=="N":
       a1=-a1
    b1=(Q1Longitude+(Q1LongDeg/60.0))*pi/180
    if Q1LongDirection=="O":
       b1=-b1
    a2=(Q2Latitude+(Q2LatiDeg/60.0))*pi/180
    if Q2LatiDirection=="N":
       a2=-a2
    b2=(Q2Longitude+(Q2LongDeg/60.0))*pi/180
    if Q2LongDirection=="O":
       b2=-b2
    RawDelta = acos(cos(a1)*cos(b1)*cos(a2)*cos(b2) + cos(a1)*sin(b1)*cos(a2)*sin(b2) + sin(a1)*sin(a2))
    return RawDelta * 6378.0


t0 = dt.today()

fusion = pd.read_csv('fusion.csv',low_memory=False)

fusion.head()

fusion.iloc[:,0]
GPS_quartiers = pd.read_excel('Stations_GPS.xlsx',sheet_name='GPS_quartiers',usecols=[0,1,2,3,4],names=['IncidentStationGround','lat_station','dir_lat_station','lon_station','dir_lon_station'])

dfw = fusion.merge(GPS_quartiers, on='IncidentStationGround',how='inner')


dfw['dir_lon'] = np.where(dfw['Longitude']<0, 'O', 'E')
dfw['dir_lat'] = 'N'
# dfw.head()


gps = []
for lat1, dir_lat_station,lon1,dir_lon_station,lat2,dir_lat,lon2,dir_lon,FirstPumpArriving_AttendanceTime in zip(dfw['lat_station'], dfw['dir_lat_station'],dfw['lon_station'],dfw['dir_lon_station'], dfw['Latitude'],dfw['dir_lat'],dfw['Longitude'],dfw['dir_lon'],dfw['FirstPumpArriving_AttendanceTime']):
    angle_lat1 = round(lat1,0)
    angle_long1 = round(lon1,0)
    angle_lat2 = round(lat2,0)
    angle_lon2 = round(lon2,0)
    coord = CoordToDeltaKm(lat1,angle_lat1,dir_lat_station,lon1,angle_long1,dir_lon_station,lat2,angle_lat2 ,dir_lat ,lon2, angle_lon2, dir_lon)
    # print(lat1, dir_lat_station,lon1,dir_lon_station,lat2,dir_lat,lon2,dir_lon,coord,FirstPumpArriving_AttendanceTime)
    np.array(gps.append(coord))
    # print(coord)
dfw['distkm']=gps
dfw = dfw[['IncidentNumber','lat_station','lon_station','distkm']]
dfw.to_csv('dfw_distkm.csv')
dfw_distkm = pd.read_csv('dfw_distkm.csv')

dfw_distkm.shape



dfw_distkm = pd.read_csv('dfw_distkm.csv',low_memory=False)


dfw_distkm = dfw_distkm.drop("Unnamed: 0",axis=1)
# dfw_distkm = dfw_distkm.drop("Unnamed: 0.1",axis=1)

dfw_distkm.head()

fusion = pd.read_csv('fusion.csv',low_memory=False)
fusion.head()

# fusion.head()

fusion = fusion.merge(dfw_distkm,how='inner',on='IncidentNumber',copy=False)

fusion.head()
dfw = fusion[(fusion.FirstPumpArriving_DeployedFromStation == fusion.DeployedFromStation_Name)&(fusion.PumpOrder == 1)]
dfw.head()

t1 = dt.today()


print(t1-t0,"secondes de chargement")


##################



dfw = dfw[dfw.CalYear == 2019]


df_cas_quart = pd.read_csv('df_cas_quart_2019.csv',low_memory=False)


df_sqigwd = pd.read_csv('df_sqigwd_2019.csv',low_memory=False)
df_sqigwd.head()


df_quartiers = pd.read_csv('df_quartiers_2019.csv')


df_sqid = pd.read_csv('df_sqid_2019.csv')
df_sqid = df_sqid.drop('Unnamed: 0',axis=1)
dfw.IncidentGroup.unique()

df_ig = pd.DataFrame(columns=('IncidentGroup', 'IncidentGroup_num' ))
for j, ig in enumerate(dfw.IncidentGroup.unique()):
 df_ig.loc[j,'IncidentGroup'] = ig
 df_ig.loc[j,'IncidentGroup_num'] = j
# print(df_ig)
df_sst = pd.DataFrame(columns=('SpecialServiceType', 'SpecialServiceType_num' ))
for m, sst in enumerate(dfw.SpecialServiceType.unique()):
 df_sst.loc[m,'SpecialServiceType'] = sst
 df_sst.loc[m,'SpecialServiceType_num'] = m
# print(df_sst)

dfw = fusion[(fusion.FirstPumpArriving_DeployedFromStation == fusion.DeployedFromStation_Name)&(fusion.PumpOrder == 1)]
dfw = dfw[dfw.CalYear == 2020]
# dfw.shape
dfw = dfw.merge(right = df_cas_quart, on = 'ProperCase' , how = 'inner') 
# dfw.shape
dfw['CalWeekDay'] = pd.to_datetime(dfw['DateOfCall']).dt.weekday
dfw = dfw.merge(right = df_sqigwd, on=["IncGeo_WardName", "IncidentGroup","CalWeekDay"] , how = 'inner') 
# dfw.shape
dfw.SpecialServiceType.fillna('None', inplace=True)
dfw = dfw.merge(right = df_quartiers, on=["ProperCase", "SpecialServiceType","HourOfCall"] , how = 'inner') 
# dfw.shape
dfw = dfw.merge(right = df_ig, on=["IncidentGroup"] , how = 'inner') 
# dfw.shape
dfw = dfw.merge(right = df_sst, on=["SpecialServiceType"] , how = 'inner') 
# dfw.shape
# dfw.DelayCodeId.value_counts()
# dfw.DelayCodeId.fillna(12, inplace=True)


dfw.DelayCodeId = dfw.DelayCodeId.replace({np.nan:12})

# dfw.DelayCodeId.unique()

# dfw.DelayCodeId = dfw.DelayCodeId.select_dtypes('float64')
# dfw = dfw.drop(['Unnamed: 0_x','Unnamed: 0_y','Unnamed: 0_x','Unnamed: 0_y'],axis=1)
# dfw.info()

df_sqid.DelayCodeId =df_sqid.DelayCodeId.replace({np.nan:12,'None':12})
df_sqid.DelayCodeId = df_sqid.DelayCodeId.astype('float64')

df_sqid
dfw.columns
# dfw = dfw.drop('Unnamed: 0_y',axis=1)
dfw = dfw.merge(right = df_sqid, on=["IncGeo_WardName", "DelayCodeId"] , how = 'inner') 
# dfw.shape

dfw.head()


dfw = dfw.drop(['IncidentNumber','TimeOfCall','IncidentGroup','StopCodeDescription','SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier','Postcode_full', 
 'Postcode_district','UPRN', 'USRN', 'IncGeo_BoroughCode','IncGeo_BoroughName','ProperCase','IncGeo_WardCode',
 'IncGeo_WardName','IncGeo_WardNameNew','Easting_m','Northing_m','FRS','IncidentStationGround','FirstPumpArriving_DeployedFromStation',
 'SecondPumpArriving_AttendanceTime','SecondPumpArriving_DeployedFromStation',
 'NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount','PumpHoursRoundUp','Notional Cost (£)',
 'ResourceMobilisationId','Resource_Code','PerformanceReporting','DateAndTimeMobilised',
 'DateAndTimeMobile', 'TimeMobileTimezoneId', 'DateAndTimeArrived',
 'TimeArrivedTimezoneId', 'AttendanceTimeSeconds', 'DateAndTimeLeft',
 'TimeLeftTimezoneId', 'DateAndTimeReturned', 'TimeReturnedTimezoneId',
 'DeployedFromStation_Code', 'DeployedFromStation_Name','DeployedFromLocation','PumpOrder', 'PlusCode_Code','PlusCode_Description', 'DelayCode_Description']
 , axis=1)

dfw.head()



dfw.tstdCodeId.fillna(0, inplace=True)
dfw.casernes_quartier = dfw.casernes_quartier.astype('int64')
dfw.casernes_sous_quartier = dfw.casernes_sous_quartier.astype('int64')
# dfw['DelayCodeId']=='None'
dfw['DelayCodeId'] = dfw['DelayCodeId'].replace({'None', 0})
dfw.HourOfCall = dfw.HourOfCall.astype('int64')
dfw.ProperCase_num = dfw.ProperCase_num.astype('int64')
dfw.CalWeekDay = dfw.CalWeekDay.astype('int64')
dfw.IncGeo_WardName_num = dfw.IncGeo_WardName_num.astype('int64')
dfw.IncidentGroup_num = dfw.IncidentGroup_num.astype('int64')
dfw.SpecialServiceType_num = dfw.SpecialServiceType_num.astype('int64')
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

dfw.info()

# dfw = dfw.drop('Unnamed: 0',axis=0)
num_data = dfw.select_dtypes(include=['float64','int64'])
print(len(num_data.columns),"colonnes numériques")

num_data = num_data.drop(['Unnamed: 0_y'],axis=1)
# num_data = num_data.drop(['Unnamed: 0_x'],axis=1)

# pour chaque colonne numérique
# for col in num_data:
#  # je supprime la colonne initiale de hp pour plus tard joindre toutes les colonnes numériques 
#  # centrées réduites de num_data à hp
#  dfw = dfw.drop(col, axis=1)
# num_data.columns
num_data = num_data[['CalYear', 'HourOfCall', 'CalWeekDay','Easting_rounded', 'Northing_rounded',
       'Latitude', 'Longitude', 'lat_station', 'lon_station','DelayCodeId', 'distkm', 'ProperCase_num','casernes_quartier', 'tmoyq',
       'tmedq', 'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho', 'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId', 'FirstPumpArriving_AttendanceTime']]


num_data.distkm = num_data.distkm.fillna(0)

num_data = num_data.dropna(subset=["Latitude"])

# Matrice de corrélation
cor = num_data.corr()
cor

print(num_data.head())
cor = num_data.corr()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(cor, annot= True, ax= ax, cmap="coolwarm");

# iteration avec le modèle SGBRegressor

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor()

dfw.columns
num_data.columns
num_data = num_data.fillna(0)
target = num_data.FirstPumpArriving_AttendanceTime
feats = num_data.drop(['FirstPumpArriving_AttendanceTime'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2)

print(X_train)
sgd_reg.fit(X_train, y_train) 
#print( "alpha sélectionné par c-v :" ,ridge_reg.alpha_)
print("score train :", sgd_reg.score(X_train, y_train))
print("score test :", sgd_reg.score(X_test, y_test))
sgd_reg_train = sgd_reg.predict(X_train)
sgd_reg_test = sgd_reg.predict(X_test)
print("mse train:", mean_squared_error(sgd_reg_train, y_train))
print("mse test:", mean_squared_error(sgd_reg_test, y_test))


print("rmse train:", np.sqrt(mean_squared_error(sgd_reg_train, y_train)))
print("rmse test:", np.sqrt(mean_squared_error(sgd_reg_test, y_test)))

num_X = X_test[['CalYear', 'HourOfCall', 'CalWeekDay','Easting_rounded', 'Northing_rounded',
       'Latitude', 'Longitude', 'lat_station', 'lon_station','DelayCodeId', 'distkm', 'ProperCase_num','casernes_quartier', 'tmoyq', 'tmedq', 'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
       'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
       'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho', 'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
       'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]
num_X.to_csv('num_X.csv')
pd.read_csv('num_X.csv').head()

# num_X1 = X_test[['CalYear','HourOfCall','Easting_rounded','Northing_rounded']]
# num_X2 = X_test[['DelayCodeId', 'ProperCase_num',
#   'casernes_quartier', 'tmoyq', 'tmedq', 'CalWeekDay',
#   'IncGeo_WardName_num', 'casernes_sous_quartier', 'tmoysqigwd',
#   'tmedsqigwd', 'tstdsqigwd', 'tminsqigwd', 'tmaxsqigwd', 'tmoyqussho',
#   'tmedqussho', 'tstdqussho', 'tminqussho', 'tmaxqussho',
#   'IncidentGroup_num', 'SpecialServiceType_num', 'tmoyCodeId',
#   'tmedCodeId', 'tstdCodeId', 'tminCodeId', 'tmaxCodeId']]
# num_X1 = num_X1.join(num_X2)
# num_X3 = pd.DataFrame(y_test)[['FirstPumpArriving_AttendanceTime']]
num_X_test = num_X.join(num_X3)
# num_X_test.columns
# num_X_test.head()
# sns.heatmap(num_X_test.corr());
num_X_test.to_csv('num_X_test.csv')

t1 = dt.today()
print(t1 - t0,"minutes de chargement")
