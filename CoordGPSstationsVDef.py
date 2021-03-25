# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from math import cos, sin, acos, pi
import pandas as pd

from urllib.request import urlopen
import requests
# from bs4 import BeautifulSoup

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
# CoordToDeltaKm(51.5136111111111,51,'N',0.270555555555555,0,'O',51.288106053,51,'N',0.1253382582,0,'O')

GPS_quartiers = pd.read_excel('Stations_GPS.xlsx',sheet_name='GPS_quartiers',usecols=[0,1,2,3,4],names=['IncidentStationGround','lat_station','dir_lat_station','lon_station','dir_lon_station'])


# 


df = pd.read_csv('LFB_incidents.csv',sep=';',low_memory=False)
df.head()
dfw = df.merge(right = GPS_quartiers,on='IncidentStationGround',how='left')
# dfw = dfw.fillna(0)
dfw.sort_values(by=["DateOfCall","TimeOfCall"],ascending=[True,True],inplace=True,na_position='last')



dfw = dfw[dfw['Latitude'].notna()]
# dfw['Latitude'] = dfw['Longitude'].fillna(0)
dfw[['IncidentStationGround','lat_station','dir_lat_station','lon_station','dir_lon_station','Latitude','Longitude']]

#

import numpy as np
dfw['dir_lon'] = np.where(dfw['Longitude']<0, 'O', 'E')
dfw['dir_lat'] = 'N'
dfw.head()

gps = []
for lat1, dir_lat_station,lon1,dir_lon_station,lat2,dir_lat,lon2,dir_lon,FirstPumpArriving_AttendanceTime in zip(dfw['lat_station'], dfw['dir_lat_station'],dfw['lon_station'],dfw['dir_lon_station'], dfw['Latitude'],dfw['dir_lat'],dfw['Longitude'],dfw['dir_lon'],dfw['FirstPumpArriving_AttendanceTime']):
    coord = CoordToDeltaKm(lat1,50,dir_lat_station,lon1,0,dir_lon_station,lat2,50,dir_lat,lon2,0,dir_lon)
    # print(lat1, dir_lat_station,lon1,dir_lon_station,lat2,dir_lat,lon2,dir_lon,coord,FirstPumpArriving_AttendanceTime)
    np.array(gps.append(coord))
    # print(coord)
dfw['distkm']=gps
dfw.groupby('IncidentStationGround').agg('distkm').mean().mean()

dfw['distkm'].mean()