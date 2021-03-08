%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importation
df_inci = pd.read_csv(filepath_or_buffer="LFB_incidents.csv", sep = ";")
df_mob = pd.read_csv(filepath_or_buffer="LFB_mobilisations.csv", sep = ";")

#supression des colonnes inutiles
df_inci_drop = df_inci.drop(["CalYear", "Postcode_full", "UPRN", "USRN", "IncGeo_BoroughName", "IncGeo_BoroughName", "IncGeo_WardName", "Easting_m", "Northing_m", "Easting_rounded", "Northing_rounded", "FRS", "IncidentStationGround"], axis=1)
df_mob_drop = df_mob.drop(["ResourceMobilisationId", "Resource_Code", "PerformanceReporting", "DateAndTimeMobilised", "TimeMobileTimezoneId", "TimeArrivedTimezoneId", "AttendanceTimeSeconds", "TimeLeftTimezoneId", "DateAndTimeReturned", "TimeReturnedTimezoneId", "DeployedFromLocation", "PumpOrder", "PlusCode_Code", "PlusCode_Description"], axis=1)

#merge
df_lfb = df_inci_drop.merge(df_mob_drop, on = "IncidentNumber", how = "left")
display(df_lfb.head())
df_lfb.info()





df_lfb.isna().sum(axis=0)





df_lfb['DateAndTimeMobile']= pd.to_datetime(df_lfb['DateAndTimeMobile'])
df_lfb['DateAndTimeLeft']= pd.to_datetime(df_lfb['DateAndTimeLeft'])
df_lfb['DateOfCall']= pd.to_datetime(df_lfb['DateOfCall'])
df_lfb['IncidentGroup'] = df_lfb['IncidentGroup'].astype('category')
df_lfb['DeployedFromStation_Name'] = df_lfb['DeployedFromStation_Name'].astype('category')
df_lfb['DelayCode_Description'] = df_lfb['DelayCode_Description'].astype('category')
df_lfb['StopCodeDescription'] = df_lfb['StopCodeDescription'].astype('category')
df_lfb['SpecialServiceType'] = df_lfb['SpecialServiceType'].astype('category')
df_lfb['PropertyCategory'] = df_lfb['PropertyCategory'].astype('category')
df_lfb['AddressQualifier'] = df_lfb['AddressQualifier'].astype('category')
df_lfb['ProperCase'] = df_lfb['ProperCase'].astype('category')
df_lfb['FirstPumpArriving_AttendanceTime'] = df_lfb['FirstPumpArriving_AttendanceTime'].astype('int64')
df_lfb['SecondPumpArriving_AttendanceTime'] = df_lfb['SecondPumpArriving_AttendanceTime'].astype('int64')
df_lfb['NumStationsWithPumpsAttending'] = df_lfb['NumStationsWithPumpsAttending'].astype('int64')
df_lfb['NumPumpsAttending'] = df_lfb['NumPumpsAttending'].astype('int64')
df_lfb['PumpCount'] = df_lfb['PumpCount'].astype('int64')
df_lfb['PumpHoursRoundUp'] = df_lfb['PumpHoursRoundUp'].astype('int64')
df_lfb['Notional Cost (£)'] = df_lfb['Notional Cost (£)'].astype('int64')
df_lfb.info()


