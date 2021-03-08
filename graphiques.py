%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("pastel")

#importation
df_inci = pd.read_csv(filepath_or_buffer="LFB_incidents.csv", sep = ";")
df_mob = pd.read_csv(filepath_or_buffer="LFB_mobilisations.csv", sep = ";")

#supression des colonnes inutiles
df_inci_drop = df_inci.drop(["CalYear", "Postcode_full", "UPRN", "USRN", "IncGeo_BoroughName", "IncGeo_BoroughName", "IncGeo_WardName", "Easting_m", "Northing_m", "Easting_rounded", "Northing_rounded", "FRS", "IncidentStationGround"], axis=1)
df_mob_drop = df_mob.drop(["ResourceMobilisationId", "Resource_Code", "PerformanceReporting", "DateAndTimeMobilised", "TimeMobileTimezoneId", "TimeArrivedTimezoneId", "AttendanceTimeSeconds", "TimeLeftTimezoneId", "DateAndTimeReturned", "TimeReturnedTimezoneId", "DeployedFromLocation", "PumpOrder", "PlusCode_Code", "PlusCode_Description"], axis=1)

#merge
df_lfb = df_inci_drop.merge(df_mob_drop, on = "IncidentNumber", how = "left")

#conversion des données concernées
df_lfb['DateOfCall']= pd.to_datetime(df_lfb['DateOfCall'])





#Graphique type d'incident

plt.figure(figsize=(7,5))
typeincident = sns.countplot(x="IncidentGroup", data=df_lfb, order = df_lfb['IncidentGroup'].value_counts().index)
plt.title("Nombre de type d'incidents principaux", fontsize=14)
plt.xlabel("Catégorie d'incident", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show();

print("Noux pouvons constater que la majorité des incidents signalés sont des fausses alarmes (automatique ou bien manuelle.)")
print("Viennent ensuite les 'Special Service', puis en derniere position les incendies.")




#Graphique temps d'arrivée moyen par periode

#Séléction des données qui nous interessent
dfmeantime = df_lfb[["FirstPumpArriving_AttendanceTime", "DateOfCall"]]
dfmeantime['year'] = dfmeantime['DateOfCall'].dt.year
dfmeantime['month'] = dfmeantime['DateOfCall'].dt.month
dftime = dfmeantime.drop(columns=["DateOfCall"])
dftime = dftime.dropna(axis = 0, how = "any")
pd.options.mode.chained_assignment = None
dfmeanfinal = dftime.groupby(['month', "year"], as_index = False).agg({'FirstPumpArriving_AttendanceTime':'sum'})
dfmeanfinal = dfmeanfinal.sort_values(by = 'FirstPumpArriving_AttendanceTime', ascending = False)

#Graphique
sns.catplot(x = 'month', y = 'FirstPumpArriving_AttendanceTime', hue = "year", kind="bar", height=6, aspect=2, data = dfmeanfinal)
plt.title("Temps d'arrivée moyen par periode de l'année", fontsize=14)
plt.xlabel("Mois", fontsize=12)
plt.ylabel("Temps moyens", fontsize=12)
plt.show();

print("Nous pouvons remarquer que le temps d'arrivée moyen fluctue selon la periode de l'année,")
print("avec des temps élevés sur la periode estivale, et en diminution sur la periode hivernale.")
print("Nous observons également un pic sur le mois de juillet 2018.")
