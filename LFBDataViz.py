# Importation et nettoyage rapide des données

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("bright")

#Importation et nettoyage rapide des données
#Importation
df_inci = pd.read_csv(filepath_or_buffer="LFB_incidents.csv", sep = ";", low_memory=False)
df_mob = pd.read_csv(filepath_or_buffer="LFB_mobilisations.csv", sep = ";", low_memory=False)

#Supression des colonnes inutiles
df_inci_drop = df_inci.drop(["CalYear", "Postcode_full", "UPRN", "USRN", "IncGeo_BoroughName", "IncGeo_BoroughName", "IncGeo_WardName", "Easting_m", "Northing_m", "Easting_rounded", "Northing_rounded", "FRS", "IncidentStationGround"], axis=1)
df_mob_drop = df_mob.drop(["ResourceMobilisationId", "Resource_Code", "PerformanceReporting", "DateAndTimeMobilised", "TimeMobileTimezoneId", "TimeArrivedTimezoneId", "AttendanceTimeSeconds", "TimeLeftTimezoneId", "DateAndTimeReturned", "TimeReturnedTimezoneId", "DeployedFromLocation", "PumpOrder", "PlusCode_Code", "PlusCode_Description"], axis=1)

#Merge
df_lfb = df_inci_drop.merge(df_mob_drop, on = "IncidentNumber", how = "left")

#Conversion des données concernées
df_lfb['DateOfCall']= pd.to_datetime(df_lfb['DateOfCall'])
df_lfb.head()


##########


#Classement des incidents principaux

#Création et paramètres du graphique
plt.figure(figsize=(7,5))
typeincident = sns.countplot(x="IncidentGroup", data=df_lfb, order = df_lfb['IncidentGroup'].value_counts().index)
plt.title("Principaux type d'incidents", fontsize=14)
plt.xlabel("Catégorie d'incident", fontsize=12)
plt.ylabel("Nombre d'incidents", fontsize=12)
plt.show();

#Analyse
print("Nous pouvons constater que la majorité des incidents signalés sont des fausses alarmes (automatique ou bien manuelle.)")
print("Viennent ensuite les 'Special Service', puis en dernière position les incendies.")


##########


#Temps passé par type d’intervention

from datetime import datetime

dftime = df_lfb[["StopCodeDescription", "DateAndTimeArrived"]]
dfarr = df_lfb["DateAndTimeArrived"]
dfarr = pd.to_datetime(dfarr)
dftime["arrived"] = dfarr.dt.strftime("%H:%M:%S")

dfdep = df_lfb["DateAndTimeLeft"]
dfdep = pd.to_datetime(dfdep)
dftime["departure"] = dfdep.dt.strftime("%H:%M:%S")

dftime = dftime.drop(columns=["DateAndTimeArrived"])
dftime = dftime.dropna(axis = 0, how = "any")


dftime['diff'] = dftime["departure"].apply(lambda x: datetime.strptime(x,'%H:%M:%S')) - dftime["arrived"].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
dftime['seconds'] = dftime['diff'].dt.total_seconds()
dftime = dftime.drop(dftime[dftime["seconds"] < 0].index)

value=3600
dftime['min']=(dftime['seconds']/value).round(2)

#Création et parametres du graphique
sns.catplot(x = 'StopCodeDescription', y = 'min', height=6, aspect=2, s=8, data = dftime)
plt.yticks([0,1,2,3,4,5,6,8,10,12.5,15,17.5,20], fontsize=12)
plt.xticks(rotation = 45, fontsize=12)
plt.title("Temps d'intervention par type d'incident", fontsize=20)
plt.xlabel("Types d'incidents", fontsize=16)
plt.ylabel("Temps en heures", fontsize=16)
plt.show();

#Analyse
print("Nous constatons que les temps d'interventions par type d'incident sont généralement de 2h,")
print("avec des temps plus élevés comme pour les 'Incendies' et 'Special Services' (inondation, suicide, ouverture d'ascenseurs...")
print("Les incendies sont les types d'incident qui peuvent retenir les pompiers plusieurs 10aines d'heures.")
print("Enfin nous pouvons visualiser que les 3 types d'alarme engrangent énormément de temps cumulé inutile.")


##########


#Temps d’arrivée moyen selon la période de l’année

#Séléction des données qui nous intéressent
dfmeantime = df_lfb[["FirstPumpArriving_AttendanceTime", "DateOfCall"]]
dfmeantime['year'] = dfmeantime['DateOfCall'].dt.year
dfmeantime['month'] = dfmeantime['DateOfCall'].dt.month
dftime = dfmeantime.drop(columns=["DateOfCall"])
dftime = dftime.dropna(axis = 0, how = "any")
dftime['FirstPumpArriving_AttendanceTime'] = dftime['FirstPumpArriving_AttendanceTime']/60
dftime['FirstPumpArriving_AttendanceTime'] = dftime['FirstPumpArriving_AttendanceTime']/10000
dfmeanfinal=dftime.groupby(["month","year"]).sum()
dfmeanfinal=dfmeanfinal.reset_index()
dfmeanfinal = dfmeanfinal.drop(dfmeanfinal[dfmeanfinal["FirstPumpArriving_AttendanceTime"] < 1].index)

#Création et parametres du graphique
fig, ax = plt.subplots(figsize=(20,10))
sns.barplot(x="month", y="FirstPumpArriving_AttendanceTime", hue="year", data=dfmeanfinal, ax=ax)
mois=['Jan','Fev','Mars','Avr',"Mai","Juin",'Juil','Aout','Sep','Oct','Nov','Dec']
ax.set_xticklabels(mois, fontsize=14)
plt.yticks(fontsize=14)
plt.title("Temps d'arrivée moyen par période", fontsize=20)
plt.xlabel(None)
plt.ylabel("Temps d'arrivée en minutes", fontsize=16)
pd.options.mode.chained_assignment = None
plt.legend(prop={'size': 14})
plt.show();

#Analyse
print("Nous pouvons remarquer que le temps d'arrivée moyen fluctue selon la période de l'année,")
print("avec des temps élevés sur la période estivale, et en diminution sur la période hivernale.")
print("Nous observons également un pic sur le mois de juillet 2018.")


##########


# Répartition des types de retards

#Séléction des données qui nous intéressent
counts = df_lfb['DelayCode_Description'].value_counts()
dfdelay = df_lfb.loc[df_lfb['DelayCode_Description'].isin(counts.index[counts < 25000])]
dfdelaylast = dfdelay['DelayCode_Description'].value_counts()

#Création et parametres du graphique
dfdelaylast.plot.pie(y='dfdelaylast', figsize=(10.5, 10.),labels=None,  autopct = lambda x: str(round(x,)) + '%',colors = ['#82ff84','#f9baff','#f7b660','#ccff99','#ff9999','#99ffec','#bad5ff','#ffee99','#6ebbff'], pctdistance = 0.83)
centre_circle = plt.Circle((0,0),0.7,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Motifs des retards en %", fontsize=15)
plt.ylabel(None)
labels=dfdelaylast.index.unique()
plt.legend(labels, loc="center",title = "Raisons des retards :", title_fontsize='large', prop={'size': 11})
plt.show();

#Analyse
print("Nous constatons que les embouteillages représentent une grande majorité avec 51% des raisons de retard,")
print("suivi par 17% des anomalies d'adresses et enfin, 14% des retards sont dus aux mesures de régulation de la circulation.")
print("Le reste des motifs reste très minoritaire. Nous allons les regrouper dans la variable 'Autres' pour le reste de cette analyse.")
print("Nous allons maintenant étudier le type de retard par période de la semaine et de la journée et enfin calculer leurs temps moyens.")


##########


# Répartition des types de retards par jour de la semaine

#Séléction des données qui nous intéressent
dflate = df_lfb[["HourOfCall", "DateOfCall", "DelayCode_Description"]]
dflate['day'] = pd.to_datetime(dflate["DateOfCall"])
dflate['day'] = dflate['day'].apply(lambda x: x.weekday())
dflate = dflate.dropna(axis = 0, how = "any")
dflate = dflate.drop(columns=["DateOfCall"])
dflate = dflate.loc[dflate['DelayCode_Description'].isin(counts.index[counts < 25000])]
dflate['DelayCode_Description'] = dflate['DelayCode_Description'].replace({'At drills when mobilised':'Others', 'On outside duty when mobilised':'Others', 'Appliance/Equipment defect':'Others', 'Weather conditions':'Others', 'Mob/Radio problems when mobilised':'Others', 'Arrived but held up - Other reason':'Others'})

#Création du graphique
fig, ax = plt.subplots(figsize=(17,8))
sns.countplot(x='day', hue = 'DelayCode_Description', palette=["#82ff84", "#6ebbff", "#f7b660", "#f9baff"], data=dflate)
jours=['Lundi','Mardi','Mercredi','Jeudi',"Vendredi","Samedi",'Dimanche']
ax.set_xticklabels(jours, fontsize=12)
plt.title("Types de retard par jour de la semaine", fontsize=16)
plt.ylabel("Retards comptabilisés", fontsize=14)
plt.xlabel(None)
plt.legend(prop={'size': 12}, title = "Raisons des retards :", title_fontsize='large')
plt.show();

#Analyse
print("Comme le démontrait le graphique précédent, les embouteillages sont fortement représentés,")
print("ceux-ci connaissent un pic en milieu de semaine et une régression le week-end.")
print("Une fois regroupé, la variable 'Autres' représente la 2eme cause des retards, avec les journées du mercredi et samedi en pic.")


##########

# Répartition des types de retards par heures de la journée

#Création du graphique reprenant les informations du graphique précédent
fig, ax = plt.subplots(figsize=(13,7))
sns.kdeplot(x='HourOfCall', hue = 'DelayCode_Description', palette=["#82ff84", "#6ebbff", "#f7b660", "#f9baff"], cut=1, linewidth=7, data=dflate)
plt.yticks(fontsize=12)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24],fontsize=12)
plt.title("Densités des types de retard par heure de la journée", fontsize=16)
plt.xlabel("Heures", fontsize=14)
plt.ylabel("Densité", fontsize=14)
liste=['Address incomplete/wrong','Traffic calming measures', 'Others','Traffic, roadworks, etc']
plt.legend(liste, loc='upper left', prop={'size': 12},title = "Raisons des retards :", title_fontsize='large')
plt.show();

#Analyse
print("Intéressons-nous à la variable la plus forte : 'Embouteillage'.")
print("Sans surprise, celle-ci est très prononcée lors des heures de travail, entre 7h et 20h.")
print("C'est ensuite la variable 'Autres' qui prend le relais avec un pic entre 10h et 16h.")
print("Et enfin la variable 'Anomalies d'adresses' qui se distingue en soirée.")


##########


# Calcul du temps d’arrivée moyen, avec et sans motifs de retards.

#Séléction des données qui nous intéressent pour le delta du temps des interventions sans retard
dfsd = df_lfb[['DateAndTimeMobile', 'DateAndTimeArrived', 'DelayCode_Description']]
dfsd['DelayCode_Description']=dfsd['DelayCode_Description'].replace('Not held up', 1)
dfsd['DelayCode_Description']=dfsd['DelayCode_Description'].fillna(1)
dfsd = dfsd.dropna(axis = 0, how = "any")
dfsd = (dfsd.loc[dfsd['DelayCode_Description'] == 1])

dfsd['DateAndTimeMobile'] = pd.to_datetime(dfsd['DateAndTimeMobile'])
dfsd['DateAndTimeMobile']= dfsd['DateAndTimeMobile'].dt.strftime("%H:%M:%S")
dfsd['DateAndTimeArrived'] = pd.to_datetime(dfsd['DateAndTimeArrived'])
dfsd['DateAndTimeArrived']= dfsd['DateAndTimeArrived'].dt.strftime("%H:%M:%S")

dfsd['diff'] = dfsd['DateAndTimeArrived'].apply(lambda x: datetime.strptime(x,'%H:%M:%S')) - dfsd['DateAndTimeMobile'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
dfsd['diff seconds'] = dfsd['diff'].dt.total_seconds()
dfsd = dfsd.drop(dfsd[dfsd["diff seconds"] < 0].index)

value=60
dfsd['min']=(dfsd['diff seconds']/value).round(2)
dfsd = dfsd.drop(columns=["DateAndTimeMobile", "DateAndTimeArrived", "diff", 'diff seconds'])

#Séléction des données qui nous interessent pour le delta du temps des interventions avec retards
dfad = df_lfb[['DateAndTimeMobile', 'DateAndTimeArrived', 'DelayCode_Description']]
dfad['DelayCode_Description']=dfad['DelayCode_Description'].replace('Not held up', 1)
dfad['DelayCode_Description']=dfad['DelayCode_Description'].fillna(1)
dfad = dfad.dropna(axis = 0, how = "any")
dfad = (dfad.loc[dfad['DelayCode_Description'] != 1])

dfad['DateAndTimeMobile'] = pd.to_datetime(dfad['DateAndTimeMobile'])
dfad['DateAndTimeMobile']= dfad['DateAndTimeMobile'].dt.strftime("%H:%M:%S")
dfad['DateAndTimeArrived'] = pd.to_datetime(dfad['DateAndTimeArrived'])
dfad['DateAndTimeArrived']= dfad['DateAndTimeArrived'].dt.strftime("%H:%M:%S")

dfad['diff'] = dfad['DateAndTimeArrived'].apply(lambda x: datetime.strptime(x,'%H:%M:%S')) - dfad['DateAndTimeMobile'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
dfad['diff seconds'] = dfad['diff'].dt.total_seconds()
dfad = dfad.drop(dfad[dfad["diff seconds"] < 0].index)

value=60
dfad['min']=(dfad['diff seconds']/value).round(2)
dfad = dfad.drop(columns=["DateAndTimeMobile", "DateAndTimeArrived", "diff", 'diff seconds'])

#Création du graphique
AD = dfad["min"].mean()
AD = np.round(AD,0)
SD = dfsd["min"].mean()
SD = np.round(SD,0)
Delta = AD - SD

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
bar_x = ['Avec retard', 'Sans retard', 'Delta']
bar_height = [AD,SD,Delta]
bar_tick_label = ['Avec retard', 'Sans retard', 'Delta']
bar_label = [AD,SD,Delta]

bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

autolabel(bar_plot)

plt.ylim(0,8)
ax.set_ylabel('Temps moyens de trajet en minutes', fontsize=14)
ax.set_title('Temps moyens avec et sans motif de retard', fontsize=15)
ax.bar(x,y, color = ["#ff8282", '#82ff84', '#a1a1a1'])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show();

#Analyse
print("Une intervention avec retard mettra en moyenne 7 min contre 4 min en temps normal, entre le départ et l'arrivée des pompiers sur les lieux.")
print("Il y a donc une moyenne de 3 minutes dès qu'un retard est enregistré.")


##########


# Graphique prédiction temps : Temps de mobilisation par arrondissement

#Séléction des données qui nous intéressent
dfrespbri = df_lfb[['ProperCase', 'TimeOfCall', 'DateAndTimeMobile']]
dfrespbri['DateAndTimeMobile'] = pd.to_datetime(dfrespbri['DateAndTimeMobile'])
dfrespbri['DateAndTimeMobile'] = dfrespbri['DateAndTimeMobile'].dt.strftime("%H:%M:%S")
dfrespbri = dfrespbri.dropna(axis = 0, how = "any")

dfrespbri['diff'] = dfrespbri['DateAndTimeMobile'].apply(lambda x: datetime.strptime(x,'%H:%M:%S')) - dfrespbri['TimeOfCall'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
dfrespbri['diff seconds'] = dfrespbri['diff'].dt.total_seconds()
dfrespbri = dfrespbri.drop(dfrespbri[dfrespbri["diff seconds"] < 0].index)
dfrespbri = dfrespbri.drop(columns=["TimeOfCall", "DateAndTimeMobile", "diff"])
dfrespbri = dfrespbri.drop(dfrespbri[dfrespbri["diff seconds"] > 10].index)
dfrespbri = dfrespbri.groupby("ProperCase").mean()
dfrespbri = dfrespbri.sort_values(by = 'diff seconds', ascending = False)
dfrespbri = dfrespbri.reset_index()
dfrespbri['diff seconds'] = dfrespbri['diff seconds']*10

#Création du graphique
ax = dfrespbri[['ProperCase','diff seconds']].plot(kind='bar', figsize=(15, 10), fontsize=12, color="#8fdfff")
ax.set_title('Temps de mobilisation par arrondissement', fontsize=15)
ax.get_legend().remove()
ax.set_xticklabels(dfrespbri.ProperCase)
ax.set_ylabel('Temps en secondes', fontsize=14)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100],fontsize=12)
plt.show();

#Analyse
print("Ce graphique représente par arrondissement, le temps moyen écoulé entre le moment de l'appel et le départ des pompiers.")
print("Nous constatons un delta moyen de 37 secondes entre les 2 arrondissements en valeurs extrêmes.")
print("La moyenne générale est de 80 secondes.")


##########


# Graphique prédiction temps : Temps de mobilisation et de trajet par type d’incident

#Séléction des données qui nous intéressent
dftpi = df_lfb[['TimeOfCall','DateAndTimeArrived' ,'DelayCode_Description', 'StopCodeDescription']]
dftpi['DelayCode_Description']=dftpi['DelayCode_Description'].replace('Not held up', 1)
dftpi['DelayCode_Description']=dftpi['DelayCode_Description'].fillna(1)
dftpi = dftpi.dropna(axis = 0, how = "any")
dftpi = (dftpi.loc[dftpi['DelayCode_Description'] == 1])
dftpi['DateAndTimeArrived'] = pd.to_datetime(dftpi['DateAndTimeArrived'])
dftpi['DateAndTimeArrived']= dftpi['DateAndTimeArrived'].dt.strftime("%H:%M:%S")

dftpi['diff'] = dftpi['DateAndTimeArrived'].apply(lambda x: datetime.strptime(x,'%H:%M:%S')) - dftpi['TimeOfCall'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
dftpi['diff seconds'] = dftpi['diff'].dt.total_seconds()
dftpi = dftpi.drop(dftpi[dftpi["diff seconds"] < 0].index)

value=60
dftpi['min']=(dftpi['diff seconds']/value).round(2)
dftpi = dftpi.drop(dftpi[dftpi["min"] > 15].index)
dftpi = dftpi.drop(columns=["TimeOfCall", "DateAndTimeArrived", "diff", "DelayCode_Description","diff seconds"])
dftpi = dftpi.groupby("StopCodeDescription").mean()
dftpi = dftpi.sort_values(by = 'min', ascending = False)
dftpi = dftpi.reset_index()

#Création du graphique
ax = dftpi[['StopCodeDescription','min']].plot(kind='bar', figsize=(12, 8), fontsize=12, color="#ffd000")
ax.set_title("Temps de mobilisation et de trajet par type d'incident", fontsize=15)
ax.get_legend().remove()
ax.set_xticklabels(dftpi.StopCodeDescription)
ax.set_ylabel('Temps en minutes', fontsize=14)
plt.yticks([0,1,2,3,4,5,6],fontsize=12)
plt.show();

#Analyse
print("Ce graphique représente par type d'incident, le temps moyen écoulé entre le moment de l'appel et l'arrivée sur les lieux.")
print("Afin d'avoir les valeurs les plus réelles, les données n'incluent aucun type de retard.")
print("Au vu du graphique, nous pouvons en déduire que les temps de mobilisation et de trajet restent les mêmes pour n'importe")
print("quel type d'incident avec un delta de 29 secondes entre les 2 valeurs extrêmes.")
print("Les équipes et le matériel sont donc déjà prêt pour tous types d’interventions.")



