# MAIN

# importa il pacchetto json 
import json
# importa la classe Ottimizzatore
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
folder_ottimizzatore_path = os.path.join(current_dir, 'EMS')
sys.path.append(folder_ottimizzatore_path)

from EMS.file_classe_EMS import EMS
# from file_classe_EMS import EMS
from EMS.file_ottimizza_FIS import ottimizzaFIS
# importa la classe della microgrid
from MG.file_classe_microgrid import MG
# importa il modulo per generare numeri random
import numpy as np
np.random.seed(1337)
# importa il modulo per plottare
import matplotlib.pyplot as plt
# importa il modulo per calcoalre il fis ottimizzato
from FO.globale_fine_ottimizzazione import globale_fine_ottimizzazione
import time
import matplotlib.pyplot as plt
import math
import statistics

# Start timer
start_time = time.time()


# SIMULAZIONE INTERNA PER AUTOCONSUMO

# parametri comuni a tutte le microgrid
PR_3=150  # prezzo dell'energia per il ritiro dedicato [€/MWh]
TRAS_e=8.48  # tariffa di trasmissione definita per le utenze in bassa tensione [€/MWh]
max_BTAU_m=0.61  # valore maggiore della componente variabile di distribuzione [€/MWh]
CPR=0.026  # coefficiente delle perdite di rete evitate [-]
Pz=3.2  # prezzo zonale orario [€/MWh]
a_x=2800
b_x=2.4
B_x=7000



# microgrid 1
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.5
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_1=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 2
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.2
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_2=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 3
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.6
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_3=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 4
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.8
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_4=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 5
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.9
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_5=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 6
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.95
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_6=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 7
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.15
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteriab=1.665  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_7=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)



# LEGGE I METADATI PER L'OTTIMIZZAZIONE
with open("meta-dati.json") as file:
    file_meta_dati=json.load(file)
nome_funzione_obiettivo=file_meta_dati["FO"]
limiti_variabili=[]
metaparametri=[]
funzione_obiettivo=None
numero_esecuzioni=file_meta_dati["numero_esecuzioni"]
if nome_funzione_obiettivo=="globale":
    from FO.globale import globale
    funzione_obiettivo=globale
    limiti_variabili=file_meta_dati["limiti_variabili_globale"]
if nome_funzione_obiettivo=="modello_CER_AUC":
    from FO.modello_CER_AUC import modello_CER_AUC
    funzione_obiettivo=modello_CER_AUC
    limiti_variabili=file_meta_dati["limiti_variabili_CER_AUC"]
if nome_funzione_obiettivo=="rastrigin":
    from FO.rastrigin import rastrigin
    funzione_obiettivo=rastrigin
    limiti_variabili=file_meta_dati["limiti_variabili_rastrigin"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="rosenbrock":
    from FO.rosenbrock import rosenbrock
    funzione_obiettivo=rosenbrock
    limiti_variabili=file_meta_dati["limiti_variabili_rosenbrock"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="sferica":
    from FO.sferica import sferica
    funzione_obiettivo=sferica
    limiti_variabili=file_meta_dati["limiti_variabili_sferica"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="schwefel":
    from FO.schwefel import schwefel
    funzione_obiettivo=schwefel
    limiti_variabili=file_meta_dati["limiti_variabili_schwefel"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="griewank":
    from FO.griewank import griewank
    funzione_obiettivo=griewank
    limiti_variabili=file_meta_dati["limiti_variabili_griewank"]
algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if algoritmo_di_ottimizzazione=="GA":
    metaparametri=file_meta_dati["metaparametri_GA"]
else:
    print("Non ci sono altri algoritmi implementati!")



# OTTIMIZZAZIONE DEL FIS

# carica l'intero training set
import scipy.io
training_set_matlab = scipy.io.loadmat('training_set_per_python.mat')
training_set = training_set_matlab['training_set_per_python']
lunghezza_training_set=95*3
training_set=training_set[0:14,0:lunghezza_training_set]
# # test con overfitting
# test_set_matlab = scipy.io.loadmat('test_set_per_python.mat')
# test_set = test_set_matlab['test_set_per_python']
# lunghezza_test_set=1
# training_set=test_set[0:14,0:lunghezza_test_set]


# dati sulle MG (modello fisico simulato)
dati_MG=[MG_1,
         MG_2,
         MG_3,
         MG_4,
         MG_5,
         MG_6,
         MG_7]

# carica il fis
dati_fis=[]
with open("fis.json") as file:
    dati_fis=json.load(file)
    

   
# ottimizza il fis
n_esecuzioni=1
n_geni=30;
fo_ottime=np.zeros([n_esecuzioni])
individui_ottimi=np.zeros([n_esecuzioni,n_geni])
for esecuzione in range(n_esecuzioni):
    dati_modello=[dati_fis,dati_MG,training_set]
    risultati_ottimizzazione=ottimizzaFIS(dati_modello,
                           funzione_obiettivo, 
                           limiti_variabili, 
                           metaparametri,
                           nome_funzione_obiettivo, 
                           esecuzione)
    fo_ottima = risultati_ottimizzazione[0][1][0]
    individuo_ottimo =risultati_ottimizzazione[0][2][0]
    fo_ottime[esecuzione]=fo_ottima
    individui_ottimi[esecuzione,:]=individuo_ottimo
fo_ottima = statistics.mean(fo_ottime)
individuo_ottimo=np.zeros([n_geni])
for g in range(n_geni):
    somma_geni=0
    for e in range(n_esecuzioni):
        gene = individui_ottimi[e][g]
        somma_geni= somma_geni+gene
    individuo_ottimo[g]=somma_geni/n_esecuzioni

# End timer
end_time = time.time()
# Calcola il tempo di esecuzione
elapsed_time_training = end_time - start_time


# test 
test_set_matlab = scipy.io.loadmat('test_set_per_python.mat')
test_set = test_set_matlab['test_set_per_python']
lunghezza_test_set=95
test_set=test_set[0:14,0:lunghezza_test_set]
dati_modello_test=[dati_fis,dati_MG, test_set]

# Start timer
start_time_test = time.time()
risultati_test=globale_fine_ottimizzazione(dati_modello_test,individuo_ottimo)
# End timer
end_time_test = time.time()
# Calcola il tempo di esecuzione
elapsed_time_test = end_time_test - start_time_test

FO_globale_energy_community=risultati_test[0]
matrice_decisioni=risultati_test[1]
matrice_SoC=risultati_test[2]
matrice_P_GL_S=risultati_test[3]
matrice_P_GL_N=risultati_test[4]
matrice_FO=risultati_test[5]
costo_computazionale=risultati_test[6]
fis_ottimo=risultati_test[7]
ascisse_MF_input=risultati_test[8]
ascisse_MF_output=risultati_test[9]
conseguenti=risultati_test[10]
pesi_regole=risultati_test[11]
FO_autoconsumo_energy_community=risultati_test[12]
vettore_FO=risultati_test[13]

# salva i risultati
nome_sottocartella = "Risultati"
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
percorso_sottocartella = os.path.join(cartella_corrente, nome_sottocartella)

nome_file = "individuo_ottimo.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),individuo_ottimo)

nome_file = "fis_ottimo.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),fis_ottimo)

nome_file = "FO_ottima_training_set.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),fo_ottima)

nome_file = "tempo_training.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),elapsed_time_training)

nome_file = "tempo_test.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),elapsed_time_test)

nome_file = "FO_ottima_test_set.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),FO_globale_energy_community)

nome_file = "matrice_decisioni.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),matrice_decisioni)

nome_file = "matrice_SoC.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),matrice_SoC)

nome_file = "matrice_P_GL_S.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_S)

nome_file = "matrice_P_GL_N.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_N)

nome_file = "matrice_FO.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),matrice_FO)


 

# resa grafica
X=np.arange(0,len(test_set[0]),1)
fig_1 = plt.figure()
plt.title('Alfa')
plt.scatter(X,matrice_decisioni[0],1,'k',label="MG 1")
plt.scatter(X,matrice_decisioni[1],1,'b',label="MG 2")
plt.scatter(X,matrice_decisioni[2],1,'g',label="MG 3")
plt.scatter(X,matrice_decisioni[3],1,'r',label="MG 4")
plt.scatter(X,matrice_decisioni[4],1,'y',label="MG 5")
plt.scatter(X,matrice_decisioni[5],1,'m',label="MG 6")
plt.scatter(X,matrice_decisioni[6],1,'c',label="MG 7")
plt.xlabel('timeslot')           
plt.ylabel('Alfa')
plt.legend()
plt.draw()
nome_file = 'alfa.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()
#
fig_2 = plt.figure()
plt.title('SoC')
plt.plot(X,matrice_SoC[0],1,'k',label="MG 1")
plt.plot(X,matrice_SoC[1],1,'b',label="MG 2")
plt.plot(X,matrice_SoC[2],1,'g',label="MG 3")
plt.plot(X,matrice_SoC[3],1,'r',label="MG 4")
plt.plot(X,matrice_SoC[4],1,'y',label="MG 5")
plt.plot(X,matrice_SoC[5],1,'m',label="MG 6")
plt.plot(X,matrice_SoC[6],1,'c',label="MG 7")
plt.xlabel('timeslot')           
plt.ylabel('SoC')
plt.legend()
plt.draw()
nome_file = 'SoC.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()
#
fig_3 = plt.figure()
plt.title('P GL->S')
plt.plot(X,matrice_P_GL_S[0],3,'k',label="MG 1")
plt.plot(X,matrice_P_GL_S[1],3,'b',label="MG 2")
plt.plot(X,matrice_P_GL_S[2],3,'g',label="MG 3")
plt.plot(X,matrice_P_GL_S[3],3,'r',label="MG 4")
plt.plot(X,matrice_P_GL_S[4],3,'y',label="MG 5")
plt.plot(X,matrice_P_GL_S[5],3,'m',label="MG 6")
plt.plot(X,matrice_P_GL_S[6],3,'m',label="MG 7")
plt.xlabel('timeslot')           
plt.ylabel('P GL->S [kW]')
plt.legend()
plt.draw()
nome_file = 'P_GL_S.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()
#
fig_4 = plt.figure()
plt.title('P GL->N')
plt.plot(X,matrice_P_GL_N[0],3,'k',label="MG 1")
plt.plot(X,matrice_P_GL_N[1],3,'b',label="MG 2")
plt.plot(X,matrice_P_GL_N[2],3,'g',label="MG 3")
plt.plot(X,matrice_P_GL_N[3],3,'r',label="MG 4")
plt.plot(X,matrice_P_GL_N[4],3,'y',label="MG 5")
plt.plot(X,matrice_P_GL_N[5],3,'m',label="MG 6")
plt.plot(X,matrice_P_GL_N[6],3,'m',label="MG 7")
plt.xlabel('timeslot')           
plt.ylabel('P GL->N [kW]')
plt.legend()
plt.draw()
nome_file = 'P_GL_N.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()

#
fig_5 = plt.figure()
plt.title('FO autoc. vs FO globale [€]: '+str(round(FO_autoconsumo_energy_community,2))+'|'+str(round(FO_globale_energy_community,2)))
plt.plot(X,matrice_FO[0],1,'k',label="MG 1")
plt.plot(X,matrice_FO[1],1,'b',label="MG 2")
plt.plot(X,matrice_FO[2],1,'g',label="MG 3")
plt.plot(X,matrice_FO[3],1,'r',label="MG 4")
plt.plot(X,matrice_FO[4],1,'y',label="MG 5")
plt.plot(X,matrice_FO[5],1,'m',label="MG 6")
plt.plot(X,matrice_FO[6],1,'c',label="MG 7")
plt.xlabel('timeslot')           
plt.ylabel('FO [€]')
plt.legend()
plt.draw()
nome_file = 'FO.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()


# FIS
# input term set
# trap
points = [(ascisse_MF_input[0][0],0),(ascisse_MF_input[0][1],1),(ascisse_MF_input[0][2],1),
          (ascisse_MF_input[0][3],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Very Low', color='red')
#tri
points = [(ascisse_MF_input[1][0],0),(ascisse_MF_input[1][1],1),(ascisse_MF_input[1][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Low', color='Orange')
#tri
points = [(ascisse_MF_input[2][0],0),(ascisse_MF_input[2][1],1),(ascisse_MF_input[2][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Medium', color='Cyan')
#tri
points = [(ascisse_MF_input[3][0],0),(ascisse_MF_input[3][1],1),(ascisse_MF_input[3][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='High', color='Blue')
# trap
points = [(ascisse_MF_input[4][0],0),(ascisse_MF_input[4][1],1),(ascisse_MF_input[4][2],1),
          (ascisse_MF_input[4][3],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Very High', color='Purple')
plt.title('Input Term Set')
plt.xlabel('x')
plt.ylabel('m(x)')
plt.legend()
plt.draw()
nome_file = 'input_term_set.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()

# output term set
# trap
points = [(ascisse_MF_output[0][0],0),(ascisse_MF_output[0][1],1),(ascisse_MF_output[0][2],1),
          (ascisse_MF_output[0][3],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Very Low', color='red')
#tri
points = [(ascisse_MF_output[1][0],0),(ascisse_MF_output[1][1],1),(ascisse_MF_output[1][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Low', color='Orange')
#tri
points = [(ascisse_MF_output[2][0],0),(ascisse_MF_output[2][1],1),(ascisse_MF_output[2][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Medium', color='Cyan')
#tri
points = [(ascisse_MF_output[3][0],0),(ascisse_MF_output[3][1],1),(ascisse_MF_output[3][2],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='High', color='Blue')
# trap
points = [(ascisse_MF_output[4][0],0),(ascisse_MF_output[4][1],1),(ascisse_MF_output[4][2],1),
          (ascisse_MF_output[4][3],0)]
x = [point[0] for point in points]
y = [point[1] for point in points]
plt.plot(x, y, label='Very High', color='Purple')
plt.title('Output Term Set')
plt.xlabel('x')
plt.ylabel('m(x)')
plt.legend()
plt.draw()
nome_file = 'output_term_set.eps'
plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
plt.show()

#regole
nome_file = "conseguenti.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),conseguenti)
nome_file = "pesi_regole.npy"
np.save(os.path.join(percorso_sottocartella, nome_file),pesi_regole)

print("fine")










