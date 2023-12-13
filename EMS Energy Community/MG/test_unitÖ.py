from MG.file_costo_batteria import calcola_costo_operazionale_batteria

P_G_predetta=3
P_L_predetta=1
delta_t=0.25
p_GL_S=0
p_GL_N=2
PR_3=150 
TRAS_e=8.48  
max_BTAU_m=0.61 
CPR=0.026  
Pz=3.2 
is_CER=1
B=5000
pv_price=7400
e_on_MWh_TO_e_on_kWh=0.001
TP_CE=110
numero_timeslot=96
SoC_0=0.5
SoC=0.5
eta=0.98
Q=5 
b=1.665 
a=2744 
p_S_k=0 



# calcola il ricavo per l'energia condivisa
E_cond=0
E_prod=delta_t*(P_G_predetta)  # energia prodotta [kWh]
if p_GL_S > 0:
    E_prel = delta_t*(P_L_predetta + p_GL_S) # energia prelevata [kWh]
if p_GL_S <= 0:
    E_prel = delta_t*(P_L_predetta) # energia prelevata [kWh]
E_cond=min(E_prod, E_prel)  # energia condivisa [kWh]
energia_venduta=0
if p_GL_N>0:
    energia_venduta=p_GL_N
energia_acquistata=0
if p_GL_N<0:
    energia_acquistata=abs(p_GL_N)
I_rit=PR_3*e_on_MWh_TO_e_on_kWh*delta_t*energia_venduta  # contributo per il ritiro dell'energia immessa in N [€/kWh]
CU_af_m=(TRAS_e+max_BTAU_m)*e_on_MWh_TO_e_on_kWh  # consumo unitario del corrispettivo forfettario mensile [€/kWh]    
I_rest=0  # restituzione componenti tariffarie [€]
if is_CER==1:  # se si tratta di una CER
    I_rest=CU_af_m*E_cond
else:  # se non si tratta di una CER, quindi si tratta di un AUC
    I_rest=CU_af_m*E_cond+CPR*Pz*e_on_MWh_TO_e_on_kWh*E_cond
I_cond =TP_CE*e_on_MWh_TO_e_on_kWh*E_cond  # incentivazione energia condivisa [€]
ricavo=I_cond+I_rest+I_rit  # ricavo in k [€]


# calcola il costo di investimento
costo_batteria=B;
anni_vita=10;
giorni_anno=365;
orizzonte_temporale=96;
costo_investimento=(pv_price+costo_batteria)/(anni_vita*365*numero_timeslot);


# calcola il costo operazionale della batteria
p_S_k=abs(p_GL_S)  # energia scambiata con S 
C_b_k=calcola_costo_operazionale_batteria(SoC_0, SoC, eta, B, Q, b, a, p_S_k, delta_t)

# calcola i costi di acquisto dell'energia
prezzo_energia=0.29 # [€/kWh]
costo_acquisto=energia_acquistata*prezzo_energia

# calcola i costi finali
# costo_investimento=0 # punto di vista del prosumer
costo_decisione=-ricavo+costo_investimento+C_b_k+costo_acquisto