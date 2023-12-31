# classe della singola microgrid

class MG:
    
    def __init__(self, is_CER, SoC, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,
                 CPR,Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0, pv_price, SoC_autoconsumo) :
        self.is_CER=is_CER
        self.SoC=SoC
        self.Q=Q
        self.P_S_max=P_S_max
        self.eta=eta
        self.SoC_min=SoC_min
        self.SoC_max=SoC_max
        self.PR_3=PR_3
        self.B=B
        self.a=a
        self.b=b
        self.CPR=CPR
        self.Pz=Pz
        self.TP_CE=TP_CE
        self.TRAS_e=TRAS_e
        self.max_BTAU_m=max_BTAU_m
        self.SoC_0=SoC_0
        self.pv_price=pv_price
        self.SoC_autoconsumo=SoC_autoconsumo


    def resetta_stato(self,
                      autoconsumo):
        if autoconsumo=='no':
            self.SoC=self.SoC_0
        if autoconsumo=='yes':
            self.SoC_autoconsumo=self.SoC_0

        
    def simula_microgrid(self,
                         alpha,
                         P_G_predetta,
                         P_L_predetta,
                         delta_t,
                         numero_timeslot,
                         autoconsumo):
        
        
        
        # SIMULAZIONE NORMALE OVVERO SENZA AUTOCONSUMO
        if autoconsumo=='no':

            SoC_prec=self.SoC 
            # importa la funzione per calcolare il costo operazionale di S
            from MG.file_costo_batteria import calcola_costo_operazionale_batteria
            # importa il modulo per passare variabili per valore 
            import copy
            # dichiarazione dei flussi energetici
            p_GL_S=0
            p_GL_N=0
            costo_decisione=0
            ricavo_da_vendita=0
            e_on_MWh_TO_e_on_kWh = 0.001
            SoC_0=copy.deepcopy(self.SoC)
            
            
            # regole alla base del calcolo
            p_GL=P_G_predetta-P_L_predetta
            p_GL=round(p_GL,2)
            capienza_energetica_residua_S=round(self.Q*(self.SoC_max-self.SoC),2)
            energia_residua_S=round(self.Q*(self.SoC-self.SoC_min),2)
            
            if p_GL==0:
                p_GL_S=0
                p_GL_N=0
                
            if p_GL>0:
                if p_GL*delta_t<=capienza_energetica_residua_S and p_GL<=self.P_S_max:
                    p_GL_S=round(alpha*p_GL,2)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    # ======================================== SOC == SOE
                    # this is the part should be updated
                    self.SoC=self.SoC+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC=self.SoC_max
                if p_GL*delta_t>capienza_energetica_residua_S and p_GL<=self.P_S_max:
                    p_GL_S=round(alpha*capienza_energetica_residua_S/delta_t,2)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    self.SoC=self.SoC+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S     
                        self.SoC=self.SoC_max
                if p_GL*delta_t>capienza_energetica_residua_S and p_GL>self.P_S_max:
                    p_GL_S=round(alpha*self.P_S_max,2)*self.eta
                    p_GL_N=p_GL-p_GL_S  
                    self.SoC=self.SoC+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC=self.SoC_max
                if p_GL*delta_t<=capienza_energetica_residua_S and p_GL>self.P_S_max:
                    p_GL_S=round(alpha*self.P_S_max)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    self.SoC=self.SoC+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S 
                        self.SoC=self.SoC_max
                
    
            if p_GL<0:
                if abs(p_GL*delta_t)<=energia_residua_S and abs(p_GL)<=self.P_S_max:
                    p_GL_S=-round(alpha*abs(p_GL),2)/self.eta
                    p_GL_N=p_GL-p_GL_S    
                    self.SoC=self.SoC-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC=self.SoC_min
                if abs(p_GL*delta_t)>energia_residua_S and abs(p_GL)<self.P_S_max:
                    p_GL_S=-round(alpha*energia_residua_S/delta_t,2)/self.eta
                    p_GL_N=p_GL-p_GL_S 
                    self.SoC=self.SoC-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S 
                        self.SoC=self.SoC_min
                if abs(p_GL*delta_t)>energia_residua_S and abs(p_GL)>self.P_S_max:
                    p_GL_S=-round(alpha*self.P_S_max,2)/self.eta
                    p_GL_N=p_GL-p_GL_S      
                    self.SoC=self.SoC-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC=self.SoC_min
                if abs(p_GL*delta_t)<=energia_residua_S and abs(p_GL)>self.P_S_max:
                    p_GL_S=-round(alpha*self.P_S_max,2)/self.eta
                    p_GL_N=p_GL-p_GL_S  
                    self.SoC=self.SoC-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC=self.SoC_min
                
     
            # calcola il ricavo per l'energia condivisa
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
            I_rit=self.PR_3*e_on_MWh_TO_e_on_kWh*delta_t*energia_venduta  # contributo per il ritiro dell'energia immessa in N [€/kWh]
            CU_af_m=(self.TRAS_e+self.max_BTAU_m)*e_on_MWh_TO_e_on_kWh  # consumo unitario del corrispettivo forfettario mensile [€/kWh]    
            I_rest=0  # restituzione componenti tariffarie [€]
            if self.is_CER==1:  # se si tratta di una CER
                I_rest=CU_af_m*E_cond
            else:  # se non si tratta di una CER, quindi si tratta di un AUC
                I_rest=CU_af_m*E_cond+self.CPR*self.Pz*e_on_MWh_TO_e_on_kWh*E_cond
            I_cond =self.TP_CE*e_on_MWh_TO_e_on_kWh*E_cond  # incentivazione energia condivisa [€]
            ricavo=I_cond+I_rest+I_rit  # ricavo in k [€]
            
            
            # calcola il costo di investimento
            costo_batteria=self.B;
            anni_vita=10;
            giorni_anno=365;
            orizzonte_temporale=96;
            costo_investimento=(self.pv_price+costo_batteria)/(anni_vita*365*numero_timeslot);
            
            
            # controlla l'SoC
            if round(self.SoC,2)>self.SoC_max or round(self.SoC,2)<self.SoC_min:
                raise ValueError('SoC fuori dai limiti !')
            # controlla il bilancio energetico
            somma=p_GL-p_GL_N-p_GL_S
            if round(somma,2)!=0:
                raise ValueError('Bilancio energetico non rispettato !')
                
                
            # calcola il costo operazionale della batteria
            p_S_k=abs(p_GL_S)  # energia scambiata con S 
            C_b_k=calcola_costo_operazionale_batteria(SoC_prec, self.SoC, self.eta, self.B,
                                                      self.Q, self.b, self.a, p_S_k, delta_t)
            
            # calcola i costi di acquisto dell'energia
            prezzo_energia=0.29 # [€/kWh]
            costo_acquisto=energia_acquistata*prezzo_energia
            
            # calcola i costi finali
            costo_investimento=0 # punto di vista del prosumer
            costo_decisione=-ricavo+costo_investimento+C_b_k+costo_acquisto
            
            # compone i risultati
            risultati=[self.SoC,alpha,round(p_GL_N,2),round(p_GL_S,2),round(costo_decisione,2)]
            
            
            
            
            
            
        # SIMULAZIONE IN AUTOCONSUMO   
        if autoconsumo=='yes':

            SoC_prec=self.SoC_autoconsumo 
            # importa la funzione per calcolare il costo operazionale di S
            from MG.file_costo_batteria import calcola_costo_operazionale_batteria
            # importa il modulo per passare variabili per valore 
            import copy
            # dichiarazione dei flussi energetici
            p_GL_S=0
            p_GL_N=0
            costo_decisione=0
            ricavo_da_vendita=0
            e_on_MWh_TO_e_on_kWh = 0.001
            SoC_0=copy.deepcopy(self.SoC)
            
            
            # regole alla base del calcolo
            p_GL=P_G_predetta-P_L_predetta
            p_GL=round(p_GL,2)
            capienza_energetica_residua_S=round(self.Q*(self.SoC_max-self.SoC_autoconsumo),2)
            energia_residua_S=round(self.Q*(self.SoC_autoconsumo-self.SoC_min),2)
            
            if p_GL==0:
                p_GL_S=0
                p_GL_N=0
                
            if p_GL>0:
                if p_GL*delta_t<=capienza_energetica_residua_S and p_GL<=self.P_S_max:
                    p_GL_S=round(alpha*p_GL,2)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    self.SoC_autoconsumo=self.SoC_autoconsumo+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC_autoconsumo=self.SoC_max
                if p_GL*delta_t>capienza_energetica_residua_S and p_GL<=self.P_S_max:
                    p_GL_S=round(alpha*capienza_energetica_residua_S/delta_t,2)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    self.SoC_autoconsumo=self.SoC_autoconsumo+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S     
                        self.SoC_autoconsumo=self.SoC_max
                if p_GL*delta_t>capienza_energetica_residua_S and p_GL>self.P_S_max:
                    p_GL_S=round(alpha*self.P_S_max,2)*self.eta
                    p_GL_N=p_GL-p_GL_S  
                    self.SoC_autoconsumo=self.SoC_autoconsumo+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC_autoconsumo=self.SoC_max
                if p_GL*delta_t<=capienza_energetica_residua_S and p_GL>self.P_S_max:
                    p_GL_S=round(alpha*self.P_S_max)*self.eta
                    p_GL_N=p_GL-p_GL_S
                    self.SoC_autoconsumo=self.SoC_autoconsumo+abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo>self.SoC_max:
                        eccesso=(self.SoC_max-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S-eccesso/delta_t
                        p_GL_N=p_GL-p_GL_S 
                        self.SoC_autoconsumo=self.SoC_max
                
    
            if p_GL<0:
                if abs(p_GL*delta_t)<=energia_residua_S and abs(p_GL)<=self.P_S_max:
                    p_GL_S=-round(alpha*abs(p_GL),2)/self.eta
                    p_GL_N=p_GL-p_GL_S    
                    self.SoC_autoconsumo=self.SoC_autoconsumo-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC_autoconsumo=self.SoC_min
                if abs(p_GL*delta_t)>energia_residua_S and abs(p_GL)<self.P_S_max:
                    p_GL_S=-round(alpha*energia_residua_S/delta_t,2)/self.eta
                    p_GL_N=p_GL-p_GL_S 
                    self.SoC_autoconsumo=self.SoC_autoconsumo-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S 
                        self.SoC_autoconsumo=self.SoC_min
                if abs(p_GL*delta_t)>energia_residua_S and abs(p_GL)>self.P_S_max:
                    p_GL_S=-round(alpha*self.P_S_max,2)/self.eta
                    p_GL_N=p_GL-p_GL_S      
                    self.SoC_autoconsumo=self.SoC_autoconsumo-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC_autoconsumo=self.SoC_min
                if abs(p_GL*delta_t)<=energia_residua_S and abs(p_GL)>self.P_S_max:
                    p_GL_S=-round(alpha*self.P_S_max,2)/self.eta
                    p_GL_N=p_GL-p_GL_S  
                    self.SoC_autoconsumo=self.SoC_autoconsumo-abs(p_GL_S*delta_t)/self.Q
                    if self.SoC_autoconsumo<self.SoC_min:
                        difetto=(self.SoC_min-self.SoC_autoconsumo)*self.Q
                        p_GL_S=p_GL_S+difetto/delta_t
                        p_GL_N=p_GL-p_GL_S
                        self.SoC_autoconsumo=self.SoC_min
                
     
            # calcola il ricavo per l'energia condivisa
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
            I_rit=self.PR_3*e_on_MWh_TO_e_on_kWh*delta_t*energia_venduta  # contributo per il ritiro dell'energia immessa in N [€/kWh]
            CU_af_m=(self.TRAS_e+self.max_BTAU_m)*e_on_MWh_TO_e_on_kWh  # consumo unitario del corrispettivo forfettario mensile [€/kWh]    
            I_rest=0  # restituzione componenti tariffarie [€]
            if self.is_CER==1:  # se si tratta di una CER
                I_rest=CU_af_m*E_cond
            else:  # se non si tratta di una CER, quindi si tratta di un AUC
                I_rest=CU_af_m*E_cond+self.CPR*self.Pz*e_on_MWh_TO_e_on_kWh*E_cond
            I_cond =self.TP_CE*e_on_MWh_TO_e_on_kWh*E_cond  # incentivazione energia condivisa [€]
            ricavo=I_cond+I_rest+I_rit  # ricavo in k [€]
            
            
            # calcola il costo di investimento
            costo_batteria=self.B;
            anni_vita=10;
            giorni_anno=365;
            orizzonte_temporale=96;
            costo_investimento=(self.pv_price+costo_batteria)/(anni_vita*365*numero_timeslot);
            
            
            # controlla l'SoC
            if round(self.SoC_autoconsumo,2)>self.SoC_max or round(self.SoC_autoconsumo,2)<self.SoC_min:
                raise ValueError('SoC fuori dai limiti !')
            # controlla il bilancio energetico
            somma=p_GL-p_GL_N-p_GL_S
            if round(somma,2)!=0:
                raise ValueError('Bilancio energetico non rispettato !')
                
                
            # calcola il costo operazionale della batteria
            p_S_k=abs(p_GL_S)  # energia scambiata con S 
            C_b_k=calcola_costo_operazionale_batteria(SoC_prec, self.SoC_autoconsumo, self.eta, self.B,
                                                      self.Q, self.b, self.a, p_S_k, delta_t)
            
            # calcola i costi di acquisto dell'energia
            prezzo_energia=0.29 # [€/kWh]
            costo_acquisto=energia_acquistata*prezzo_energia
            
            # calcola i costi finali
            costo_investimento=0 # punto di vista del prosumer
            costo_decisione=-ricavo+costo_investimento+C_b_k+costo_acquisto
            
            # compone i risultati
            risultati=[self.SoC_autoconsumo,alpha,round(p_GL_N,2),round(p_GL_S,2),round(costo_decisione,2)]
        
        
        
        
        return risultati
            
            
        