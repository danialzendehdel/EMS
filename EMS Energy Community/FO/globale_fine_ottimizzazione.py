


def globale_fine_ottimizzazione(dati_modello, individuo):
# def globale_fine_ottimizzazione(individuo, dati_modello):
    

    
    # importa i moduli necessari
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import os
    import copy
    import time
    
    # Start timer
    start_time = time.time()
    
    # estrae i dati
    copia_dati_modello=copy.deepcopy(dati_modello)
    dati_fis=copia_dati_modello[0]
    dati_MG=copia_dati_modello[1]
    dataset=copia_dati_modello[2]

  
    
    # decodifica l'individuo
    # MFs di input
    numeroMFInput = dati_fis["n_MF_i"]
    for mf_input in range(0,numeroMFInput):
        gene_1=individuo[mf_input*2]
        gene_2=individuo[mf_input*2+1]
        # trapezoidali
        if mf_input==0 or mf_input==4:
            gamma=gene_1
            omega=gene_2
            if mf_input==0:
                prima_ascissa=dati_fis["ascisse_MF_1_i"][0]
                seconda_ascissa=dati_fis["ascisse_MF_1_i"][1]
                b0=dati_fis["ascisse_MF_1_i"][3]
                b=gamma*b0
                a=omega*b
                dati_fis["ascisse_MF_1_i"][2]=round(a,2)
                dati_fis["ascisse_MF_1_i"][3]=round(b,2)
            if mf_input==4:
                b0 = dati_fis["ascisse_MF_5_i"][0]
                a0 = dati_fis["ascisse_MF_5_i"][1]
                terzaAscissa = dati_fis["ascisse_MF_5_i"][2]
                quartaAscissa = dati_fis["ascisse_MF_5_i"][3]
                b = gamma * (1 - b0)
                a = b + omega * (1 - b)
                dati_fis["ascisse_MF_5_i"][0]=round(b,2)
                dati_fis["ascisse_MF_5_i"][1]=round(a,2)
        # triangolari
        larghezzaDiRiferimentoMFTri = 0.30
        if mf_input==1 or mf_input==2 or mf_input==3:
            alfa=gene_1
            beta=gene_2
            indice_MF=str(mf_input+1)
            if alfa >= 1:
                a0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][0]
                c0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][1]
                b0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][2]
                a = a0 - (larghezzaDiRiferimentoMFTri * alfa/2 - larghezzaDiRiferimentoMFTri/2);
                b = b0 + (larghezzaDiRiferimentoMFTri * alfa/2 - larghezzaDiRiferimentoMFTri/2);
                c = a + beta * (b - a)/2; 
                dati_fis["ascisse_MF_"+indice_MF+"_i"][0]=round(a,2)
                dati_fis["ascisse_MF_"+indice_MF+"_i"][1]=round(c,2)
                dati_fis["ascisse_MF_"+indice_MF+"_i"][2]=round(b,2)
            if alfa >= 0.01 and alfa < 1:
                a0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][0]
                c0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][1]
                b0 = dati_fis["ascisse_MF_"+indice_MF+"_i"][2]
                a = a0 + (-larghezzaDiRiferimentoMFTri * alfa/2 + larghezzaDiRiferimentoMFTri/2);
                b = b0 - (-larghezzaDiRiferimentoMFTri * alfa/2 + larghezzaDiRiferimentoMFTri/2);
                c = a + beta * (b - a)/2; 
                dati_fis["ascisse_MF_"+indice_MF+"_i"][0]=round(a,2)
                dati_fis["ascisse_MF_"+indice_MF+"_i"][1]=round(c,2)
                dati_fis["ascisse_MF_"+indice_MF+"_i"][2]=round(b,2)
    # MFs di output
    numeroMFOutput = dati_fis["n_MF_o"]
    for mf_output in range(0,numeroMFOutput):
        gene_1=individuo[numeroMFInput*2+mf_output*2]
        gene_2=individuo[numeroMFInput*2+mf_output*2+1]
        # trapezoidali
        if mf_output==0 or mf_output==4:
            gamma=gene_1
            omega=gene_2
            if mf_output==0:
                prima_ascissa=dati_fis["ascisse_MF_1_o"][0]
                seconda_ascissa=dati_fis["ascisse_MF_1_o"][1]
                b0=dati_fis["ascisse_MF_1_o"][3]
                b=gamma*b0
                a=omega*b
                dati_fis["ascisse_MF_1_o"][2]=round(a,2)
                dati_fis["ascisse_MF_1_o"][3]=round(b,2)
            if mf_output==4:
                b0 = dati_fis["ascisse_MF_5_o"][0]
                a0 = dati_fis["ascisse_MF_5_o"][1]
                terzaAscissa = dati_fis["ascisse_MF_5_o"][2]
                quartaAscissa = dati_fis["ascisse_MF_5_o"][3]
                b = gamma * (1 - b0);
                a = b + omega * (1 - b);
                dati_fis["ascisse_MF_5_o"][0]=round(b,2)
                dati_fis["ascisse_MF_5_o"][1]=round(a,2)
        # triangolari
        larghezzaDiRiferimentoMFTri = 0.30
        if mf_output==1 or mf_output==2 or mf_output==3:
            alfa=gene_1
            beta=gene_2
            indice_MF=str(mf_output+1)
            if alfa >= 1:
                a0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][0]
                c0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][1]
                b0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][2]
                a = a0 - (larghezzaDiRiferimentoMFTri * alfa/2 - larghezzaDiRiferimentoMFTri/2);
                b = b0 + (larghezzaDiRiferimentoMFTri * alfa/2 - larghezzaDiRiferimentoMFTri/2);
                c = a + beta * (b - a)/2; 
                dati_fis["ascisse_MF_"+indice_MF+"_o"][0]=round(a,2)
                dati_fis["ascisse_MF_"+indice_MF+"_o"][1]=round(c,2)
                dati_fis["ascisse_MF_"+indice_MF+"_o"][2]=round(b,2)
            if alfa >= 0.01 and alfa < 1:
                a0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][0]
                c0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][1]
                b0 = dati_fis["ascisse_MF_"+indice_MF+"_o"][2]
                a = a0 + (-larghezzaDiRiferimentoMFTri * alfa/2 + larghezzaDiRiferimentoMFTri/2);
                b = b0 - (-larghezzaDiRiferimentoMFTri * alfa/2 + larghezzaDiRiferimentoMFTri/2);
                c = a + beta * (b - a)/2; 
                dati_fis["ascisse_MF_"+indice_MF+"_o"][0]=round(a,2)
                dati_fis["ascisse_MF_"+indice_MF+"_o"][1]=round(c,2)
                dati_fis["ascisse_MF_"+indice_MF+"_o"][2]=round(b,2)
                
    # pesi
    numero_regole=dati_fis["n_regole"]   
    for indice_regola in range(numero_regole):
        peso=individuo[numeroMFInput*2+numeroMFOutput*2+indice_regola]
        dati_fis["regole"][indice_regola][2]=round(peso,2)
    # conseguenti   
    for indice_conseguente in range(numero_regole):
        conseguente=individuo[numeroMFInput*2+numeroMFOutput*2+numero_regole+indice_conseguente]
        dati_fis["regole"][indice_conseguente][3]=round(conseguente)
    
    somma_FO_globale_energy_community=0
    somma_FO_autoconsumo_energy_community=0
    matrice_decisioni=np.zeros([len(dati_MG),len(dataset[0])])
    matrice_SoC=np.zeros([len(dati_MG),len(dataset[0])])
    matrice_P_GL_S=np.zeros([len(dati_MG),len(dataset[0])])
    matrice_P_GL_N=np.zeros([len(dati_MG),len(dataset[0])])
    matrice_FO=np.zeros([len(dati_MG),len(dataset[0])])
    vettore_FO=np.zeros(96)

    for timeslot in range(len(dataset[0])):
        # ottieni gli SoC delle MG come input del fis
        # MG 1
        risultati_MG_1=dati_MG[0].simula_microgrid(1,
                                    dataset[0][timeslot],
                                    dataset[7][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_1=risultati_MG_1[0]
        alfa_1=risultati_MG_1[1]
        P_GL_N_1=risultati_MG_1[2]
        p_GL_S_1=risultati_MG_1[3]
        costo_1=risultati_MG_1[4]
                
        # MG 2
        risultati_MG_2=dati_MG[1].simula_microgrid(1,
                                    dataset[1][timeslot],
                                    dataset[8][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_2=risultati_MG_2[0]
        alfa_2=risultati_MG_2[1]
        P_GL_N_2=risultati_MG_2[2]
        p_GL_S_2=risultati_MG_2[3]
        costo_2=risultati_MG_2[4]
        # MG 3
        risultati_MG_3=dati_MG[2].simula_microgrid(1,
                                    dataset[2][timeslot],
                                    dataset[9][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_3=risultati_MG_3[0]
        alfa_3=risultati_MG_3[1]
        P_GL_N_3=risultati_MG_3[2]
        p_GL_S_3=risultati_MG_3[3]
        costo_3=risultati_MG_3[4]
        # MG 4
        risultati_MG_4=dati_MG[3].simula_microgrid(1,
                                    dataset[3][timeslot],
                                    dataset[10][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_4=risultati_MG_4[0]
        alfa_4=risultati_MG_4[1]
        P_GL_N_4=risultati_MG_4[2]
        p_GL_S_4=risultati_MG_4[3]
        costo_4=risultati_MG_4[4]
        # MG 5
        risultati_MG_5=dati_MG[4].simula_microgrid(1,
                                    dataset[4][timeslot],
                                    dataset[11][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_5=risultati_MG_5[0]
        alfa_5=risultati_MG_5[1]
        P_GL_N_5=risultati_MG_5[2]
        p_GL_S_5=risultati_MG_5[3]
        costo_5=risultati_MG_5[4]
        # MG 6
        risultati_MG_6=dati_MG[5].simula_microgrid(1,
                                    dataset[5][timeslot],
                                    dataset[12][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_6=risultati_MG_6[0]
        alfa_6=risultati_MG_6[1]
        P_GL_N_6=risultati_MG_6[2]
        p_GL_S_6=risultati_MG_6[3]
        costo_6=risultati_MG_6[4]
        # MG 7
        risultati_MG_7=dati_MG[6].simula_microgrid(1,
                                    dataset[6][timeslot],
                                    dataset[13][timeslot],
                                    0.25,
                                    96,
                                    'yes')
        SoC_MG_7=risultati_MG_7[0]
        alfa_7=risultati_MG_7[1]
        P_GL_N_7=risultati_MG_7[2]
        p_GL_S_7=risultati_MG_7[3]
        costo_7=risultati_MG_7[4]
        
        
        vettore_SoC_input=[SoC_MG_1,
                           SoC_MG_2,
                           SoC_MG_3,
                           SoC_MG_4,
                           SoC_MG_5,
                           SoC_MG_6,
                           SoC_MG_7]
        

        
        FO_globale_autoconsumo = costo_1+costo_2+costo_3+costo_4+costo_5+costo_6+costo_7
        somma_FO_autoconsumo_energy_community= somma_FO_autoconsumo_energy_community+FO_globale_autoconsumo
        
        
        # azzeramento della SoC
        # dati_MG[0].SoC = dati_MG[0].SoC_0
        # dati_MG[1].SoC = dati_MG[1].SoC_0
        # dati_MG[2].SoC = dati_MG[2].SoC_0
        # dati_MG[3].SoC = dati_MG[3].SoC_0
        # dati_MG[4].SoC = dati_MG[4].SoC_0
        # dati_MG[5].SoC = dati_MG[5].SoC_0
        # dati_MG[6].SoC = dati_MG[6].SoC_0
        
        # calcola l'output del fis (gli alpha)
        # definisci il dominio di input e output
        SoC = ctrl.Antecedent(np.arange(0, 1, 0.01), 'SoC')
        alpha = ctrl.Consequent(np.arange(0, 1, 0.01), 'alpha')
        # definisci le MFs
        SoC["molto_basso"]=fuzz.trapmf(SoC.universe, dati_fis["ascisse_MF_1_i"])
        SoC["basso"]=fuzz.trimf(SoC.universe, dati_fis["ascisse_MF_2_i"])
        SoC["medio"]=fuzz.trimf(SoC.universe, dati_fis["ascisse_MF_3_i"])
        SoC["alto"]=fuzz.trimf(SoC.universe, dati_fis["ascisse_MF_4_i"])
        SoC["molto_alto"]=fuzz.trapmf(SoC.universe, dati_fis["ascisse_MF_5_i"])
        alpha["molto_basso"]=fuzz.trapmf(alpha.universe, dati_fis["ascisse_MF_1_o"])
        alpha["basso"]=fuzz.trimf(alpha.universe, dati_fis["ascisse_MF_2_o"])
        alpha["medio"]=fuzz.trimf(alpha.universe, dati_fis["ascisse_MF_3_o"])
        alpha["alto"]=fuzz.trimf(alpha.universe, dati_fis["ascisse_MF_4_o"])
        alpha["molto_alto"]=fuzz.trapmf(alpha.universe, dati_fis["ascisse_MF_5_o"])
        ascisse_MF_input=[dati_fis["ascisse_MF_1_i"],dati_fis["ascisse_MF_2_i"],
                          dati_fis["ascisse_MF_3_i"],dati_fis["ascisse_MF_4_i"],
                          dati_fis["ascisse_MF_5_i"]]
        ascisse_MF_output=[dati_fis["ascisse_MF_1_o"],dati_fis["ascisse_MF_2_o"],
                          dati_fis["ascisse_MF_3_o"],dati_fis["ascisse_MF_4_o"],
                          dati_fis["ascisse_MF_5_o"]]
        # definisci delle regole fuzzy e impostazione dei valori dei pesi
        indice_regola=0
        fuzzy_set_output="none"
        if dati_fis["regole"][indice_regola][3] == 1:
            fuzzy_set_output="molto_basso"
        if dati_fis["regole"][indice_regola][3] == 2:
            fuzzy_set_output="basso"   
        if dati_fis["regole"][indice_regola][3] == 3:
            fuzzy_set_output="medio"
        if dati_fis["regole"][indice_regola][3] == 4:
            fuzzy_set_output="alto"
        if dati_fis["regole"][indice_regola][3] == 5:
            fuzzy_set_output="molto_alto"            
        regola1 = ctrl.Rule(SoC['molto_basso'], alpha[fuzzy_set_output])
        indice_regola=1
        fuzzy_set_output="none"
        if dati_fis["regole"][indice_regola][3] == 1:
            fuzzy_set_output="molto_basso"
        if dati_fis["regole"][indice_regola][3] == 2:
            fuzzy_set_output="basso"   
        if dati_fis["regole"][indice_regola][3] == 3:
            fuzzy_set_output="medio"
        if dati_fis["regole"][indice_regola][3] == 4:
            fuzzy_set_output="alto"
        if dati_fis["regole"][indice_regola][3] == 5:
            fuzzy_set_output="molto_alto"
        regola2 = ctrl.Rule(SoC['basso'], alpha[fuzzy_set_output])
        indice_regola=2
        fuzzy_set_output="none"
        if dati_fis["regole"][indice_regola][3] == 1:
            fuzzy_set_output="molto_basso"
        if dati_fis["regole"][indice_regola][3] == 2:
            fuzzy_set_output="basso"   
        if dati_fis["regole"][indice_regola][3] == 3:
            fuzzy_set_output="medio"
        if dati_fis["regole"][indice_regola][3] == 4:
            fuzzy_set_output="alto"
        if dati_fis["regole"][indice_regola][3] == 5:
            fuzzy_set_output="molto_alto"
        regola3 = ctrl.Rule(SoC['medio'], alpha[fuzzy_set_output])
        indice_regola=3
        fuzzy_set_output="none"
        if dati_fis["regole"][indice_regola][3] == 1:
            fuzzy_set_output="molto_basso"
        if dati_fis["regole"][indice_regola][3] == 2:
            fuzzy_set_output="basso"   
        if dati_fis["regole"][indice_regola][3] == 3:
            fuzzy_set_output="medio"
        if dati_fis["regole"][indice_regola][3] == 4:
            fuzzy_set_output="alto"
        if dati_fis["regole"][indice_regola][3] == 5:
            fuzzy_set_output="molto_alto"
        regola4 = ctrl.Rule(SoC['alto'], alpha[fuzzy_set_output])
        indice_regola=4
        fuzzy_set_output="none"
        if dati_fis["regole"][indice_regola][3] == 1:
            fuzzy_set_output="molto_basso"
        if dati_fis["regole"][indice_regola][3] == 2:
            fuzzy_set_output="basso"   
        if dati_fis["regole"][indice_regola][3] == 3:
            fuzzy_set_output="medio"
        if dati_fis["regole"][indice_regola][3] == 4:
            fuzzy_set_output="alto"
        if dati_fis["regole"][indice_regola][3] == 5:
            fuzzy_set_output="molto_alto"
        regola5 = ctrl.Rule(SoC['molto_alto'], alpha[fuzzy_set_output])
        # definisci i pesi
        regola1.weight=dati_fis["regole"][0][2]
        regola2.weight=dati_fis["regole"][1][2]
        regola3.weight=dati_fis["regole"][2][2]
        regola4.weight=dati_fis["regole"][3][2]
        regola5.weight=dati_fis["regole"][4][2]
        pesi_regole=[regola1.weight, regola2.weight, regola3.weight, 
                     regola4.weight, regola5.weight]
        conseguenti=[dati_fis["regole"][0][3],
                     dati_fis["regole"][1][3],
                     dati_fis["regole"][2][3],
                     dati_fis["regole"][3][3],
                     dati_fis["regole"][4][3]]
                     
        # crea il fis
        sistema_inferenza = ctrl.ControlSystem([regola1, 
                                            regola2, 
                                            regola3, 
                                            regola4, 
                                            regola5])
        simulatore = ctrl.ControlSystemSimulation(sistema_inferenza)
        # calcola output per ogni MG
        numero_MG=len(dati_MG)
        vettore_decisioni=[0]*numero_MG
        for kk in range(0,numero_MG):
            # assegna l'input
            simulatore.input['SoC'] = vettore_SoC_input[kk]
            # calcola l'output
            simulatore.output['alpha']=0.5
            dummy_variable=0
            try:
                simulatore.compute()
            except:
                dummy_variable=1
                # print('No firing!')
            vettore_decisioni[kk] = round(simulatore.output['alpha'],2)
            matrice_decisioni[kk,timeslot]=vettore_decisioni[kk]
            
            
            
            
        if timeslot==47:
            print('antonino')
        # calcola il valore della fo usando l'output del fis
        # MG 1
        risultati_MG_1=dati_MG[0].simula_microgrid(vettore_decisioni[0],
                                    dataset[0][timeslot],
                                    dataset[7][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_1=risultati_MG_1[0]
        alfa_1=risultati_MG_1[1]
        P_GL_N_1=risultati_MG_1[2]
        p_GL_S_1=risultati_MG_1[3]
        costo_1=risultati_MG_1[4]
        # riordinai dati estratti
        matrice_SoC[0,timeslot]=SoC_MG_1
        matrice_P_GL_S[0,timeslot]= p_GL_S_1
        matrice_P_GL_N[0,timeslot]= P_GL_N_1
        matrice_FO[0,timeslot]= costo_1
        
        # MG 2
        risultati_MG_2=dati_MG[1].simula_microgrid(vettore_decisioni[1],
                                    dataset[1][timeslot],
                                    dataset[8][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_2=risultati_MG_2[0]
        alfa_2=risultati_MG_2[1]
        P_GL_N_2=risultati_MG_2[2]
        p_GL_S_2=risultati_MG_2[3]
        costo_2=risultati_MG_2[4]
        # riordinai dati estratti
        matrice_SoC[1,timeslot]=SoC_MG_2
        matrice_P_GL_S[1,timeslot]= p_GL_S_2
        matrice_P_GL_N[1,timeslot]= P_GL_N_2
        matrice_FO[1,timeslot]= costo_2
        
        # MG 3
        risultati_MG_3=dati_MG[2].simula_microgrid(vettore_decisioni[2],
                                    dataset[2][timeslot],
                                    dataset[9][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_3=risultati_MG_3[0]
        alfa_3=risultati_MG_3[1]
        P_GL_N_3=risultati_MG_3[2]
        p_GL_S_3=risultati_MG_3[3]
        costo_3=risultati_MG_3[4]
        # riordinai dati estratti
        matrice_SoC[2,timeslot]=SoC_MG_3
        matrice_P_GL_S[2,timeslot]= p_GL_S_3
        matrice_P_GL_N[2,timeslot]= P_GL_N_3
        matrice_FO[2,timeslot]= costo_3
        
        # MG 4
        risultati_MG_4=dati_MG[3].simula_microgrid(vettore_decisioni[3],
                                    dataset[3][timeslot],
                                    dataset[10][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_4=risultati_MG_4[0]
        alfa_4=risultati_MG_4[1]
        P_GL_N_4=risultati_MG_4[2]
        p_GL_S_4=risultati_MG_4[3]
        costo_4=risultati_MG_4[4]
        # riordinai dati estratti
        matrice_SoC[3,timeslot]=SoC_MG_4
        matrice_P_GL_S[3,timeslot]= p_GL_S_4
        matrice_P_GL_N[3,timeslot]= P_GL_N_4
        matrice_FO[3,timeslot]= costo_4
        
        # MG 5
        risultati_MG_5=dati_MG[4].simula_microgrid(vettore_decisioni[4],
                                    dataset[4][timeslot],
                                    dataset[11][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_5=risultati_MG_5[0]
        alfa_5=risultati_MG_5[1]
        P_GL_N_5=risultati_MG_5[2]
        p_GL_S_5=risultati_MG_5[3]
        costo_5=risultati_MG_5[4]
        # riordinai dati estratti
        matrice_SoC[4,timeslot]=SoC_MG_5
        matrice_P_GL_S[4,timeslot]= p_GL_S_5
        matrice_P_GL_N[4,timeslot]= P_GL_N_5
        matrice_FO[4,timeslot]= costo_5
        
        # MG 6
        risultati_MG_6=dati_MG[5].simula_microgrid(vettore_decisioni[5],
                                    dataset[5][timeslot],
                                    dataset[12][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_6=risultati_MG_6[0]
        alfa_6=risultati_MG_6[1]
        P_GL_N_6=risultati_MG_6[2]
        p_GL_S_6=risultati_MG_6[3]
        costo_6=risultati_MG_6[4]
        # riordinai dati estratti
        matrice_SoC[5,timeslot]=SoC_MG_6
        matrice_P_GL_S[5,timeslot]= p_GL_S_6
        matrice_P_GL_N[5,timeslot]= P_GL_N_6
        matrice_FO[5,timeslot]= costo_6
        
        # MG 7
        risultati_MG_7=dati_MG[6].simula_microgrid(vettore_decisioni[6],
                                    dataset[6][timeslot],
                                    dataset[13][timeslot],
                                    0.25,
                                    96,
                                    'no')
        SoC_MG_7=risultati_MG_7[0]
        alfa_7=risultati_MG_7[1]
        P_GL_N_7=risultati_MG_7[2]
        p_GL_S_7=risultati_MG_7[3]
        costo_7=risultati_MG_7[4]
        # riordinai dati estratti
        matrice_SoC[6,timeslot]=SoC_MG_7
        matrice_P_GL_S[6,timeslot]= p_GL_S_7
        matrice_P_GL_N[6,timeslot]= P_GL_N_7
        matrice_FO[6,timeslot]= costo_7
        
        

            
        FO_globale_energy_community = costo_1+costo_2+costo_3+costo_4+costo_5+costo_6+costo_7
        somma_FO_globale_energy_community=somma_FO_globale_energy_community+ FO_globale_energy_community
        vettore_FO[timeslot]=FO_globale_energy_community
        
    
    # azzeramento della Soc alla fine della simulazione
    dati_MG[0].SoC = dati_MG[0].SoC_0
    dati_MG[1].SoC = dati_MG[1].SoC_0
    dati_MG[2].SoC = dati_MG[2].SoC_0
    dati_MG[3].SoC = dati_MG[3].SoC_0
    dati_MG[4].SoC = dati_MG[4].SoC_0
    dati_MG[5].SoC = dati_MG[5].SoC_0
    dati_MG[6].SoC = dati_MG[6].SoC_0
    dati_MG[0].SoC_autoconsumo = dati_MG[0].SoC_0
    dati_MG[1].SoC_autoconsumo  = dati_MG[1].SoC_0
    dati_MG[2].SoC_autoconsumo  = dati_MG[2].SoC_0
    dati_MG[3].SoC_autoconsumo  = dati_MG[3].SoC_0
    dati_MG[4].SoC_autoconsumo  = dati_MG[4].SoC_0
    dati_MG[5].SoC_autoconsumo  = dati_MG[5].SoC_0
    dati_MG[6].SoC_autoconsumo  = dati_MG[6].SoC_0
    
    fis=sistema_inferenza
    
    # End timer
    end_time = time.time()
    # Calcola il tempo di esecuzione
    elapsed_time = end_time - start_time
    
    risultati_finali=[somma_FO_globale_energy_community,
                      matrice_decisioni,
                      matrice_SoC,
                      matrice_P_GL_S,
                      matrice_P_GL_N,
                      matrice_FO,
                      elapsed_time,
                      fis, 
                      ascisse_MF_input,
                      ascisse_MF_output,
                      conseguenti,
                      pesi_regole,
                      somma_FO_autoconsumo_energy_community,
                      vettore_FO
                      ]
    

    
    return risultati_finali


