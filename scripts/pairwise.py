import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import micom
import micomapi
import cobra
import multiprocessing as mp

def get_medium_files(folder_pref):
    folders=[f'{folder_pref}_{x}_fd' for x in ['4h','3h','2h','1h','45min','30min','25min']]
    medfiles=[]
    for fold in folders:
        heres=[x for x in os.listdir(fold) if x.startswith('medium_')]
        ordered=sorted(heres,key=lambda x:int(x.replace('medium_','').replace('.tsv','')))
        medfiles.append(fold+'/'+ordered[-1])
    return medfiles

def get_taxdict():
    d,count={},{}
    for x in open('elgotaxdict.txt'):
        mag,sp=x.strip().split('\t')[0],x.strip().split('\t')[1]
        if sp in count: n=count[sp]; count[sp]+=1
        else: n=''; count[sp]=1
        d[mag]=sp+'_'+str(n)
        d[mag]=d[mag].replace('.','').replace(' ','_')
    return d

def chunks(l,nproc):
    right_div=len(l)//nproc
    nmore=len(l)%nproc
    return [l[i*(right_div+1):i*(right_div+1)+right_div+1] for i in range(nmore)]+[l[nmore*(right_div+1)+i*right_div:nmore*(right_div+1)+i*right_div+right_div] for i in range(nproc-nmore) if nmore<len(l)]


makepairwise=True
dopairwise=False


if makepairwise:
    #spp=['Methanothermobacter_marburgensis_1', 'Bacillota_G_sp_3', 'Aggregatilineales_sp_', 'Bacillota_G_sp_2', 'Caldanaerobacter_subterraneus_', 'Bellilinea_sp_', 'Lutisporaceae_sp_', 'Bacteroidales_sp_', 'Kapabacteriales_sp_', 'Saccharicenans_sp_', 'Acetomicrobium_sp_', 'Anaerolinea_thermophila_', 'Tepidiphilus_sp_', 'Limnochordales_sp_1', 'Methanothermobacter_wolfei_', 'Acetivibrionales_sp_1', 'Proteinivoracia_sp_', 'Bacillota_E_sp_4', 'Pseudothermotoga_B_sp_', 'Sphaerobacter_thermophilus_', 'Tepidiphilus_succinatimandens_', 'Mycobacterium_sp_1', 'Coprothermobacter_proteolyticus_', 'Methanothermobacter_thermautotrophicus_']
    spp=['Saccharicenans_sp_','Pseudothermotoga_B_sp_','Methanothermobacter_marburgensis_1','Methanothermobacter_thermautotrophicus_']
    nproc=100
    def main(spps):
        for spp2 in spps:
            d={'id':[],'species':[],'file':[],'sample_id':[],'abundance':[]}
            for sp in spp2:
                d['id'].append(sp)
                d['species'].append(sp)
                d['sample_id'].append('None')
                #d['file'].append(modfold+'/'+'.'.join(i.split('.')[:-1])+'.xml')
                d['file'].append('renamed_models/'+sp+'.xml')
                d['abundance'].append(0.5)
            formicom=pd.DataFrame.from_dict(d)
            com=micom.Community(formicom,rel_threshold=0.001,solver='cplex')
            com.to_pickle('pairwise_models/'+spp2[0]+'.'+spp2[1]+".pickle")
    all_spps=[]
    for i in range(len(spp)):
        for j in range(i+1,len(spp)):
            all_spps.append([spp[i],spp[j]])
    chunked=chunks(all_spps,nproc)
    with mp.Pool(nproc) as pool:
        pool.map(main,chunked)

if dopairwise:
    spp=['Methanothermobacter_marburgensis_1', 'Bacillota_G_sp_3', 'Aggregatilineales_sp_', 'Bacillota_G_sp_2', 'Caldanaerobacter_subterraneus_', 'Bellilinea_sp_', 'Lutisporaceae_sp_', 'Bacteroidales_sp_', 'Kapabacteriales_sp_', 'Saccharicenans_sp_', 'Acetomicrobium_sp_', 'Anaerolinea_thermophila_', 'Tepidiphilus_sp_', 'Limnochordales_sp_1', 'Methanothermobacter_wolfei_', 'Acetivibrionales_sp_1', 'Proteinivoracia_sp_', 'Bacillota_E_sp_4', 'Pseudothermotoga_B_sp_', 'Sphaerobacter_thermophilus_', 'Tepidiphilus_succinatimandens_', 'Mycobacterium_sp_1', 'Coprothermobacter_proteolyticus_', 'Methanothermobacter_thermautotrophicus_']
    #spp=['Saccharicenans_sp_','Pseudothermotoga_B_sp_','Methanothermobacter_marburgensis_1','Methanothermobacter_thermautotrophicus_']
    nproc=100
    def main(spps):
        allres=[]
        for spp2 in spps:
            model=micom.load_pickle('pairwise_models/'+spp2[0]+'.'+spp2[1]+".pickle")
            tog1,tog2,alone1,alone2=0,0,0,0
            times=1000
            for u in range(times):
                model.medium={x.id:np.random.random() for x in model.exchanges}
                with model:
                    model.reactions.get_by_id('bio1__'+spp2[1]).bounds=(0,0)
                    model.optimize()
                    tog1+=model.reactions.get_by_id('bio1__'+spp2[0]).flux
                with model:
                    model.reactions.get_by_id('bio1__'+spp2[0]).bounds=(0,0)
                    model.optimize()
                    tog2+=model.reactions.get_by_id('bio1__'+spp2[1]).flux
                with model:
                    for r in model.reactions:
                        if r.id.startswith('EX_') and r.id.endswith(spp2[1]): r.bounds=(0,0)
                    #model.reactions.get_by_id('bio1__'+spp[j]).bounds=(0,0)
                    model.optimize()
                    alone1+=model.reactions.get_by_id('bio1__'+spp2[0]).flux
                with model:
                    for r in model.reactions:
                        if r.id.startswith('EX_') and r.id.endswith(spp2[0]): r.bounds=(0,0)
                    #model.reactions.get_by_id('bio1__'+spp[i]).bounds=(0,0)
                    model.optimize()
                    alone2+=model.reactions.get_by_id('bio1__'+spp2[1]).flux
            allres.append([alone1/times,alone2/times,tog1/times,tog2/times])
        return allres
    all_spps=[]
    for i in range(len(spp)):
        for j in range(i+1,len(spp)):
            all_spps.append([spp[i],spp[j]])
    chunked=chunks(all_spps,nproc)
    with mp.Pool(nproc) as pool:
        results=pool.map(main,chunked)
    matrix=np.zeros((len(spp),len(spp)))
    sp_index={spp[i]:i for i in range(len(spp))}
    for sub_ch,sub_res in zip(chunked,results):
        for pair,res in zip(sub_ch,sub_res):
            i1,i2=sp_index[pair[0]],sp_index[pair[1]]
            matrix[i1][i1]+=(1/(len(spp)-1)*res[0])
            matrix[i2][i2]+=(1/(len(spp)-1)*res[1])
            matrix[i1][i2]=res[2]
            matrix[i2][i1]=res[3]
    df=pd.DataFrame(data=matrix,index=spp,columns=spp)
    df.to_csv('all_pairwise_1000.tsv',sep='\t')