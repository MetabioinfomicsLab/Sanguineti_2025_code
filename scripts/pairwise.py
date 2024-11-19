import numpy as np
import pandas as pd
import micom
import multiprocessing as mp


def chunks(l,nproc):
    right_div=len(l)//nproc
    nmore=len(l)%nproc
    return [l[i*(right_div+1):i*(right_div+1)+right_div+1] for i in range(nmore)]+[l[nmore*(right_div+1)+i*right_div:nmore*(right_div+1)+i*right_div+right_div] for i in range(nproc-nmore) if nmore<len(l)]

def main(spps): # this function implements the 
    allres=[]
    for spp2 in spps:
        model=micom.load_pickle('models/'+spp2[0]+'.'+spp2[1]+".pickle")
        tog1,tog2,alone1,alone2=0,0,0,0
        times=1000 # number of iterations per pair of models
        for u in range(times):
            model.medium={x.id:np.random.random() for x in model.exchanges}
            with model:
                model.reactions.get_by_id('bio1__'+spp2[1]).bounds=(0,0) # this enforces the maximization of only one species' growth
                model.optimize()
                tog1+=model.reactions.get_by_id('bio1__'+spp2[0]).flux
            with model:
                model.reactions.get_by_id('bio1__'+spp2[0]).bounds=(0,0)
                model.optimize()
                tog2+=model.reactions.get_by_id('bio1__'+spp2[1]).flux
            with model:
                for r in model.reactions:
                    if r.id.startswith('EX_') and r.id.endswith(spp2[1]): r.bounds=(0,0) # shut down all exchanges from the other species
                model.optimize()
                alone1+=model.reactions.get_by_id('bio1__'+spp2[0]).flux
            with model:
                for r in model.reactions:
                    if r.id.startswith('EX_') and r.id.endswith(spp2[0]): r.bounds=(0,0)
                model.optimize()
                alone2+=model.reactions.get_by_id('bio1__'+spp2[1]).flux
        allres.append([alone1/times,alone2/times,tog1/times,tog2/times])
    return allres

if __name__=='__main__':
    # to change the number of iterations, change the corresponding parameter in function main()
    spp=['Saccharicenans_sp_','Pseudothermotoga_B_sp_','Methanothermobacter_marburgensis_1','Methanothermobacter_thermautotrophicus_']
    nproc=6
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
    df.to_csv('pairwise_results.tsv',sep='\t') 
    """
    in the resulting dataframe, the value at row i and column j 
    displays average maximum growth rate of species i 
    when growing with species j and maximizing for growth of species i
    """