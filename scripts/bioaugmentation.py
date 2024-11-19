import os
import pandas as pd
import micom
import cobra
import multiprocessing as mp
from optlang.symbolics import Zero
from functools import partial

def weighted_regularize_l2_norm(community, min_growth, beta): # function from micom modified to account for beta parameter
    l2 = Zero
    community.variables.community_objective.lb = min_growth
    context = micom.util.get_context(community)
    if context is not None:
        context(partial(micom.util.reset_min_community_growth, community))
    for sp in community.taxa:
        taxa_obj = community.constraints["objective_" + sp]
        ab=community.taxonomy.loc[sp,'abundance']
        ex = sum(v for v in taxa_obj.variables if (v.ub - v.lb) > 1e-6)
        if not isinstance(ex, int):
            l2 += (community.scale * ((ab**beta*ex) ** 2)).expand()
    community.objective = -l2
    community.modification = "l2 regularization"

def inverse_ct(model,fraction=1.0,beta=1.0,fluxes=False,pfba=False,reactions=None,grates=False): # function from micom modified to account for beta parameter
    maxcgr=model.optimize().objective_value
    with model:
        weighted_regularize_l2_norm(model,maxcgr*fraction,beta)
        if reactions is not None:
           s=model.optimize(pfba=pfba)
           return {rid:model.reactions.get_by_id(rid).flux for rid in reactions},s.members if grates else {rid:model.reactions.get_by_id(rid).flux for rid in reactions}
        s=model.optimize(fluxes=fluxes,pfba=pfba)
    return s # return a dictionary reaction_id:flux_value


def multiproc_model_to_pickle(modfold,ra_tsv,solver='cplex',rathresh=1e-3,prefix=''): 
    """function to parallelize micom model generation across samples"""
    df=pd.read_csv(ra_tsv,sep='\t',index_col=0) 
    def main(modfold,df,sample,solver,rathresh,prefix):                        
        d={'id':[],'species':[],'file':[],'sample_id':[],'abundance':[]}  
        for i in df.index:
            d['id'].append(i)
            d['species'].append(i)
            d['sample_id'].append(sample)
            d['file'].append(modfold+'/'+i+'.xml')
        for i in df[sample]:
            d['abundance'].append(float(i))
        formicom=pd.DataFrame.from_dict(d)
        com=micom.Community(formicom,rel_threshold=rathresh,solver=solver)
        com.to_pickle(prefix+sample+".pickle")
    jobs=[]
    for sample in df:
        jobs.append(mp.Process(target=main,args=(modfold,df,sample,solver,rathresh,prefix)))
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()


# set this flag to True to run the corresponding commands
make_models=False
solve_models=False

who=['Methanosuratincola_petrocarbonis_isnew'] # prefix of the .xml model file of the (one or more) species to introduce in the community models
folderId='petrocarbonis' # prefix of the output folder

if make_models: # generate community models with bioaugmented species
    fraction=0.2 # relative abundance to be taken from original community and to be distributed among the introduced species
    os.system("mkdir "+folderId+'_models'+'_'+str(fraction))
    formic=pd.read_csv('data/relative_abundances.csv',index_col=0,sep=',')
    decrease=formic*fraction
    if who[0] in formic.index: decrease.loc[who[0]]*=0.0
    formic-=decrease # take abundance from community members
    for w in who: # equally distribute abundance to added species 
        if w not in formic.index: formic.loc[w]=(decrease.sum())/len(who)
        else: formic.loc[w]+=decrease.sum()/len(who)
    formic.to_csv(folderId+'_models'+'_'+str(fraction)+'/formicom.tsv',sep='\t')
    # mind that the number of processes that are spawned is 7 
    multiproc_model_to_pickle('renamed_models/',folderId+'_models'+'_'+str(fraction)+'/formicom.tsv',solver='cplex',rathresh=1e-8,prefix=folderId+'_models'+'_'+str(fraction)+'/')
    allg=['TBR1_UP_'+x for x in ['4h','3h','2h','1h','45min','30min','25min']]
    dfvals=pd.read_csv('data/gasses.csv',index_col=0,sep=',')
    for n,grt in enumerate(allg): # here, for each generated model, add OUT_* reactions, set import gas constraints, and save model to pickle
        modelfile=folderId+'_models'+'_'+str(fraction)+'/'+grt+'.pickle'
        model=micom.load_pickle(modelfile)
        outh2=cobra.Reaction('OUT_cpd11640_m')
        outh2.add_metabolites({model.metabolites.cpd11640_m:-1})
        outh2.bounds=(0,1000)
        outh2.global_id='OUT_cpd11640_m'
        outh2.community_id='medium'
        outco2=cobra.Reaction('OUT_cpd00011_m')
        outco2.add_metabolites({model.metabolites.cpd00011_m:-1})
        outco2.bounds=(0,1000)
        outco2.global_id='OUT_cpd00011_m'
        outco2.community_id='medium'
        model.add_reactions([outco2,outh2])
        model.reactions.EX_cpd00011_m.bounds=(dfvals[grt]['cpd00011'],dfvals[grt]['cpd00011'])
        model.reactions.EX_cpd11640_m.bounds=(dfvals[grt]['cpd11640'],dfvals[grt]['cpd11640'])
        model.to_pickle(modelfile)

if solve_models: # solve generated models
    def main(fraction):
        allmets=[]
        for n,grt in enumerate(allg):
            if not fraction: model=micom.load_pickle('nozeros_models/'+grt+'.pickle')
            else: model=micom.load_pickle(folderId+'_models'+'_'+str(fraction)+'/'+grt+'.pickle')
            medium={line.split('\t')[0]:float(line.strip().split('\t')[1]) for line in open(medium_files[n]) if not line.startswith('compound_code')}
            medium['EX_cpd00001_m']=1000
            medium['EX_cpd11640_m']=abs(model.reactions.EX_cpd11640_m.bounds[0])
            medium['EX_cpd00011_m']=abs(model.reactions.EX_cpd00011_m.bounds[0])
            model.medium=medium
            if grt=='TBR1_UP_4h': alpha=0.6 # specific alpha parameter for first conidition. 
            else: alpha=0.8
            s,grates=inverse_ct(model,fraction=alpha,beta=beta,pfba=True,reactions=['EX_cpd01024_m','OUT_cpd00011_m','OUT_cpd11640_m','EX_cpd00029_m'],grates=True)
            s,grates=inverse_ct(model,fraction=alpha,beta=beta,pfba=True,reactions=['EX_cpd01024_m','OUT_cpd00011_m','OUT_cpd11640_m','EX_cpd00029_m'],grates=True)
            for sp in grates.index:
                if sp=='medium': continue
                model.reactions.get_by_id('bio1__'+sp).bounds=(grates['growth_rate'][sp],grates['growth_rate'][sp]) # fix growth rates
            model.objective={model.reactions.EX_cpd01024_m:1} # set methane as objective
            model.objective_direction='max'
            model.optimize(pfba=True)
            allmets.append(str(round(model.reactions.EX_cpd01024_m.flux,2)))
        return allmets
    
    alpha=0.8; beta=0.8
    fractions=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    medium_files=media=['media/medium_4h.tsv', 'media/medium_3h.tsv', 'media/medium_2h.tsv', 'media/medium_1h.tsv', 'media/medium_45min.tsv', 'media/medium_30min.tsv', 'media/medium_25min.tsv']
    allg=['TBR1_UP_'+x for x in ['4h','3h','2h','1h','45min','30min','25min']]
    shortg=['4h','3h','2h','1h','45min','30min','25min']
    with mp.Pool(len(fractions)) as pool:
        totmets=pool.map(main,fractions)
    df=pd.DataFrame(data=totmets,index=[str(x) for x in fractions],columns=shortg)
    df.to_csv(folderId+'_dataframe.csv')  

