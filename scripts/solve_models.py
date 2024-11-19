import micom
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

models=['models/TBR1_UP_4h.pickle', 'models/TBR1_UP_3h.pickle', 'models/TBR1_UP_2h.pickle', 'models/TBR1_UP_1h.pickle', 'models/TBR1_UP_45min.pickle', 'models/TBR1_UP_30min.pickle', 'models/TBR1_UP_25min.pickle']
media=['media/medium_4h.tsv', 'media/medium_3h.tsv', 'media/medium_2h.tsv', 'media/medium_1h.tsv', 'media/medium_45min.tsv', 'media/medium_30min.tsv', 'media/medium_25min.tsv']
alphas=[0.6,0.8,0.8,0.8,0.8,0.8,0.8] # alpha and beta parameters for each corresponding GRT model
betas=[0.8,0.8,0.8,0.8,0.8,0.8,0.8]
grts=['4h','3h','2h','1h','45min','30min','25min']

for model,medium,alpha,beta,grt in zip(models,media,alphas,betas,grts): # iterate over each GRT model and corresponding medium and parameters
    model=micom.load_pickle(model) # the model is already constrained in H2 and CO2 import
    medium={line.split('\t')[0]:float(line.strip().split('\t')[1]) for line in open(medium)}
    medium['EX_cpd00001_m']=1000 # water can be uptaken
    medium['EX_cpd11640_m']=abs(model.reactions.EX_cpd11640_m.bounds[0]) # to avoid error in reaction bounds
    medium['EX_cpd00011_m']=abs(model.reactions.EX_cpd00011_m.bounds[0]) # to avoid error in reaction bounds
    model.medium=medium
    s,grates=inverse_ct(model,fraction=alpha,beta=beta,pfba=True,reactions=['EX_cpd00001_m'],grates=True) # the first optimization results in slightly different values than the second
    s,grates=inverse_ct(model,fraction=alpha,beta=beta,pfba=True,reactions=[x.id for x in model.reactions],grates=True) # thus the results of the second optimization are those that are kept
    # save growth rates and fluxes
    grates.to_csv('growth_rates_'+grt+'.csv')
    new=open('fluxes_'+grt+'.csv','w')
    print('reaction_id,flux',file=new)
    for rid in s:
        print(rid+','+str(s[rid]),file=new)
    new.close()    
