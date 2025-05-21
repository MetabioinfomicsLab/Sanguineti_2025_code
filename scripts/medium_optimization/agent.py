from torch import tensor,unsqueeze,full,diag
from torch.distributions import Normal,MultivariateNormal
from math import isnan




from optlang.symbolics import Zero
from functools import partial
import micom
### OPTIMIZATION FUNCTIONS ###
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
    s=model.optimize()
    if s is None: return None
    maxcgr=s.objective_value
    with model:
        weighted_regularize_l2_norm(model,maxcgr*fraction,beta)
        if reactions is not None:
           s=model.optimize(pfba=pfba)
           return {rid:model.reactions.get_by_id(rid).flux for rid in reactions},s.members if grates else {rid:model.reactions.get_by_id(rid).flux for rid in reactions}
        s=model.optimize(fluxes=fluxes,pfba=pfba)
    return s # return a dictionary reaction_id:flux_value

### COMPATIBILTY CLASSES ###
class fake_layer: # the actual medium values
    def __init__(self,medinit):
        self.weight=medinit
        self.bias=tensor([])
class fake_policy: # the "policy", i.e. it stores medium values and reaction ID
    def __init__(self,rids,medinit):
        self.rids=rids
        self.layers=[fake_layer(medinit)]
    def get_shape(self):
        return ([tuple(layer.weight.size()) for layer in self.layers], 
        [tuple(layer.bias.size()) for layer in self.layers]) 
class multiobj_medopt_agent: # the "agent", i.e. has a policy that is the medium, performs the modified cooperativoe tradeoff optimization and returns values of interest
    def __init__(self,rids,target,model,loss,alpha,beta,medinit):
        self.rids=rids
        self.target=target
        self.model_file=model
        self.loss=loss
        self.alpha=alpha
        self.beta=beta
        self.policy=fake_policy(rids,medinit)
        self.name='agent'
    def initialize(self):
        self.model=micom.load_pickle(self.model_file)
        self.get_action() # just to set the solver to a reproducible state
    def get_action(self):
        apply_medium={x:max(0.0,val)  for x,val in zip(self.rids,self.policy.layers[0].weight)} # get medium values from "policy"
        values=[]
        apply_medium['EX_cpd00001_m']=1000 # water is unlimited
        apply_medium['EX_cpd11640_m']=abs(self.model.reactions.EX_cpd11640_m.bounds[0]) # set h2 to avoid error coming up given the fixed bound
        apply_medium['EX_cpd00011_m']=abs(self.model.reactions.EX_cpd00011_m.bounds[0]) # set co2 to avoid error coming up given the fixed bound
        self.model.medium=apply_medium        
        s,grates=inverse_ct(self.model,fraction=self.alpha,beta=self.beta,pfba=True,grates=True,reactions=['EX_cpd01024_m','OUT_cpd00011_m','OUT_cpd11640_m','EX_cpd00029_m'])
        if s is None: 
            values.append(None)
            self.values=values
            print('INFEASIBLE')
            #for rid,val in self.model.medium.items():
            #    print(rid,val)
            return -50.0
        else: 
            mingrowth=1e-3
            c=0
            for i in grates.index:
                if i!='medium':
                    if grates['growth_rate'][i]<mingrowth: c+=1
            underthresh=c/(len(grates.index)-1)
            self.members=grates
            values.append(abs(s['EX_cpd01024_m']/(s['EX_cpd01024_m']+s['OUT_cpd00011_m']+s['OUT_cpd11640_m']))) # methane purity
            values.append(s['EX_cpd00029_m']) # acetate exchange
            values.append(s['EX_cpd01024_m']) # methane exchange
            self.topr=[abs(x-y) for x,y in zip(self.target,values)]
            self.vals=[x for x in values]+[underthresh] 
            return self.vals # return values of interest to be ranked and combined by the optimizer
class fake_env: # "environment" that return values of interest
    def __init__(self,agents):
        self.agents=agents
    def run_batched_episode(self,something,somethingelse,printstats=False,printtest=False):
        #return tensor([self.agents[0].get_action()]).float()
        return self.agents[0].get_action()
