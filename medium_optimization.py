import numpy as np
import ray
import copy
import time
import micom
import os
import matplotlib.pyplot as plt
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


def MSE(pred,real):
    return np.mean([(x-y)**2 for x,y in zip(pred,real)])
def MAE(pred,real):
    return np.mean([abs(x-y) for x,y in zip(pred,real)])
def weighted_MAE(pred,real):
    ws=[0.6,0.4] if len(pred)>1 else [1.0]
    return np.sum([w*abs(x-y) for x,y,w in zip(pred,real,ws)])


@ray.remote
class ray_container:
    def __init__(self,medium,target,model,which,loss,alpha,beta,l2_decay=0.0,printer=False):
        self.medium=copy.deepcopy(medium) # il medium deve includere h20 e i constraints, che non saranno mai toccati
        self.target=target
        self.model=micom.load_pickle(model)
        self.which=which
        self.loss=loss
        self.alpha=alpha
        self.beta=beta
        self.printer=printer
        self.l2_decay=l2_decay
    def get_loss(self,change):
        self.medium[self.which]+=change
        purities=[]
        acetate=[]
        apply_medium=copy.deepcopy(self.medium)
        apply_medium['EX_cpd00001_m']=1000
        apply_medium['EX_cpd11640_m']=abs(self.model.reactions.EX_cpd11640_m.bounds[0])
        apply_medium['EX_cpd00011_m']=abs(self.model.reactions.EX_cpd00011_m.bounds[0])
        self.model.medium=apply_medium
        if self.printer: print('Starting optimization')
        try:
            s=inverse_ct(self.model,fraction=self.alpha,beta=self.beta,pfba=True,reactions=['EX_cpd01024_m','OUT_cpd00011_m','OUT_cpd11640_m','EX_cpd00029_m'])[0]
        except Exception as e:
            if self.printer: print('Exception:',e)
            s=None
        if self.printer: print('Ending optimization')
        if s is None: purities.append(None)
        else: 
            purities.append(abs(s['EX_cpd01024_m']/(s['EX_cpd01024_m']+s['OUT_cpd00011_m']+s['OUT_cpd11640_m'])))
            if len(self.target)>1:
                purities.append(s['EX_cpd00029_m'])
            acetate.append(abs(s['EX_cpd00029_m']))
        if any([x is None for x in purities]): return None
        else: return self.loss(purities,self.target) 
    def step(self,newmed):
        self.medium=copy.deepcopy(newmed)

class finite_difference:
    def __init__(self,medium,target,model,delta,lrate,loss,alpha,beta,l2_decay=0.0):
        self.containers=[[ray_container.remote(medium,target,model,rid,loss,alpha,beta,l2_decay,printer=False),ray_container.remote(medium,target,model,rid,loss,alpha,beta,l2_decay)] for rid in medium]
        self.medium=medium
        self.previous_medium=copy.deepcopy(self.medium) 
        self.target=target
        self.model=micom.load_pickle(model)
        self.delta=delta
        self.lrate=lrate
        self.loss=loss
        self.ltol=1e-5 # if delta loss is lower than 1e-5, then it is done as if delta loss = zero
        self.mainFitTraj=[]
        self.childFitTraj=[]
        self.velocities={x:0 for x in self.medium}
        self.rmsp={x:0 for x in self.medium}
        self.b1=0.9
        self.b2=0.999
        self.alpha=alpha
        self.beta=beta
        self.l2_decay=l2_decay
    def _make_changes(self):
        changes=[([-self.medium[rid]*self.delta,self.medium[rid]*self.delta] if self.medium[rid]>0 else [0,self.delta]) for rid in self.medium] # [...,[-,+],...]
        return changes
    def _compute_gradients(self,result): # result has the same shape of changes and of self.containers
        gradients={}
        for res,rid in zip(result,self.medium.keys()):
            minus_loss,plus_loss=res
            delta_loss=plus_loss-minus_loss if not any([minus_loss is None,plus_loss is None]) else None
            # if both solutions are infeasible increase the value
            if all([minus_loss is None,plus_loss is None]): gradients[rid]='up'
            # if only one solution is infeasible do nothing
            elif any([minus_loss is None,plus_loss is None]): gradients[rid]='stay'
            # if both solution are equal decrease the value
            elif abs(delta_loss)<self.ltol: gradients[rid]='stay' # it was 'down', change to stay
            # in all other cases, compute the gradient
            else:
                increment=self.medium[rid]*self.delta if self.medium[rid]>0 else self.ltol
                gradient=delta_loss/(2*increment)
                if np.isnan(gradient): gradient=increment/self.lrate if delta_loss>0 else -increment/self.lrate # numerical stability
                gradients[rid]=gradient
        return gradients
    def step(self,gradients): # standard gradient descent
        for rid in gradients:
            gradient=gradients[rid]
            if gradient=='up':self.medium[rid]=self.medium[rid]/(1-self.delta)
            elif gradient=='stay': pass
            elif gradient=='down': self.medium[rid]=self.medium[rid]-self.medium[rid]*self.delta
            else: self.medium[rid]=self.medium[rid]-self.lrate*gradient
        for rid in self.medium: self.medium[rid]=max(0,self.medium[rid])
        res_ids=[[pair[0].step.remote(self.medium),pair[1].step.remote(self.medium)] for pair in self.containers]
        [[ray.get(pair[0]),ray.get(pair[1])] for pair in res_ids] # solo per aspettare
    def adam_step(self,gradients): # adam optimization
        for rid in gradients:
            gradient=gradients[rid]
            if gradient=='up':
                self.velocities[rid]=self.b1*self.velocities[rid]+(1-self.b1)*-1*(self.medium[rid]/(1-self.delta)-self.medium[rid])/self.lrate # mimicking gradient
                self.rmsp[rid]=self.b2*self.rmsp[rid]+(1-self.b2)*((self.medium[rid]/(1-self.delta)-self.medium[rid])/self.lrate)**2 # mimicking gradient
            elif gradient=='stay': pass
            elif gradient=='down':
                self.velocities[rid]=self.b1*self.velocities[rid]+(1-self.b1)*-1*(self.medium[rid]-self.medium[rid]*self.delta-self.medium[rid])/self.lrate # mimicking gradient
                self.rmsp[rid]=self.b2*self.rmsp[rid]+(1-self.b2)*((self.medium[rid]-self.medium[rid]*self.delta-self.medium[rid])/self.lrate)**2 # mimicking gradient
            else:
                self.velocities[rid]=self.b1*self.velocities[rid]+(1-self.b1)*gradient
                self.rmsp[rid]=self.b2*self.rmsp[rid]+(1-self.b2)*gradient**2
            self.medium[rid]=self.medium[rid]-self.lrate*((self.velocities[rid]/(1-self.b1))/(np.sqrt(self.rmsp[rid]/(1-self.b2))+1e-7))
        for rid in self.medium: self.medium[rid]=max(0,self.medium[rid])
        res_ids=[[pair[0].step.remote(self.medium),pair[1].step.remote(self.medium)] for pair in self.containers]
        [[ray.get(pair[0]),ray.get(pair[1])] for pair in res_ids] # solo per aspettare
    def train(self,generations,test_every=tuple(),checkpoint_every=tuple(),lrate_decay=tuple(),starting_epoch=0):
        testRew=None
        for gen in range(starting_epoch+1,starting_epoch+generations+1):
            now=time.time()
            changes=self._make_changes()
            res_ids=[[self.containers[i][0].get_loss.remote(changes[i][0]),self.containers[i][1].get_loss.remote(changes[i][1])] for i in range(len(self.medium))]
            testRew=self.test()
            self.previous_medium=copy.deepcopy(self.medium) 
            self.mainFitTraj.append(testRew)
            result=[[ray.get(pair[0]),ray.get(pair[1])] for pair in res_ids]
            if isinstance(testRew,float): 
                every,outfold=checkpoint_every
                if len(self.topr)>1: # if optimizing both for methane and acetate, terminate when reaching these conditions
                    if self.topr[0]<0.004 and 0.005<self.vals[1]<0.007: self.checkpoint(outfold,gen); print(self.topr); break
                else: # if optimizing only for acetate
                    if self.topr[0]<0.003: self.checkpoint(outfold,gen); print(self.topr); break
            gradients=self._compute_gradients(result)
            self.adam_step(gradients)
            child_values=[x for y in result for x in y if x!=None]
            if child_values: self.childFitTraj.append(np.mean(child_values))
            childRew=np.mean(child_values) if child_values else None      
            if checkpoint_every:
                every,outfold=checkpoint_every
                if not gen%every: 
                    self.checkpoint(outfold,gen)
                    if isinstance(testRew,float):
                        pass
            print(f"Generation: {gen} time: {round(time.time()-now,4)} childMeanFit: {round(childRew,6)} mainFit: {round(testRew,6)} delta: {round(self.delta,6)} lrate: {round(self.lrate,6)} reg_scaled: {round(self.l2_decay*np.mean([self.medium[x]**2 for x in self.medium if x not in ['EX_cpd00013_m','EX_cpd00099_m','EX_cpd00971_m','EX_cpd00254_m','EX_cpd00063_m','EX_cpd00205_m','EX_cpd00009_m','EX_cpd10515_m','EX_cpd00034_m','EX_cpd00058_m','EX_cpd00030_m','EX_cpd11574_m','EX_cpd00149_m','EX_cpd00244_m','EX_cpd00048_m']]),6)} spec diffs: {[round(x,6) for x in self.topr]}")
    def test(self):
        values=[]
        apply_medium=copy.deepcopy(self.medium)
        apply_medium['EX_cpd00001_m']=1000
        apply_medium['EX_cpd11640_m']=abs(self.model.reactions.EX_cpd11640_m.bounds[0])
        apply_medium['EX_cpd00011_m']=abs(self.model.reactions.EX_cpd00011_m.bounds[0])
        self.model.medium=apply_medium        
        s=inverse_ct(self.model,fraction=self.alpha,beta=self.beta,pfba=True,reactions=['EX_cpd01024_m','OUT_cpd00011_m','OUT_cpd11640_m','EX_cpd00029_m'])[0]
        if s is None: values.append(None)
        else: 
            values.append(abs(s['EX_cpd01024_m']/(s['EX_cpd01024_m']+s['OUT_cpd00011_m']+s['OUT_cpd11640_m'])))
            if len(self.target)>1:
                #values.append(0 if s['EX_cpd00029_m']<0.05 else s['EX_cpd00029_m'])
                values.append(s['EX_cpd00029_m'])
            self.topr=[abs(x-y) for x,y in zip(self.target,values)]
            self.vals=[x for x in values]
        if any([x is None for x in values]): return None
        else: 
            return self.loss(values,self.target)
    def checkpoint(self,outfolder,gen):
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        new=open(outfolder+'/info.txt','w')
        print('generation,'+str(gen),file=new)
        print('delta,'+str(self.delta),file=new)
        print('lrate,'+str(self.lrate),file=new)
        new.close()
        k=50
        k_test=1
        new=open(outfolder+'/medium_'+str(gen)+'.tsv','w')
        for x in self.medium:
            print(x+'\t'+str(self.medium[x]),file=new)
        new.close()
        plt.plot([np.mean(self.childFitTraj[i:i+k]) for i in range(len(self.childFitTraj)-k+1)]); plt.savefig(outfolder+'/child'+str(gen)+'.png'); plt.clf() 
        if self.mainFitTraj: plt.plot([np.mean(self.mainFitTraj[i:i+k_test]) for i in range(len(self.mainFitTraj)-k_test+1)]); plt.savefig(outfolder+'/main'+str(gen)+'.png'); plt.clf()

all_target_purities=[x/100 for x in [98.70,98.82,98.57,99.03,96.73,93.55,92.49]]
all_target_acetate=[x for x in [0.006,0.006,0.006,0.006,0.006,0.006,None]]
models=['models/TBR1_UP_4h.pickle', 'models/TBR1_UP_3h.pickle', 'models/TBR1_UP_2h.pickle', 'models/TBR1_UP_1h.pickle', 'models/TBR1_UP_45min.pickle', 'models/TBR1_UP_30min.pickle', 'models/TBR1_UP_25min.pickle']
media=['media/medium_4h.tsv', 'media/medium_3h.tsv', 'media/medium_2h.tsv', 'media/medium_1h.tsv', 'media/medium_45min.tsv', 'media/medium_30min.tsv', 'media/medium_25min.tsv']
grts=['4h','3h','2h','1h','45min','30min','25min']

i=0 # which sample to optimize
alpha=0.8 if i!=0 else 0.6
beta=0.8
lrate=0.001
delta=0.1
startep=0

#target_purities=[all_target_purities[i]]+[all_target_acetate[i]] # if also optimization for acetate
target_purities=[all_target_purities[i]]
model=models[i]
medium={x.split('\t')[0]:float(x.split('\t')[1]) for x in open('data/starting_medium.tsv')}

fd=finite_difference(medium,target_purities,model,delta=delta,lrate=lrate,loss=weighted_MAE,alpha=alpha,beta=beta,l2_decay=0.01)
fd.train(500,test_every=(10000,1),checkpoint_every=(100,f'medium_opt_{grts[i]}'),starting_epoch=startep)





