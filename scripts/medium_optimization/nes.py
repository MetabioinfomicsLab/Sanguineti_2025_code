import ray
from copy import deepcopy
from torch import normal,manual_seed,tensor,cat,matmul,tensordot,unsqueeze,set_num_threads,cdist,zeros_like,sqrt,reshape,randperm,clamp,min,exp,multinomial,ones,diag
from torch.nn import MSELoss
from torch.nn.functional import sigmoid,relu,pad
from torch.optim import SGD,Adam
from torch.distributions import Normal,MultivariateNormal
from secrets import randbits
import time
import os
from pickle import dump,load
import matplotlib.pyplot as plt
from statistics import mean

def chunks(l,nproc):
    right_div=len(l)//nproc
    nmore=len(l)%nproc
    return [l[i*(right_div+1):i*(right_div+1)+right_div+1] for i in range(nmore)]+[l[nmore*(right_div+1)+i*right_div:nmore*(right_div+1)+i*right_div+right_div] for i in range(nproc-nmore) if nmore<len(l)]
def numChunks(l,nproc):
    right_div=len(l)//nproc
    nmore=len(l)%nproc
    return [len(l[i*(right_div+1):i*(right_div+1)+right_div+1]) for i in range(nmore)]+[len(l[nmore*(right_div+1)+i*right_div:nmore*(right_div+1)+i*right_div+right_div]) for i in range(nproc-nmore) if nmore<len(l)]


class base_container_multiagent:
    def __init__(self,env,indexes,totChildren,lrate,sigma,batchlen):
        self.envs=[deepcopy(env) for _ in indexes] # create N environments as much is the number of children evaluated here
        self.base_policies=[deepcopy(agent.policy) for agent in self.envs[0].agents] # these will be the network that will be updated with the real gradients and then re-assigned as attribute to the agents
        self.indexes=indexes # which indexes in the perturbations have to be evaluated by this container
        self.children=totChildren # total number of perturbations that are to be generated
        self.lrate=lrate # learning rate of the evolutionary strategy
        self.sigma=sigma # sigma of the evolutionary strategy
        self.W_shapes=[self.envs[0].agents[i].policy.get_shape()[0] for i in range(len(self.envs[0].agents))]
        self.b_shapes=[self.envs[0].agents[i].policy.get_shape()[1] for i in range(len(self.envs[0].agents))]
        self.batchlen=batchlen
        set_num_threads(1)
    def _make_perturbations(self,seed): 
        """Seed is decided by the wrapper. The policy of the agents is assumed to have the same architecture.
        Save perturbations as attribute."""
        manual_seed(seed)
        self.weight_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.W_shapes[i]] for i in range(len(self.envs[0].agents))] # a list of perturbations for each agent for each layer of agent's policy
        self.bias_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.b_shapes[i]] for i in range(len(self.envs[0].agents))]
    def _perturbate_policies(self):
        for index,env in zip(self.indexes,self.envs):
            for ith_ag in range(len(env.agents)):
                agent=env.agents[ith_ag]
                for ith_layer,layer in enumerate(agent.policy.layers):
                    layer.weight+=self.sigma*self.weight_perturbations[ith_ag][ith_layer][index]
                    layer.bias+=self.sigma*self.bias_perturbations[ith_ag][ith_layer][index]
    def _get_envs_to_step(self,steps): # get the env to be at a certain num step
        for step,env in zip(steps,self.envs):
            env.run_batched_episode(step,None)
        return True
    def _get_envstepcounter(self): # only for debugging wrapper.batched_train()
        return [env.step_counter for env in self.envs]
    def _run_episode(self,reset_seed):
        #return cat([unsqueeze(env.run_episode(reset_seed),0) for env in self.envs],dim=0) # return matrix where each row contains rewards of each agent for each simulated environment
        return cat([unsqueeze(env.run_batched_episode(self.batchlen,reset_seed),0) for env in self.envs],dim=0)
    def _update_policies(self,all_rewards): # update according to gradients the base policies
        for i in range(len((self.base_policies))):
            policy,rewards,wpert,bpert=self.base_policies[i],all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                layer.weight+=(self.lrate*(tensordot(wpert_layer,rewards,dims=([0],[0]))))/(self.children*self.sigma)
                layer.bias+=(self.lrate*matmul(bpert_layer.T,rewards))/(self.children*self.sigma)
    def get_rewards(self,perturbation_seed,reset_seed):
        self._make_perturbations(perturbation_seed)
        self._perturbate_policies()
        rews=self._run_episode(reset_seed)
        return rews
    def step(self,rewards):
        self._update_policies(rewards) # update base policies
        for env in self.envs: # deepcopy base policies to each agent of each child environment
            for agent,policy in zip(env.agents,self.base_policies):
                agent.policy=deepcopy(policy)
    def update_params(self,lr,sigma):
        self.lrate=lr
        self.sigma=sigma

@ray.remote
class ray_container_multiagent:
    def __init__(self,env,indexes,totChildren,lrate,sigma):
        self.envs=[deepcopy(env) for _ in indexes] # create N environments as much is the number of children evaluated here
        self.base_policies=[deepcopy(agent.policy) for agent in self.envs[0].agents] # these will be the network that will be updated with the real gradients and then re-assigned as attribute to the agents
        self.indexes=indexes # which indexes in the perturbations have to be evaluated by this container
        self.children=totChildren # total number of perturbations that are to be generated
        self.lrate=lrate # learning rate of the evolutionary strategy
        self.sigma=sigma # sigma of the evolutionary strategy
        self.W_shapes=[self.envs[0].agents[i].policy.get_shape()[0] for i in range(len(self.envs[0].agents))]
        self.b_shapes=[self.envs[0].agents[i].policy.get_shape()[1] for i in range(len(self.envs[0].agents))]
        set_num_threads(1)
    def _make_perturbations(self,seed): 
        """Seed is decided by the wrapper. The policy of the agents is assumed to have the same architecture.
        Save perturbations as attribute."""
        manual_seed(seed)
        self.weight_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.W_shapes[i]] for i in range(len(self.envs[0].agents))] # a list of perturbations for each agent for each layer of agent's policy
        self.bias_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.b_shapes[i]] for i in range(len(self.envs[0].agents))]
    def _perturbate_policies(self):
        for index,env in zip(self.indexes,self.envs):
            for ith_ag in range(len(env.agents)):
                agent=env.agents[ith_ag]
                for ith_layer,layer in enumerate(agent.policy.layers):
                    layer.weight+=self.sigma*self.weight_perturbations[ith_ag][ith_layer][index]
                    layer.bias+=self.sigma*self.bias_perturbations[ith_ag][ith_layer][index]
    def _run_episode(self,reset_seed):
        #return cat([unsqueeze(env.run_episode(reset_seed),0) for env in self.envs],dim=0) # return matrix where each row contains rewards of each agent for each simulated environment
        return cat([unsqueeze(env.run_batched_episode(self.batchlen,reset_seed),0) for env in self.envs],dim=0) # return matrix where each row contains rewards of each agent for each simulated environment
    def _update_policies(self,all_rewards): # update according to gradients the base policies
        for i in range(len((self.base_policies))):
            policy,rewards,wpert,bpert=self.base_policies[i],all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                layer.weight+=(self.lrate*(tensordot(wpert_layer,rewards,dims=([0],[0]))))/(self.children*self.sigma)
                layer.bias+=(self.lrate*matmul(bpert_layer.T,rewards))/(self.children*self.sigma)
    def get_rewards(self,perturbation_seed,reset_seed):
        self._make_perturbations(perturbation_seed)
        self._perturbate_policies()
        rews=self._run_episode(reset_seed)
        return rews
    def step(self,rewards):
        self._update_policies(rewards) # update base policies
        for env in self.envs: # deepcopy base policies to each agent of each child environment
            for agent,policy in zip(env.agents,self.base_policies):
                agent.policy=deepcopy(policy)
    def update_lr(self,lr):
        self.lrate=lr

class wrapper_multiagent:
    def __init__(self,env,nproc,totChildren,lrate=0.01,sigma=0.1,make_containers=True):
        batched_indexes=chunks(list(range(totChildren)),nproc)
        self.children_per_proc=numChunks(list(range(totChildren)),nproc)
        if make_containers: self.containers=[ray_container_multiagent.remote(env,ind,totChildren,lrate,sigma) for ind in batched_indexes]
        self.children=totChildren # total number of perturbations that are to be generated
        self.lrate=lrate # learning rate of the evolutionary strategy
        self.sigma=sigma # sigma of the evolutionary strategy
        self.env=env
        self.W_shapes=[self.env.agents[i].policy.get_shape()[0] for i in range(len(self.env.agents))]
        self.b_shapes=[self.env.agents[i].policy.get_shape()[1] for i in range(len(self.env.agents))]
        self.childRews,self.testRews=[],[]
        set_num_threads(1)
    def _make_perturbations(self,seed):
        manual_seed(seed)
        self.weight_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.W_shapes[i]] for i in range(len(self.env.agents))] # a list of perturbations for each agent for each layer of agent's policy
        self.bias_perturbations=[[normal(0,1,(self.children,)+layer_shape) for layer_shape in self.b_shapes[i]] for i in range(len(self.env.agents))]
    def _update_policies(self,all_rewards): # update according to gradients the base policies
        for i in range(len((self.env.agents))):
            policy,rewards,wpert,bpert=self.env.agents[i].policy,all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                layer.weight+=(self.lrate*(tensordot(wpert_layer,rewards,dims=([0],[0]))))/(self.children*self.sigma)
                layer.bias+=(self.lrate*matmul(bpert_layer.T,rewards))/(self.children*self.sigma)
    def get_rewards(self,reset_seed):
        #return self.env.run_episode(reset_seed,printstats=False) # QUI METTI PRINTSTATS
        return self.env.run_batched_episode(self.batchlen,reset_seed,printstats=False,printtest=True) # QUI METTI PRINTSTATS
        #return self.env.timed_run_episode(reset_seed,printstats=False) # QUI METTI PRINTSTATS
    def step(self):
        reset_seed,pert_seed=randbits(64),randbits(64)
        rewards=[cont.get_rewards.remote(pert_seed,reset_seed) for cont in self.containers] # send command to containers
        self._make_perturbations(pert_seed) # do other things meanwhile
        test_reward=self.get_rewards(reset_seed) # do other things meanwhile
        rewards=cat([ray.get(x) for x in rewards],dim=0).T # Wait for containers to finish and concatenate-transpose reward matrix such that all rewards of first agent in first raw, and so on
        norm_rewards=(rewards-rewards.mean(axis=1,keepdims=True))/(rewards.std(axis=1,keepdims=True)+1e-5) # standardize rewards
        l=[cont.step.remote(norm_rewards) for cont in self.containers] # send command to containers to uodate policies
        self._update_policies(norm_rewards) # update policies
        [ray.get(x) for x in l] # only for waiting
        return rewards.mean(axis=1),test_reward # return child rewards and test reward of previous policies
    def checkpoint(self,epoch,outfolder):
        if not os.path.isdir(outfolder): os.makedirs(outfolder)
        for n,agent in enumerate(self.env.agents):
            with open(f"{outfolder}/{agent.name}_{n}_policy_{epoch}.pickle",'wb') as new:
                dump(agent.policy,new)
        k=50 # average rewards over sliding window equal to k
        for i in range(len(self.env.agents)): 
            tomean=[float(episode[i]) for episode in self.childRews]
            plt.plot([mean(tomean[j:j+k]) for j in range(len(tomean)-k+1)],label=self.env.agents[i].name+'_'+str(i))
        plt.legend(loc='upper left'); plt.savefig(f'{outfolder}/rewardCurve_children_{epoch}.png'); plt.clf()
        for i in range(len(self.env.agents)): 
            tomean=[float(episode[i]) for episode in self.testRews]
            plt.plot([mean(tomean[j:j+k]) for j in range(len(tomean)-k+1)],label=self.env.agents[i].name+'_'+str(i))
        plt.legend(loc='upper left'); plt.savefig(f'{outfolder}/rewardCurve_test_{epoch}.png'); plt.clf()
    def train(self,epochs,checkpoint_every=tuple(),lrate_decay=tuple(),starting_epoch=0):
        """checkpoint_every: (every,outfolder)
           lrate_decay: (every,decay)"""
        for ep in range(starting_epoch+1,starting_epoch+epochs+1):
            now=time.time()
            meanchildrew,testrew=self.step()
            self.childRews.append(meanchildrew)
            self.testRews.append(testrew)
            if checkpoint_every:
                every,outfold=checkpoint_every
                if not ep%every: self.checkpoint(ep,outfold)
            if lrate_decay:
                every,decay=lrate_decay
                if not ep%every: 
                    self.lrate*=decay
                    res=[cont.update_lr.remote(self.lrate) for cont in self.containers]
                    res=[ray.get(x) for x in res] # only for waiting
            print(f"Epoch: {ep}, Meanchildrew: {[round(float(x),4) for x in meanchildrew]}, Testrew: {[round(float(x),4) for x in testrew]}, Time: {round(time.time()-now,2)}")#,'ChosenBounds',[ag.last_chosen_bounds for ag in self.env.agents])
            ########### ROBE MANUALI ADDIZIONALI
            if ep==100: # RIAZZERA PER EVITARE GRANDI NEGATIVI
                self.childRews=[]
                self.testRews=[]

class base_container_nsr(base_container_multiagent):
    def __init__(self,env,indexes,totChildren,lrate,sigma,max_archive_size,k,adam,batchlen,clip):
        super().__init__(env,indexes,totChildren,lrate,sigma,batchlen)
        self.max_archive_size=max_archive_size
        self.archive=[[] for _ in self.envs[0].agents]
        self.adam=adam
        self.k=k
        self.clip=clip
        if adam:
            self.adam=True; self.m1=adam[0]; self.m2=adam[1]; self.velocities=[]; self.rmsp=[]
            for i in range(len((self.base_policies))):
                policy=self.base_policies[i]
                self.velocities.append([[zeros_like(policy.layers[j].weight),zeros_like(policy.layers[j].bias)] for j in range(len(policy.layers))])
                self.rmsp.append([[zeros_like(policy.layers[j].weight),zeros_like(policy.layers[j].bias)] for j in range(len(policy.layers))])
    def _adam_update_policies(self,all_rewards): 
        for i in range(len((self.base_policies))):
            policy,rewards,wpert,bpert=self.base_policies[i],all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                gradientsW=tensordot(wpert_layer,rewards,dims=([0],[0]))/(self.children*self.sigma)
                gradientsB=matmul(bpert_layer.T,rewards)/(self.children*self.sigma)
                self.velocities[i][j][0]=self.m1*self.velocities[i][j][0]+(1-self.m1)*gradientsW
                self.velocities[i][j][1]=self.m1*self.velocities[i][j][1]+(1-self.m1)*gradientsB
                self.rmsp[i][j][0]=self.m2*self.rmsp[i][j][0]+(1-self.m2)*gradientsW**2
                self.rmsp[i][j][1]=self.m2*self.rmsp[i][j][1]+(1-self.m2)*gradientsB**2
                layer.weight+=self.lrate*((self.velocities[i][j][0]/(1-self.m1))/(sqrt(self.rmsp[i][j][0]/(1-self.m2))+1e-7))
                layer.bias+=self.lrate*((self.velocities[i][j][1]/(1-self.m1))/(sqrt(self.rmsp[i][j][1]/(1-self.m2))+1e-7))
    def _update_policies(self,all_rewards): # update according to gradients the base policies
        for i in range(len((self.base_policies))):
            policy,rewards,wpert,bpert=self.base_policies[i],all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                layer.weight+=self.lrate*clamp((tensordot(wpert_layer,rewards,dims=([0],[0])))/(self.children*self.sigma),-self.clip,self.clip)
                layer.bias+=self.lrate*clamp(matmul(bpert_layer.T,rewards)/(self.children*self.sigma),-self.clip,self.clip)
    def get_rewards(self,perturbation_seed,reset_seed):
        self._make_perturbations(perturbation_seed)
        self._perturbate_policies()
        rews=self._run_episode(reset_seed)
        return rews 
    def step(self,rewards): # update archive too
        if self.adam: self._adam_update_policies(rewards) # update base policies
        else: self._update_policies(rewards)
        for env in self.envs: # deepcopy base policies to each agent of each child environment
            for agent,policy in zip(env.agents,self.base_policies):
                agent.policy=deepcopy(policy)
    def load_archive(self,archive):
        self.archive=deepcopy(archive)

@ray.remote
class ray_container_nsr(base_container_nsr):
    def __init__(self, env, indexes, totChildren, lrate, sigma, max_archive_size, k, adam, batchlen,clip):
        super().__init__(env, indexes, totChildren, lrate, sigma, max_archive_size, k, adam, batchlen,clip)
        for env in self.envs:
            for ag in env.agents:
                ag.initialize()

    

class nsr_wrapper(wrapper_multiagent):
    def __init__(self,env,nproc,totChildren,batchlen,lrate=0.01,sigma=0.1,novelty_weight=0.5,max_archive_size=250,k=10,adam=(0.9,0.999),makecontainer=True,clip=1.0):
        """npvelty_weight: weight of novelty rewards in overall rewards
           max_archive_set: max number of previous episodes to store for each agent
           k: k-nearest neighbours in novelty computation
           adam: tuple m1,m2. Set empty if want to use standard gradient descent """
        super().__init__(env,nproc,totChildren,lrate,sigma,make_containers=False)
        batched_indexes=chunks(list(range(totChildren)),nproc)
        if makecontainer: self.containers=[ray_container_nsr.remote(env,ind,totChildren,lrate,sigma,max_archive_size,k,adam,batchlen,clip) for ind in batched_indexes]
        self.novelty_weight=novelty_weight
        self.max_archive_size=max_archive_size
        self.archive=[[] for _ in self.env.agents]
        self.adam=adam
        self.k=k
        self.batchlen=batchlen
        self.clip=clip
        self.childmeans=[[] for _ in self.env.agents]
        self.testmeans=[[] for _ in self.env.agents]
        self.best_loss=-1000.0
        for ag in self.env.agents: ag.initialize()
        if adam:
            self.adam=True; self.m1=adam[0]; self.m2=adam[1]; self.velocities=[]; self.rmsp=[]
            for i in range(len((self.env.agents))):
                policy=self.env.agents[i].policy
                self.velocities.append([[zeros_like(policy.layers[j].weight),zeros_like(policy.layers[j].bias)] for j in range(len(policy.layers))])
                self.rmsp.append([[zeros_like(policy.layers[j].weight),zeros_like(policy.layers[j].bias)] for j in range(len(policy.layers))])
    def _adam_update_policies(self,all_rewards): 
        for i in range(len((self.env.agents))):
            policy,rewards,wpert,bpert=self.env.agents[i].policy,all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                gradientsW=tensordot(wpert_layer,rewards,dims=([0],[0]))/(self.children*self.sigma)
                gradientsB=matmul(bpert_layer.T,rewards)/(self.children*self.sigma)
                self.velocities[i][j][0]=self.m1*self.velocities[i][j][0]+(1-self.m1)*gradientsW
                self.velocities[i][j][1]=self.m1*self.velocities[i][j][1]+(1-self.m1)*gradientsB
                self.rmsp[i][j][0]=self.m2*self.rmsp[i][j][0]+(1-self.m2)*gradientsW**2
                self.rmsp[i][j][1]=self.m2*self.rmsp[i][j][1]+(1-self.m2)*gradientsB**2
                layer.weight+=self.lrate*((self.velocities[i][j][0]/(1-self.m1))/(sqrt(self.rmsp[i][j][0]/(1-self.m2))+1e-7))
                layer.bias+=self.lrate*((self.velocities[i][j][1]/(1-self.m1))/(sqrt(self.rmsp[i][j][1]/(1-self.m2))+1e-7))
    def _update_policies(self,all_rewards): # update according to gradients the base policies
        for i in range(len((self.env.agents))):
            policy,rewards,wpert,bpert=self.env.agents[i].policy,all_rewards[i],self.weight_perturbations[i],self.bias_perturbations[i]
            for j in range(len(policy.layers)):
                layer,wpert_layer,bpert_layer=policy.layers[j],wpert[j],bpert[j]
                layer.weight+=self.lrate*clamp((tensordot(wpert_layer,rewards,dims=([0],[0])))/(self.children*self.sigma),-self.clip,self.clip)
                layer.bias+=self.lrate*clamp(matmul(bpert_layer.T,rewards)/(self.children*self.sigma),-self.clip,self.clip)
    def step(self):
        reset_seed,pert_seed=randbits(64),randbits(64)
        both_rewards=[cont.get_rewards.remote(pert_seed,reset_seed) for cont in self.containers] # send command to containers
        self._make_perturbations(pert_seed) # do other things meanwhile
        test_reward=self.get_rewards(reset_seed) # do other things meanwhile
        ## Here put END CONDITIONS
        purcondition=abs(self.env.agents[0].vals[0]-self.env.agents[0].target[0])<0.005
        accondition=self.env.agents[0].vals[1]<0.002 if self.env.agents[0].target[1]==0.0001 else 0.0095<self.env.agents[0].vals[1]<0.0105
        growthcondition=not self.env.agents[0].vals[2]
        if purcondition and accondition and growthcondition: 
            self.policy_checkpoint(self.outfolder) 
            print('END CRITERION MET !'); return 0.0,0.0 
        ## Above for END CONDITIONS
        both_rewards=[ray.get(x) for x in both_rewards] # get both rewards and novelties (list of tuples)
        rewards=cat([both for both in both_rewards],dim=0).T
        norm_rewards=(rewards-rewards.mean(axis=1,keepdims=True))/(rewards.std(axis=1,keepdims=True)+1e-5)
        l=[cont.step.remote(norm_rewards) for cont in self.containers] # send command to containers to uodate policies and update archive
        if self.adam: self._adam_update_policies(norm_rewards) # update policies
        else: self._update_policies(norm_rewards)
        [ray.get(x) for x in l] # only for waiting
        return rewards.mean(axis=1),test_reward # return child rewards and test reward of previous policies
    def batch_train(self,epochs,checkpoint_every=tuple(),lrate_decay=tuple(),starting_epoch=0,start_novelty=1,target_fitness=-0.01):
        self.outfolder=checkpoint_every[1]
        for ep in range(starting_epoch+1,starting_epoch+epochs+1):
            now=time.time()
            meanchildrew,testrew=self.step()
            if not meanchildrew and not testrew: break
            #if self.best_loss>target_fitness: print('loss is smaller than 0.01 !!!'); break # general end condition
            self.childRews.append(meanchildrew); self.testRews.append(testrew)
            if checkpoint_every:
                every,outfold=checkpoint_every
                if not ep%every: self.checkpoint(ep,outfold)
            if lrate_decay:
                every,decay=lrate_decay
                if not ep%every: 
                    self.lrate*=decay
                    res=[cont.update_lr.remote(self.lrate) for cont in self.containers]
                    res=[ray.get(x) for x in res] # only for waiting
            #print(f"Epoch: {ep}, Meanchildrew: {[round(float(x),4) for x in meanchildrew]}, Testrew: {[round(float(x),7) for x in testrew]}, Novelty_weight: {[round(float(x[0]),3) for x in self.novelty_weight] if not isinstance(self.novelty_weight,float) else self.novelty_weight},Time: {round(time.time()-now,2)}")
            print(f"Epoch: {ep}, Meanchildrew: {[round(float(x),4) for x in meanchildrew]}, Testrew: {[round(float(x),7) for x in testrew]}, Values: {[round(x,5) for x in self.env.agents[0].vals]}, Target:  {[round(x,5) for x in self.env.agents[0].target]}, Best: {self.best_loss}, Time: {round(time.time()-now,2)}")
    def policy_checkpoint(self,outfolder):
        if not os.path.isdir(outfolder): os.makedirs(outfolder)
        for n,agent in enumerate(self.env.agents):
            with open(f"{outfolder}/{agent.name}_{n}_policy_best.pickle",'wb') as new:
                dump(agent.policy,new)
    def checkpoint(self,epoch,outfolder):
        if not os.path.isdir(outfolder): os.makedirs(outfolder)
        for n,agent in enumerate(self.env.agents):
            with open(f"{outfolder}/{agent.name}_{n}_policy_{epoch}.pickle",'wb') as new:
                dump(agent.policy,new)
        plt.clf(); plt.close()
        k=20
        for i in range(len(self.env.agents)): 
            tomean=[float(episode[i]) for episode in self.childRews]
            for j in range(len(tomean)-k+1): self.childmeans[i].append(mean(tomean[j:j+k]))
            plt.plot(self.childmeans[i],label=self.env.agents[i].name+'_'+str(i))
        plt.legend(loc='upper left'); plt.savefig(f'{outfolder}/rewardCurve_children_{epoch}.png'); plt.clf()
        self.childRews=[]
        for i in range(len(self.env.agents)): 
            tomean=[float(episode[i]) for episode in self.testRews]
            for j in range(len(tomean)-k+1): self.testmeans[i].append(mean(tomean[j:j+k]))
            plt.plot(self.testmeans[i],label=self.env.agents[i].name+'_'+str(i))
        plt.legend(loc='upper left'); plt.savefig(f'{outfolder}/rewardCurve_test_{epoch}.png'); plt.clf()
        self.testRews=[]

import numpy as np
from torch import inf

@ray.remote
class multiobjective_container(base_container_nsr):
    def __init__(self, env, indexes, totChildren, lrate, sigma, max_archive_size, k, adam, batchlen, clip):
        super().__init__(env, indexes, totChildren, lrate, sigma, max_archive_size, k, adam, batchlen, clip)
        for env in self.envs:
            for ag in env.agents:
                ag.initialize()
    def _run_episode(self,reset_seed):
        return  self.envs[0].run_batched_episode(self.batchlen,reset_seed)
    def step(self,rewards): # update archive too
        if self.adam: self._adam_update_policies(rewards) # update base policies
        else: self._update_policies(rewards)
        for env in self.envs: # deepcopy base policies to each agent of each child environment
            for agent,policy in zip(env.agents,self.base_policies):
                policy.layers[0].weight=policy.layers[0].weight.clamp(-self.sigma,inf) ### avoid too negative
                agent.policy=deepcopy(policy)
class multiobjective_wrapper(nsr_wrapper):
    def __init__(self, env, nproc, totChildren, batchlen, target, loss, weights, lrate=0.01, sigma=0.1, novelty_weight=0.5, max_archive_size=250, k=10, adam=(0.9, 0.999), makecontainer=True, clip=1):
        super().__init__(env, nproc, totChildren, batchlen, lrate, sigma, novelty_weight, max_archive_size, k, adam, False, clip)
        batched_indexes=chunks(list(range(totChildren)),nproc)
        if makecontainer: self.containers=[multiobjective_container.remote(env,ind,totChildren,lrate,sigma,max_archive_size,k,adam,batchlen,clip) for ind in batched_indexes] 
        self.target=target
        self.loss=loss
        self.weights=weights
    def _get_loss_tensor(self,value,target):
        return -(value-target).abs()
    def step(self):
        reset_seed,pert_seed=randbits(64),randbits(64)
        both_rewards=[cont.get_rewards.remote(pert_seed,reset_seed) for cont in self.containers] # send command to containers
        self._make_perturbations(pert_seed) # do other things meanwhile
        test_reward=self.get_rewards(reset_seed) # do other things meanwhile
        test_reward=self.loss(test_reward,self.target)
        ## Here put END CONDITIONS
        purcondition=abs(self.env.agents[0].vals[0]-self.env.agents[0].target[0])<0.01
        accondition=self.env.agents[0].vals[1]<0.002
        metcondition=True#abs(self.env.agents[0].vals[2]-self.env.agents[0].target[2])<0.5
        growthcondition=not self.env.agents[0].vals[3]
        if purcondition and accondition and metcondition and growthcondition: 
            self.policy_checkpoint(self.outfolder) 
            print('END CRITERION MET !'); return 0.0,0.0 
        ## Above for END CONDITIONS
        both_rewards=[ray.get(x) for x in both_rewards] # get both rewards and novelties (list of tuples)
        all_losses=[self._get_loss_tensor(tensor([[child[i] for child in both_rewards]]),self.target[i]) for i in range(len(self.target))]
        norm_losses=[(rewards-rewards.mean(axis=1,keepdims=True))/(rewards.std(axis=1,keepdims=True)+1e-5) for rewards in all_losses]
        norm_rewards=zeros_like(norm_losses[0])
        for l,w in zip(norm_losses,self.weights): norm_rewards+=(l*w)
        l=[cont.step.remote(norm_rewards) for cont in self.containers] # send command to containers to uodate policies and update archive
        if self.adam: self._adam_update_policies(norm_rewards) # update policies
        else: self._update_policies(norm_rewards)
        self.env.agents[0].policy.layers[0].weight=self.env.agents[0].policy.layers[0].weight.clamp(-self.sigma,inf)
        [ray.get(x) for x in l] # only for waiting
        return [np.mean([self.loss(this,self.target) for this in both_rewards])],[test_reward] # return child rewards and test reward of previous policies
    def batch_train(self,epochs,checkpoint_every=tuple(),lrate_decay=tuple(),starting_epoch=0,start_novelty=1,target_fitness=-0.01):
        self.outfolder=checkpoint_every[1]
        for ep in range(starting_epoch+1,starting_epoch+epochs+1):
            now=time.time()
            meanchildrew,testrew=self.step()
            if not meanchildrew and not testrew: break
            #if self.best_loss>target_fitness: print('loss is smaller than 0.01 !!!'); break # general end condition
            self.childRews.append(meanchildrew); self.testRews.append(testrew)
            if checkpoint_every:
                every,outfold=checkpoint_every
                if not ep%every: self.checkpoint(ep,outfold)
            if lrate_decay:
                every,decay=lrate_decay
                if not ep%every: 
                    self.lrate*=decay
                    res=[cont.update_lr.remote(self.lrate) for cont in self.containers]
                    res=[ray.get(x) for x in res] # only for waiting
            #print(f"Epoch: {ep}, Meanchildrew: {[round(float(x),4) for x in meanchildrew]}, Testrew: {[round(float(x),7) for x in testrew]}, Novelty_weight: {[round(float(x[0]),3) for x in self.novelty_weight] if not isinstance(self.novelty_weight,float) else self.novelty_weight},Time: {round(time.time()-now,2)}")
            print(f"Epoch: {ep}, Meanchildrew: {[round(float(x),4) for x in meanchildrew]}, Testrew: {[round(float(x),7) for x in testrew]}, Values: {[round(x,5) for x in self.env.agents[0].vals]}, Target:  {[round(x,5) for x in self.env.agents[0].target]}, Best: {self.best_loss}, Time: {round(time.time()-now,2)}")
    