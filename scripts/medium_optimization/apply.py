import agent
import nes
from pickle import load
import numpy as np
import torch
import sys


### LOSS FUNCTIONS ###
def MSE(pred,real):
    return np.mean([(x-y)**2 for x,y in zip(pred,real)])
def MAE(pred,real):
    return np.mean([abs(x-y) for x,y in zip(pred,real)])
def weighted_MAE(pred,real):
    ws=[0.2,0.8] 
    return np.sum([w*abs(x-y) for x,y,w in zip(pred,real,ws)])


alpha=0.5
beta=0.6
grts=['4h','3h','2h','1h','45min','30min','25min']

iteration=int(sys.argv[1]) # set the i-th iteration to perform
i=int(sys.argv[2]) # set the i-th GRT-sample to optimize
if iteration==0 and i<=2: exit()
print(iteration,grts[i])

#all_target_purities=[x/100 for x in [98.70,98.82,98.57,99.03,96.73,93.55,92.49]] # if target is to reproduce experimental data
#all_target_acetate=[x for x in [0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.01]] # if target is to reproduce experimental data
all_target_purities=[x/100 for x in [100]*7] # if target is to get optimal performance
all_target_acetate=[x for x in [0.0]*7] # if target is to get optimal performance
all_target_methane=[1.76, 2.33, 3.55, 7.11, 9.33, 13.21, 15.47] # if target is to get optimal performance
targets=[[x,y,z,0.0] for x,y,z in zip(all_target_purities,all_target_acetate,all_target_methane)] # the zero is for the number of not growing species
models=['models/TBR1_UP_4h.pickle', 'models/TBR1_UP_3h.pickle', 'models/TBR1_UP_2h.pickle', 'models/TBR1_UP_1h.pickle', 'models/TBR1_UP_45min.pickle', 'models/TBR1_UP_30min.pickle', 'models/TBR1_UP_25min.pickle']
# extracellular compounds in the medium
rids=['EX_cpd00013_m', 'EX_cpd00099_m', 'EX_cpd00971_m', 'EX_cpd00254_m', 'EX_cpd00063_m', 'EX_cpd00205_m', 'EX_cpd00009_m', 'EX_cpd10515_m', 'EX_cpd00034_m', 'EX_cpd00058_m', 'EX_cpd00030_m', 'EX_cpd11574_m', 'EX_cpd00149_m', 'EX_cpd00244_m', 'EX_cpd00048_m', 'EX_cpd00104_m', 'EX_cpd00393_m', 'EX_cpd00220_m', 'EX_cpd00305_m', 'EX_cpd00644_m', 'EX_cpd00067_m', 'EX_cpd10516_m', 'EX_cpd00528_m', 'EX_cpd00027_m', 'EX_cpd00023_m', 'EX_cpd00036_m', 'EX_cpd00073_m', 'EX_cpd00159_m', 'EX_cpd00047_m', 'EX_cpd00489_m', 'EX_cpd00041_m', 'EX_cpd03662_m', 'EX_cpd00098_m', 'EX_cpd00130_m', 'EX_cpd00064_m', 'EX_cpd00020_m', 'EX_cpd00264_m', 'EX_cpd00249_m', 'EX_cpd15432_m', 'EX_cpd00060_m', 'EX_cpd00017_m', 'EX_cpd00010_m', 'EX_cpd00054_m', 'EX_cpd00053_m', 'EX_cpd00033_m', 'EX_cpd00035_m', 'EX_cpd00039_m', 'EX_cpd00051_m', 'EX_cpd00065_m', 'EX_cpd00066_m', 'EX_cpd00069_m', 'EX_cpd00084_m', 'EX_cpd00161_m', 'EX_cpd00107_m', 'EX_cpd00118_m', 'EX_cpd00119_m', 'EX_cpd00156_m', 'EX_cpd00322_m', 'EX_cpd00028_m', 'EX_cpd00129_m', 'EX_cpd00132_m']

target=targets[i]
model=models[i]
startingsigma=0.05
medinit=torch.rand(len(rids))*startingsigma

ags=[agent.multiobj_medopt_agent(rids,target,model,MAE,alpha,beta,medinit)] 

outfold='output_folder' # store media, loss curve
nproc=100
numchild=nproc
eplen=1
env=agent.fake_env(ags)
trainer=nes.multiobjective_wrapper(env, # "environment" that generate reward (loss) (needed for compatibility)
                                   nproc, # number of threads for parallelization
                                   numchild, # number of perturbated parameters for each iteration
                                   eplen, # length of the episode, here clearly only 1 (needed for compatibility)  
                                   target, # target values, i.e. methane purity, acetate export, methane export, number of not growing species
                                   MSE, # loss function, here mean squared error
                                   [0.33,0.33,0.01,0.33], # weights for each component in the loss function
                                   lrate=0.001, # learning rate for adam optimizer
                                   sigma=0.01, # standard deviation of sampled perturbations
                                   novelty_weight=0.0, # (needed for compatibility)
                                   adam=(0.9,0.999) # adam optimizer parameters
                                   )
trainer.batch_train(5000, # number of iteration
                    (50,outfold) # checkpoint params (every,output_folder)
                    )

