# Content
This repository contains scripts and data associated to the work described in **title**, Sanguineti et al, **year**.  
The whole content of the repository is mentioned below, where each section describes a specific procedure applied in the present study, addressing the corresponding scripts and data. 
## Modified cooperative tradeoff strategy
In the original strategy (https://doi.org/10.1128/mSystems.00606-19), implemented in the Micom Python package (https://github.com/micom-dev/micom), the community growth rate CG is defined as a weighted sum of individual species’ growth rates. In Python code, this can be written as:
```
CG = sum( rel_abundances[i] * growth_rates[i] for i in range(num_species) )
# where rel_abundances and growth_rates are sequences listing all species relative abundances and growth rates, respectively
```
The first step of the strategy consists in finding the maximum community growth rate. A second optimization step is then performed, where community growth rate is allowed to take values down to a certain fraction alpha of its maximum and the squared sum of the growth rates is minimized:
```
sum( growth_rates[i]**2 for i in range(num_species) ) # find growth rates minimizing this function
```
The fraction alpha can be defined by the user to meet specific criteria.  
This minimization has the effect of distributing community growth among members. Moreover, as shown by the developers of Micom, this minimization ends up to result in growth rates that are linearly and positively dependent on the species’ relative abundances, if the problem's constraints allow it. This effect works fine in systems like the human gut, the main target of the cooperative tradeoff strategy, where actual growth rates tend to be positively related to relative abundances (taken as a proxy for absolute abundances). However, it may not work well in other systems (e.g. a trickle bed reactor) potentially displaying a different relative abundance - growth rates relationship. In order to be able to impose a different relationship, I introduced the relative abundance and a weighting parameter beta in the objective function to minimize:
```
sum( ( rel_abundances[i]**beta * growth_rates[i] )**2 for i in range(num_species) ) # find growth rates minimizing this function
```
The parameter beta can be defined by the user to meet specific criteria.  
When beta equals zero, the function to minimize corresponds to the one of the original cooperative tradeoff strategy, that is, the optimization strategy forces a linear positive relationship between growth rates and relative abundances. When beta equals one, inverse proportionality between growth rates and relative abundances is forced. Moreover, empirical analysis suggests that, when beta equals 0.5, equality of growth rates seems to be forced, although the actual effect is dependent on the value of α. These tendencies are met when the problem’s constraints allow them, and when the α parameter is sufficiently low.  

In this repository, the scripts implementing this modified version of the cooperative tradeoff strategy are scripts/community.py and scripts/problems.py. These scripts were copy-pasted from the Micom package, and the modification was implemented by modifying lines 753, 802 of scripts/community.py and lines 27, 63, 69, 88 of scripts/problems.py.  
Replacing the corresponding files of Micom with these two files allows to run the modified cooperative tradeoff strategy (as well as the original one if beta is set to zero) within the Micom framework.  

Given the problem of multiple solutions and the dependance of the obtained solution on the solver status, to exactly reproduce the results presented in the article using the models in models/ and the media in media/, the script scripts/solve_models.py should be used, as well as IBM ILOG CPLEX Optimization Studio 22.1.1 as solver.
## Pairwise modelling
In the study, pairwise metabolic modelling was used to infer the relative interaction strength of all pairs of species among {'Saccharicenans_sp_', 'Pseudothermotoga_B_sp_', 'Methanothermobacter_marburgensis_1', 'Methanothermobacter_thermautotrophicus_'}. Two-species community models were built with Micom, and are available in models/ .  The script implementing the approach to solve such pairwise models is implemented in scripts/pairwise.py .
## Simulating bioaugmentation
Bioaugmentation simulations were performed by introducing in the community models corresponding to each condition all the species whose genome was reconstructed, one (or several) at a time.  
The genome-scale models associated to each species are available in single_models/ . For each species (or set of species) and for each condition, the script scripts/bioaugmentation.py was used to generate Micom community models that include the additional species, as well as to solve them as described in the manuscript. This script needs data that are stored in the files data/relative_abundances.csv and data/gasses.csv (H2 and CO2 fluxes in millimoles/h).  
## Medium optimization
Since we were interested in testing whether addition of certain species could result in higher conversion rate of CO2 to methane, we needed the models to mimic the experimental methane purities without constraining CO2, H2 and CH4 exports to do so. Thus, we wanted to find condition-specific media that, given the applied optimization strategy, generated CO2, H2 and CH4 exports fluxes such that predicted and observed methane purities were approximately equal. 
The script implementing such approach is can be found in scripts/medium_optimization.py, and it is an in-house implementation of the evolution strategy developed by Salismans et al (https://doi.org/10.48550/arXiv.1703.03864).
The idea is to start from a given medium (exchange reactions' lower bound), that constitutes the parameter vector. Given a population size of N members, at each iteration the parameter vector is perturbed N times with noise sampled from a normal distribution with zero mean and a certain variance, and, for each perturbation, the FBA solution is obtained. Methane purity is calculated for all solutions using the resulting fluxes, it is compared with the experimental methane purity and a loss function is calculated. By combining the values of the loss function for all parameter perturbations, the natural gradient of the exchange reaction lower bound on the loss function can be obtained. To minimize the loss function, such lower bound is updated by taking a step in the opposite direction of the gradient (actually, Adam optimization (https://doi.org/10.48550/arXiv.1412.6980) was used). 
Here, the population size was 100, the variance of the perturbations was 0.01, the learning rate 0.001, Adam parameters β1 and β2 were 0.9 and 0.999, respectively. Gradients were clipped within -1 and 1.
The following table lists the compounds present in the media, the corresponding exchange reaction and the absolute value of the corresponding lower bound; exchange reactions that are not listed have lower bound of 0.0.
  
