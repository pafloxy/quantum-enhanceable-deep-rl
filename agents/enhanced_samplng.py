###################################################################################
## IMPORTS ##
###################################################################################

import numpy as np
from typing import Optional
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt

from qumcmc.basic_utils import qsm, states, MCMCChain
# from .prob_dist import *
from qumcmc.energy_models import IsingEnergyFunction, Exact_Sampling
from qumcmc.classical_mcmc_routines import test_accept, get_random_state
from qumcmc.quantum_mcmc_routines_qulacs import *
from typing import List

from .debn_ps import DEBN
###################################################################################
## REDEFINE DATACLASSES ##
###################################################################################
@dataclass
class MCMCState:
    
    hidden: str
    action: str
    state: str
    
    accepted: bool = False
    
    def __post_init__(self):
        self.var = self.hidden + self.action; self.fixed = self.state
        
        self.bitstring = self.var + self.fixed
        self.len_var = len(self.var)
        self.len_fixed = len(self.fixed)
        self.len = len(self.bitstring)
    
    @property
    def get_action(self):
          return self.action  
    @property
    def get_percept(self):
          return self.state 
    
    @property
    def get_visible(self):
         return self.action + self.state

    def _update_var(self, new_state:str):
        if len(new_state) == self.len_var:
                self.var = new_state
                self.hidden = new_state[0:len(self.hidden)] , self.action = new_state[-len(self.action):]
                self.bitstring = self.var + self.fixed
        else : raise ValueError("Updated 'var' should be of len "+str(self.len_var))

@dataclass(init=True)
class MCMCChain:

    def __init__(self, states: Optional[List[MCMCState]] = None):
        if len(states) is None:
            self._states: List[MCMCState] = []
            self._current_state: MCMCState = None
            self._states_accepted: List[MCMCState] = []
        else:
            self._states = states
            self._current_state : MCMCState = next((s for s in self._states[::-1] if s.accepted), None)
            self._states_accepted : List[MCMCState] = [ state for state in states if state.accepted]


    def add_state(self, state: MCMCState):
        if state.accepted:
            self._current_state = state
            self._states_accepted.append(state)
        self._states.append(state)


    @property
    def states(self):
        return self._states

    
    @property
    def current_state(self):
        return self._current_state


    @property
    def accepted_states(self) -> List[str]:
        # return [s.bitstring for s in self._states if s.accepted]
        return [state.bitstring for state in self._states_accepted]
    
    ### added by neel 13-Jan-2023
    @property
    def list_markov_chain_in_state(self)-> List[str]:
        markov_chain_in_state=[self.states[0].bitstring]
        for i in range(1,len(self.states)):
            mcmc_state=self.states[i].bitstring
            whether_accepted=self.states[i].accepted
            if whether_accepted==True:
                markov_chain_in_state.append(mcmc_state)
            else:
                markov_chain_in_state.append(markov_chain_in_state[i-1])
        return markov_chain_in_state
    ### added by neel 13-Jan-2023
    def emp_distn_markov_chain_dict(self,skip_first_few:int=0,normalize: bool=False):
        if normalize:
            length = len(self.list_markov_chain_in_state[skip_first_few:])
            empirical_distn_dict = Counter({s: count/length for s, count in Counter(self.list_markov_chain_in_state[skip_first_few:]).items()})
        else:
            empirical_distn_dict = Counter(self.list_markov_chain_in_state[skip_first_few:])
        return empirical_distn_dict


    
    def get_accepted_dict(self, normalize: bool=False, until_index: int = -1):# -> Counter[str, int]:
        if until_index != -1:
            accepted_states = [s.get_visible for s in self._states[:until_index] if s.accepted]
        else:
            accepted_states = [s.get_visible for s in self._states if s.accepted]

        if normalize:
            length = len(accepted_states)
            accepted_dict = Counter({s: count/length for s, count in Counter(accepted_states).items()})
        else:
            accepted_dict = Counter(accepted_states)

        return accepted_dict

##TODO
class RestrictedEnergyFunction(IsingEnergyFunction):
     

    pass

###################################################################################
## REDEFINE SAMPLING ROUTINES ##
###################################################################################

from typing import Union
from qumcmc.basic_utils import plot_bargraph_desc_order, plot_multiple_bargraphs
from qumcmc.prob_dist import DiscreteProbabilityDistribution, value_sorted_dict

# @dataclass
class RestrictedExactSampling(Exact_Sampling) :
     
    def __init__(self, model: IsingEnergyFunction, percept:str, dims: tuple, beta: float = 1) -> None:
          
          assert dims[2] == len(percept)
          self.beta = beta
          self.dim_percept = len(percept)
          self.percept = percept
          self.dim_hidden = dims[0]
          self.dim_action = dims[1]
          self.dim_var = self.dim_action + self.dim_hidden
          self.all_configs = [f"{k:0{self.dim_var}b}"+percept for k in range(0, 2 ** (self.dim_var))]

          super().run_exact_sampling(self.beta)

    def get_boltzmann_distribution(
        self, beta:float = 1.0, sorted:bool = False, save_distribution:bool = False , return_dist:bool= True, plot_dist:bool = False
        ) -> dict :
        """ Get normalised boltzmann distribution over states 

            ARGS:
            ----
            beta : inverse temperature (1/ T)
            sorted  : if True then the states are sorted in in descending order of their probability
            save_dist : if True then the boltzmann distribution is saved as an attribute of this class -> boltzmann_pd 
            plot_dist : if True then plots histogram corresponding to the boltzmann distribution

            RETURNS:
            -------
            'dict' corresponding to the distribution
        """
        
        bltzmann_probs = dict( [ ( state, super().get_boltzmann_factor(state, beta= beta) ) for state in tqdm(self.all_configs, desc= 'running over all possible configurations') ] )
        partition_sum=np.sum(np.array(list(bltzmann_probs.values())))
        prob_vals=list(np.array(list(bltzmann_probs.values()))*(1./partition_sum))

        bpd= dict(zip(self.all_configs, prob_vals ))
        bpd_sorted_desc= value_sorted_dict( bpd, reverse=True )
        
        if save_distribution :
            self.boltzmann_pd = DiscreteProbabilityDistribution(bpd_sorted_desc)

        if plot_dist:
                plt.figure(2)
                plot_bargraph_desc_order(bpd_sorted_desc, label="analytical",plot_first_few=30); plt.legend()
        
        if return_dist :   
            if sorted: 
                return bpd_sorted_desc
            else :
                return bpd    
          
    def sampling_summary(self, plot_dist:bool=True):
        
        if self.exact_sampling_status :
            tmp = np.array(list(self.boltzmann_pd.values()))
            count_non_zero = len(tmp[tmp > 0.01])
            
            print("=============================================")
            print("     MODEL : "+str(self.name)+" |  beta : "+str(self.beta) )
            print("=============================================")
            
            
            print("Num Most Probable States : " + str( count_non_zero )   )
            print("Percept :", self.percept)
            # print("Entropy : " + str( self.get_entropy() ))
            print("Dims : ", (self.dim_hidden, self.dim_action, self.dim_percept) )
            print("---------------------------------------------")

            if plot_dist:
                plot_bargraph_desc_order(self.boltzmann_pd, label= 'Boltzmann Dist.', plot_first_few= count_non_zero)

        else:
            raise RuntimeError("Please Run Exact Sampling at any specified temperature first")
   

     

@dataclass
class RestrictedSampling:
        
        model : IsingEnergyFunction
        len_hidden : int
        len_action: int
        len_state : int
        percept : str
        # iterations : int = 10000
        temperature : float = 1.00
        initial_state : Optional[Union[ str, MCMCState]] = None
        
                
        def __post_init__(self):
                
                assert self.len_hidden + self.len_action + self.len_state == self.model.num_spins

                if self.initial_state is None : 
                        self.initial_state = MCMCState(get_random_state(self.len_hidden), get_random_state(self.len_action) , self.percept , accepted= True)
                elif not isinstance(self.initial_state, MCMCState):
                        self.initial_state = MCMCState(self.initial_state, accepted=True)
                
                self.current_state: MCMCState = self.initial_state
                
                self.mcmc_chain: MCMCChain = MCMCChain([self.current_state])

                self.len_var = self.current_state.len_var; self.len_fixed = self.len_state 
                
                

        def run_classical_mcmc(self, iterations, verbose:bool = False):
                
                energy_s = self.model.get_energy(self.current_state.bitstring)
                if verbose : print('current state: ', self.current_state)
                for _ in tqdm(range(0, iterations), desc= 'running MCMC steps ...', disable= not verbose):
                        # get sprime #
                        s_prime = MCMCState(get_random_state(self.len_hidden), get_random_state(self.len_action) , self.current_state.fixed )
                        # print('s_prime:', s_prime)
                        
                        # accept/reject s_prime
                        energy_sprime = self.model.get_energy(s_prime.bitstring)   # to make this scalable, I think you need to calculate energy ratios.
                        accepted = test_accept(
                        energy_s, energy_sprime, temperature=self.temperature
                        )
                        if accepted:
                                s_prime.accepted = accepted
                                self.current_state = s_prime
                                if verbose : print('current state: ', self.current_state)
                                energy_s = self.model.get_energy(self.current_state.bitstring)
                        
                        self.mcmc_chain.add_state(s_prime)

                return self.mcmc_chain

        def _fn_qc_h2_restricted(self, J:np.array, alpha:float, gamma:float, delta_time) -> QuantumCircuit :
                """
                # updated version.
                Create a Quantum Circuit for time-evolution under
                hamiltonain H2 (described in the paper)

                ARGS:
                ----
                J: interaction matrix, interaction between different spins
                gamma: float
                alpha: float
                delta_time: (default= 0.8, as suggested in the paper)total evolution time time/num_trotter_steps

                """
                num_spins=np.shape(J)[0]
                qc_for_evol_h2=QuantumCircuit(num_spins)
                # calculating theta_jk
                upper_triag_without_diag=np.triu(J,k=1)
                theta_array=(-2*(1-gamma)*alpha*delta_time)*upper_triag_without_diag
                pauli_z_index=[3,3]## Z tensor Z
                for j in range(0, num_spins-1):
                        for k in range(j+1, num_spins):
                                #print("j,k is:",(j,k))
                                target_list=[num_spins-1-j,num_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
                                angle=theta_array[j,k]
                                qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle=angle)
                        
                return qc_for_evol_h2

      
        def _get_quantum_proposition(
            self, qc_initialised_to_s: QuantumCircuit, check_fixed: bool = True, max_checks: int = 1000
        ) -> str:

            """
            Takes in a qc initialized to some state "s". After performing unitary evolution U= exp(-iHt)
            , circuit is measured once. Function returns the bitstring s', the measured state .

            ARGS:
            ----
            qc_initialised_to_s:
            model:
            
            """

            h = self.model.get_h
            J = self.model.get_J

            # init_qc=initialise_qc(model.num_spins=model.num_spins, bitstring='1'*model.num_spins)
            gamma = np.round(np.random.uniform(0.25, 0.6), decimals=2)
            time = np.random.choice(list(range(2, 12)))  # earlier I had [2,20]
            delta_time = 0.8 
            num_trotter_steps = int(np.floor((time / delta_time)))
            qc_evol_h1 = fn_qc_h1(self.model.num_spins, gamma, self.model.alpha, h, delta_time)
            qc_evol_h2 = self._fn_qc_h2_restricted(J, self.model.alpha, gamma, delta_time=delta_time)
            trotter_ckt = trottered_qc_for_transition(
                self.model.num_spins, qc_evol_h1, qc_evol_h2, num_trotter_steps=num_trotter_steps
            )
            qc_for_mcmc = combine_2_qc(qc_initialised_to_s, trotter_ckt)# i can get rid of this!
            # run the circuit ##
            q_state=QuantumState(qubit_count=self.model.num_spins)
            q_state.set_zero_state()
            qc_for_mcmc.update_quantum_state(q_state)

            check_fixed_state = lambda bitstr : bitstr[ - self.initial_state.len_fixed: ] == self.percept
            if check_fixed :
                ## repeats sampling untill right fixed state is found ##
                right_sample = False; checks= 0
                while not right_sample :#and checks < max_checks:
                    state_obtained= q_state.sampling(sampling_count= 1)[0] ; checks+= 1
                    if check_fixed_state( f"{state_obtained:0{self.model.num_spins}b}" ) : right_sample = True
                    
            else :
                state_obtained= q_state.sampling(sampling_count= 1)[0]

            # state_obtained= [f"{state:0{model.num_spins}b}" for state in state_obtained]
            return f"{state_obtained:0{self.model.num_spins}b}"


        def run_quantum_enhanced_mcmc(self, iterations:int , num_post_selection_runs:int= 100, verbose:bool = False):

                energy_s = self.model.get_energy(self.current_state.bitstring)
                if verbose: print('current state: ', self.current_state)
                qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring= self.current_state.bitstring )
                for _ in tqdm(range(0, iterations), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
                        
                        # get sprime #
                        # qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring=self.current_state.bitstring)
                        s_prime = self._get_quantum_proposition(
                        qc_initialised_to_s=qc_s, max_checks= num_post_selection_runs
                        )
                        s_prime = MCMCState(s_prime[:self.len_hidden], s_prime[self.len_hidden: self.len_hidden+self.len_action], s_prime[self.len_hidden+self.len_action:] ) 
                        # if verbose: print('s_prime:', s_prime)

                        # accept/reject s_prime
                        energy_sprime = self.model.get_energy(s_prime.bitstring)
                        accepted = test_accept(
                        energy_s, energy_sprime, temperature=self.temperature
                        )
                        if accepted:
                                s_prime.accepted = accepted
                                self.current_state = s_prime
                                # print('current state: ', self.current_state)
                                energy_s = self.model.get_energy(self.current_state.bitstring)
                        
                        self.mcmc_chain.add_state(s_prime)
                                

                return self.mcmc_chain 


###################################################################################
## HELPER FUNCTIONS ##
###################################################################################

from torch import tensor
def percept_to_str(percept:tensor) :
    
    s = ''
    for elem in (percept[0].numpy().astype(int)) :
        s += str(elem)

    return s

def build_energy_model(net: DEBN):
    """ Works with only single hidden layer type RBMs """
    
    model_dim = net.visible.in_features + net.visible.out_features
    # named_params = list(net.named_parameters())
    param_dict = dict([(item[0], item[1].data) for item in net.named_parameters()])

    interactions = param_dict['visible.weight'].numpy()
    z0 = np.zeros((interactions.shape[0],interactions.shape[0])); z1 = np.zeros((interactions.shape[1],interactions.shape[1]))
    up = np.concatenate((z0, interactions), axis= 1); down = np.concatenate( (interactions.transpose(), z1), axis= 1)
    J = np.concatenate((up,down), axis= 0)

    biases = np.append(param_dict['visible.bias'].numpy(), param_dict['b_input.weight'].numpy() )

    ##checks
    assert model_dim == len(biases)
    assert model_dim == J.shape[0] 

    ##construct model
    model = IsingEnergyFunction(J, biases, "DEBN")

    return model