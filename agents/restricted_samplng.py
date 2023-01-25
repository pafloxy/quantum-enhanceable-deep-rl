###################################################################################
## IMPORTS ##
###################################################################################

import numpy as np
from typing import Optional
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass

from qumcmc.basic_utils import qsm, states, MCMCChain
# from .prob_dist import *
from qumcmc.energy_models import IsingEnergyFunction
from qumcmc.classical_mcmc_routines import test_accept, get_random_state
from qumcmc.quantum_mcmc_routines_qulacs import *
from typing import List


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



###################################################################################
## REDEFINE SAMPLING ROUTINES ##
###################################################################################

from typing import Union


@dataclass
class RestrictedSampling:
        
        model : IsingEnergyFunction
        len_hidden : int
        len_action: int
        len_state : int
        # iterations : int = 10000
        temperature : float = 1.00
        initial_state : Optional[Union[ str, MCMCState]] = None
        
                
        def __post_init__(self):
                
                assert self.len_hidden + self.len_action + self.len_state == self.model.num_spins

                if self.initial_state is None : 
                        self.initial_state = MCMCState(get_random_state(self.model.num_spins), accepted=True)
                elif not isinstance(self.initial_state, MCMCState):
                        self.initial_state = MCMCState(self.initial_state, accepted=True)
                
                self.current_state: MCMCState = self.initial_state
                
                self.mcmc_chain: MCMCChain = MCMCChain([self.current_state])

                self.len_var = self.current_state.len_var; self.len_fixed = self.len_state 
                
                

        def run_classical_mcmc(self, iterations):
                
                energy_s = self.model.get_energy(self.current_state.bitstring)
                print('current state: ', self.current_state)
                for _ in tqdm(range(0, iterations), desc= 'running MCMC steps ...'):
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
                                print('current state: ', self.current_state)
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
                for j in range(self.len_hidden, num_spins):
                        for k in range(j+1, self.len_hidden):
                                #print("j,k is:",(j,k))
                                target_list=[num_spins-1-j,num_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
                                angle=theta_array[j,k]
                                qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle=angle)
                        
                return qc_for_evol_h2

      
        def _get_quantum_proposition(
            self, qc_initialised_to_s: QuantumCircuit, check_fixed: bool = True, max_checks: int = 100
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

            check_fixed_state = lambda bitstr : bitstr[ - self.initial_state.len_fixed: ] == self.initial_state.fixed
            if check_fixed :
                ## repeats sampling untill right fixed state is found ##
                right_sample = False; checks= 0
                while not right_sample and checks < max_checks:
                    state_obtained= q_state.sampling(sampling_count= 1)[0] ; checks+= 1
                    if check_fixed_state( f"{state_obtained:0{self.model.num_spins}b}" ) : right_sample = True
                    
            else :
                state_obtained= q_state.sampling(sampling_count= 1)[0]

            # state_obtained= [f"{state:0{model.num_spins}b}" for state in state_obtained]
            return f"{state_obtained:0{self.model.num_spins}b}"


        def run_quantum_enhanced_mcmc(self, iterations:int , verbose:bool = False):

                energy_s = self.model.get_energy(self.current_state.bitstring)
                if verbose: print('current state: ', self.current_state)
                qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring= self.current_state.bitstring )
                for _ in tqdm(range(0, iterations), desc='runnning quantum MCMC steps . ..' ):
                        
                        # get sprime #
                        # qc_s = initialise_qc(n_spins= self.model.num_spins, bitstring=self.current_state.bitstring)
                        s_prime = self._get_quantum_proposition(
                        qc_initialised_to_s=qc_s
                        )
                        s_prime = MCMCState(s_prime[:self.len_hidden], s_prime[self.len_hidden: self.len_hidden+self.len_action], s_prime[self.len_hidden+self.len_action:] ) 
                        if verbose: print('s_prime:', s_prime)

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
                

