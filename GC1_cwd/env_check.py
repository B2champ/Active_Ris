import numpy as np
import math as mt
import copy
import torch
import torch.nn as nn

class RIS_MISO(object):
    #Tried changing the channel_noise_var to 4.402e-14
    def __init__(self, num_antennas, num_RIS_elements, num_groups, max_steps, num_users, channel_est_error=False, 
                 AWGN_var=mt.pow(10, -((230-30)/10)), channel_noise_var=mt.pow(10, -((120-30)/10)), carrfreq=10**9, 
                 alpha_t=2.2, alpha_r=2.2, Rican_BR=10, Rican_RU=10, Rican_BU=10, 
                 h_ris=20, h_base=10, x_bs=0, y_bs=0, x_ris=0, y_ris=150):
        
        # Basic Parameters
        self.max_steps = max_steps
        self.M = num_antennas
        self.N = num_RIS_elements
        self.Group = num_groups
        self.K = num_users
        self.channel_est_error = channel_est_error
        self.awgn_var = mt.pow(10, -((170-30)/10))
        self.channel_noise_var = channel_noise_var
        self.carrfreq = carrfreq
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.Rican_BR = Rican_BR
        self.Rican_RU = Rican_RU
        self.Rican_BU = Rican_BU
        self.h_ris = h_ris
        self.h_base = h_base
        self.x_bs = x_bs
        self.y_bs = y_bs
        self.x_ris = x_ris
        self.y_ris = y_ris
        self.eps = 1e-10
        self.max_value = 1e10

        # System Parameters
        self.SystemPower = mt.pow(10, (24-30)/10) 
        self.lambda_c = 3e8
        self.delta_not = self.lambda_c / 2 
        self.delta_a = self.lambda_c / 2
        self.C_0 = mt.pow(10, -30/10)
        self.regularization = 10**6
        self.weights = np.random.rand(self.K)
        self.weights /= np.sum(self.weights)
        # Derived Parameters
        self.d_BR = self.compute_distance(x_bs, y_bs, x_ris, y_ris)
        self.Qt_BU = self.C_0 * mt.pow(130, -alpha_t)
        self.delta_h = h_base - h_ris
        self.Qt_BR = self.C_0 * mt.pow(self.d_BR, -alpha_r)
        self.pshi_AoD_BR = mt.atan(self.delta_h / self.d_BR)
        self.phi_AoA_BR = self.pshi_AoD_BR
        self.Azimu_AoA_BR = mt.atan2(y_ris - y_bs, x_ris - x_bs)
        # Target Parameters
        self.set_target_location(-50, 50)
        # Actor and Observation Dimension Calculation
        self.calculate_action_and_state_dims()

        # Debug Channel Matrices
        self.H_1 = None
        self.H_2 = []
        self.H_D = None
        
        # Initialize BS and RIS steering vectors
        self.F_bs = self.compute_bs_steering_vector()
        self.F_ris = self.compute_ris_steering_vector()

        # Generate Channel Matrices
        self.generate_channel_matrices()


    def CalcGroupElements(self):
        if self.N % self.Group == 0:
            Num_GE = (self.N *self.N) // self.Group
            return Num_GE
        else:
            print("The number of group elements is not an integer")
            return None 

    def GroupConnected(self, action, Group):
        action = action.flatten()
        size = action.size
        # print(action)
        Group_elements = self.CalcGroupElements()  

        groupedaction = []
        for i in range(size):
            # Adjust replication factor here if needed to get exactly N*N elements
            groupedaction.extend([action[i]] * (self.N * self.N // size))  
        
        # Convert to a 1D NumPy array of the correct size
        groupedaction = np.array(groupedaction[:self.N * self.N])  # Trim or expand to N*N if needed
        print("Final groupedaction size:", groupedaction.size)  # Check size here
        return groupedaction.reshape(self.N, self.N)




    def compute_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def set_target_location(self, x_target, y_target):
        self.x_target = x_target
        self.y_target = y_target
        self.phi_target = mt.atan2(y_target - self.y_bs, x_target - self.x_bs)
        self.N_M = np.arange(self.M)
        self.steering_AOD_Target = np.exp(1j * 2 * np.pi * self.N_M * (self.lambda_c / 2) * 
                                          np.sin(self.phi_target) / self.lambda_c)
        
    def calculate_action_and_state_dims(self):
        power_size = 2 * self.K
        channel_size = 2 * (self.N * self.M + self.N * self.K)

        self.action_dim = 2 * self.M * self.K + self.Group * self.Group+ 2 * self.Group * self.Group
        self.state_dim = 2 * self.K + self.action_dim + 2 * self.N* self.M + 2 * self.N* self.K
        # print("state dimension check", self.state_dim)
        # self.state_dim = 2 * self.K + self.N * self.N + 2 * self.N * self.N + 2 * self.M * self.K + 2 * self.N* self.M + 2 * self.N* self.K

    def compute_bs_steering_vector(self):
        F_bs = np.exp(-1j * np.arange(self.M) * 2 * np.pi * self.delta_a * 
                      np.sin(self.pshi_AoD_BR) / self.lambda_c)
        return F_bs.reshape(-1, 1)

    def compute_ris_steering_vector(self): 
        F_ris = np.exp(-1j * 2 * np.pi * self.delta_not * 
                       (np.arange(self.N)[:, np.newaxis] * np.sin(self.phi_AoA_BR) * np.cos(self.Azimu_AoA_BR) + 
                        np.arange(self.N)[:, np.newaxis] * np.cos(self.phi_AoA_BR)) / self.lambda_c)
        return F_ris

    def generate_channel_matrices(self):
        # RIS to BS Channel
        self.H_br_Los = np.sqrt(self.Qt_BR) * (self.F_ris @ self.F_bs.T)
        self.H_br_NLos = np.sqrt(self.Qt_BR) * (np.random.randn(self.N, self.M) + 1j * np.random.randn(self.N, self.M))
        self.H_1 = np.sqrt(self.Rican_BR / (1 + self.Rican_BR)) * self.H_br_Los + \
                   np.sqrt(1 / (1 + self.Rican_BR)) * self.H_br_NLos
        
        # RIS to User Channel
        for _ in range(self.K):
            H_RU = self.generate_user_channel()
            self.H_2.append(H_RU)
        self.H_2 = np.squeeze(self.H_2).T

        # Direct BS to User Channel
        self.H_D = self.generate_station_to_user_channel()

    def generate_user_channel(self):
        radius = 10
        user_radius = radius * np.sqrt(np.random.rand())
        theta = np.pi + np.pi * np.random.rand()
        x_user = self.x_ris + user_radius * np.cos(theta)
        y_user = self.y_ris + user_radius * np.sin(theta)

        d_RU = self.compute_distance(self.x_ris, self.y_ris, x_user, y_user)
        Qrk_RU = self.C_0 * d_RU**-self.alpha_t
        pshi_AoD_RU = np.arctan(self.h_ris / d_RU)
        Azimu_AoA_RU = np.arctan2(self.y_ris - y_user, self.x_ris - y_user)

        F_risD = np.exp(-1j * 2 * np.pi * self.delta_not *
                        (np.arange(self.N)[:, None] * np.sin(pshi_AoD_RU) * np.cos(Azimu_AoA_RU) +
                         np.arange(self.N)[:, None] * np.cos(pshi_AoD_RU)) / self.lambda_c)
        
        h_RU_LoS = np.sqrt(Qrk_RU) * F_risD.T
        h_RU_NLoS = np.sqrt(Qrk_RU) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))
        
        return np.sqrt(self.Rican_RU / (1 + self.Rican_RU)) * h_RU_LoS + \
               np.sqrt(1 / (1 + self.Rican_RU)) * h_RU_NLoS 

    def generate_station_to_user_channel(self):
        # Direct path between Base Station and User (BS-User), Rician channel
        H_D_Los = np.sqrt(self.Qt_BU) * (np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K))
        H_D_NLoS = np.sqrt(self.Qt_BU) * (np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K))
        
        return np.sqrt(self.Rican_BU / (1 + self.Rican_BU)) * H_D_Los + \
               np.sqrt(1 / (1 + self.Rican_BU)) * H_D_NLoS

        #Compute Tilda 
    def _compute_H_2_tilde(self):
         return self.H_2.T @self.Phi@self.H_1 @self.G  
    
    
    def reset(self):
        self.episode_t = 0

        # Active RIS Component
        self.Phi = np.exp(1j * np.random.uniform(0, np.pi, (self.Group, self.Group)))
        self.AmplificationPower = 10 ** ((5 - 30) / 10)
        # Generate a random NxN complex matrix for Amplification Factor
        self.AF = np.random.uniform(1, 16, size=(self.Group, self.Group))
        
        # Apply Amplification Factor to Phi
        self.Phi = self.Phi @ self.AF

        # Generate random complex channel matrix G
        self.G = np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K)
        
        # Normalize each column such that its power is equal to SystemPower/K
        column_powers = np.sum(np.abs(self.G)**2, axis=0)
        normalization_factors = np.sqrt(self.SystemPower / (self.K * column_powers))
        self.G = self.G * normalization_factors

        # Initial action construction
        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_ampli_AF = np.real(self.AF.reshape(1, -1))
        init_action_Phi = np.hstack((np.real(self.Phi.reshape(1, -1)), np.imag(self.Phi.reshape(1, -1))))
        init_action = np.hstack((init_action_G, init_ampli_AF, init_action_Phi))

      # Extract amplified factors and phase shifts
        ampli_AF = init_action[:, 2 * self.M * self.K: 2 * self.M * self.K + self.Group*self.Group]
        Phi_real = init_action[:, -2 * self.Group*self.Group: -self.Group*self.Group]
        Phi_imag = init_action[:, -self.Group*self.Group:]
        
        ampli_AF = self.GroupConnected(ampli_AF, self.Group)
        Phi_real = self.GroupConnected(Phi_real, self.Group)
        Phi_imag = self.GroupConnected(Phi_imag, self.Group)

        # Reshape AF and Phi
        self.AF = np.reshape(ampli_AF, (self.N, self.N))
        self.Phi = np.reshape(Phi_real + 1j * Phi_imag, (self.N, self.N))

        self.Phi = self.Phi @ self.AF

        # Compute power and channel matrices for state representation
        power_t = np.linalg.norm((self.G), axis=0).reshape(1, -1)**2
        H_2_tilde = self._compute_H_2_tilde()
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1)**2
        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        # Construct the state
        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state
 

    
   
    def _compute_reward(self, Phi):

        opt_reward = 0
        sum_prob = 0
        sum_rate = 0

        weight1 = 1e6
        weight2 = 1

        def safe_norm(x):
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                return 0
            return np.linalg.norm(x)
        
        def safe_log2(x):
            return np.log2(np.maximum(x, self.eps))

        a = self.steering_AOD_Target.reshape(-1, 1)
        H_2_tilde = self._compute_H_2_tilde()

        H_2_tilde = np.clip(H_2_tilde, -self.max_value, self.max_value)
        
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
        power_t = np.sum(np.abs(self.G) ** 2, axis=0).reshape(1, -1)
        active_interfer = 0
   

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)
            
            numerator = np.linalg.norm(np.abs((h_2_k.T @ Phi @ self.H_1 + self.H_D[:, k]) @ g_k)) ** 2
            
            numerator = np.clip(numerator, -self.max_value, self.max_value)
            numerator = safe_norm(numerator)
            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.linalg.norm(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            
            denominator = interference + self.awgn_var + np.linalg.norm(h_2_k.T @ Phi) ** 2 * self.channel_noise_var +  self.eps
            print("Numerator:", str(numerator), "Denominator:", str(denominator))

            rho_k = numerator / denominator # Calculate SNR
            s_k = self.weights[k] * np.log2(1 + rho_k)
            sum_rate += s_k
            # print(f"Rate for user {k}: {self.weights[k] * self.regularization * np.log2(1 + rho_k)}")
            print(f"Rate for user {k}: {s_k}")

            # Calculate probing power for K-th user
            prob = np.linalg.norm(a.T @ g_k @ g_k.T @ a)
            sum_prob += prob 
            print("probing Power", prob)

            opt_reward += self.weights[k] * np.log2(1 + numerator / ((self.K - 1) * self.awgn_var))
            
            # Update active interference
            active_interfer += np.linalg.norm(self.Phi.T @ self.H_1 @ g_k) ** 2
            
        
        # Check power constraints
        active_interfer += (np.linalg.norm(self.Phi.T) ** 2) * self.awgn_var

        reward = 0
        violation = 0
        if active_interfer <= self.AmplificationPower:
            reward = weight1*sum_rate + weight2*sum_prob
        else:
        # Soft penalty for violating power constraint
            violation = (active_interfer - self.AmplificationPower) / self.AmplificationPower
            reward = (weight1 * sum_rate + weight2 * sum_prob) / (1 + violation)**6 +self.eps
            print("violation Detected")
        
        opt_reward = opt_reward*weight1 + weight2*sum_prob

        violation = (active_interfer - self.AmplificationPower) / self.AmplificationPower
        print('if Violation',(weight1 * sum_rate + weight2 * sum_prob) / (1 + violation)**6 + self.eps) 
    
  

          

        # # Diagnostics
        print('################## Diagnostics ##################')
        print(f'Power Initially Transmitted: {self.SystemPower:.10f}')
        print(f'  Power         Transmitted: {np.sum(power_t):.10f}')
        print(f'      Power        Received: {np.sum(power_r):.10f}')
      
        print(f'Amplification Power Initial: {self.AmplificationPower:.10f}')
        print(f'Active  Power  Interference: {active_interfer:.10f}')
        print(f'Sum  of  Probing      Power: {sum_prob:.10f}')
        print(f'Sum  of  Probing      Power: {sum_prob*weight2:.10f}')
        print(f'       Sum  of  Rates      : {sum_rate:.10f}')
        print(f'        Reward             : {reward:.10f}')
        print(f'   Optimal Reward          : {opt_reward:.10f}')
        print(f'violation                  : {violation:.10f}')
        print('#################################################')
        
        return reward, opt_reward

    
    def step(self, action):
        self.episode_t += 1
        
        # action = action.reshape(1,-1)
        G_real = action[:, :self.M *self.K]
        G_imag = action[:, self.M *self.K: 2*self.M *self.K]
        Phi_real = action[:, -2*self.Group*self.Group: -self.Group*self.Group]
        Phi_imag = action[:, -self.Group*self.Group:]
      
        ampli_AF= action[:, 2*self.M *self.K  : 2*self.M *self.K+ self.Group*self.Group]

        # ampli_AF = ((ampli_AF + 1.0) * 1) + 1.0  # Maps [-1, 1] to [1, 3]
        ampli_AF = (ampli_AF + 1.0)  # Maps [-1, 1] to [1, 3]
        
        ampli_AF = np.clip(ampli_AF, 1.0, 15.0)
        
        ampli_AF = self.GroupConnected(ampli_AF, self.Group)
        
        self.AF = np.reshape(ampli_AF, (self.N, self.N))
        
        self.AF=np.abs(self.AF)
        Phi_real = self.GroupConnected(Phi_real, self.Group)
        print(np.shape(Phi_real))
        Phi_imag = self.GroupConnected(Phi_imag, self.Group)
        

        self.Phi = np.exp(1j * np.angle(Phi_real + 1j * Phi_imag, (self.N, self.N)))
        self.Phi = np.exp(1j * np.angle(np.reshape(Phi_real + 1j * Phi_imag, (self.N, self.N))))
   
        self.Phi = (self.Phi + self.Phi.T.conj()) / 2 # Making it symettrical 
        
        self.G = G_real.reshape(self.M, self.K) +1j* G_imag.reshape(self.M, self.K)


        # self.G = safe_reshape(G_real + 1j * G_imag, (self.M, self.K))
        # self.G = np.clip(self.G, -self.max_value, self.max_value)

        # G_norm = np.linalg.norm(self.G)
        # if G_norm > self.max_value:
        #     self.G = self.G * (self.max_value / G_norm)
        # self.Phi = np.clip(self.Phi, -self.max_value, self.max_value)
        
        
        power_t = power_t = np.linalg.norm((self.G), axis=0).reshape(1, -1)**2
        
        

        H_2_tilde = self._compute_H_2_tilde()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("shape of G", self.G.shape)
        print("shape of Phi", self.Phi.shape)
        print("shape of AF", self.AF.shape)
        print("Norm of H_2.T:", np.linalg.norm(self.H_2.T))
        print("Norm of Phi:", np.linalg.norm(self.Phi))
        print("Norm of H_1:", np.linalg.norm(self.H_1))
        print("Norm of G:", np.linalg.norm(self.G))
        print("Amplification Factor:", np.sum(np.abs(self.AF)))
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1)**2

        print("Norm of power_r:", np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1))
        print("Norm of power_t:", power_t)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1,-1), np.imag(self.H_1).reshape(1,-1) #Why reshape?
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1,-1), np.imag(self.H_2).reshape(1,-1) #Why reshape?

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        print("state shape ", self.state.shape)
        

        reward, opt_reward = self._compute_reward(self.Phi)

        done = opt_reward == reward
        done = self.episode_t >= self.max_steps
        return self.state, reward, done, None
    def close(self):
        pass
        