import numpy as np
import math as mt

class RIS_MISO(object):
    def __init__(self, 
                 num_antennas,
                 num_RIS_elements,
                 num_users,
                 channel_est_error=False,
                 AWGN_var=mt.pow(10, (-(110-30/10))),
                 channel_noise_var=1e-2,carrfreq=10**9,
                 alpha_t=2.2,
                 alpha_r=2.2,
                 Rican_BI=10,
                 Rican_IU=10,
                 Rican_BU=10,
                 h_ris=20,
                 h_base=10,
                 x_bs=0, y_bs=0,
                 x_irs=0, y_irs=150,):
        # System parameters
        self.SystemPower=mt.pow(10,(20-30)/10)

        self.awgn_var=mt.pow(10, (-(110-30/10)))
        self.carrfreq = carrfreq
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.Rican_BI = Rican_BI
        self.Rican_IU = Rican_IU
        self.Rican_BU = Rican_BU
        self.h_ris = h_ris
        self.h_base = h_base
        self.x_bs = x_bs
        self.y_bs = y_bs
        self.x_irs = x_irs
        self.y_irs = y_irs
        # Number of elements and users
        self.M = num_antennas
        self.N= num_RIS_elements
        self.K = num_users
        # weight factor for sinr
        self.weight = np.random.rand(4)
        # Flags
        self.channel_est_error = channel_est_error
        
        # Noise parameters
        self.AWGN_var = AWGN_var
        self.channel_noise_var = channel_noise_var
        
        # Derived parameters
        self.lambda_c = 3 * 10**8 / self.carrfreq  # Wavelength
        self.delta_not = self.lambda_c / 2  # Element spacing at BS
        self.delta_a = self.lambda_c / 2  # Element spacing at IRS
        self.delta_h = self.h_base - self.h_ris  # Height difference between base station and IRS


        
        # Path loss at reference distance 1 meter
        self.C_0 = 10**(-30/10)  # Path loss at reference distance in dB

        # Compute distance between BS and IRS
        self.d_BI = np.sqrt((self.x_irs - self.x_bs)**2 + (self.y_irs - self.y_bs)**2)

        # Compute large-scale path loss for BS-IRS link
        self.Qt_BI = self.C_0 * self.d_BI**(-self.alpha_r)

        # Compute angle of departure (AoD) for BS-IRS link
        self.pshi_AoD_BI = np.arctan(self.delta_h / self.d_BI)

        # Compute BS steering vector
        self.F_bs = np.exp(-1j * np.arange(self.M) * 2 * np.pi * self.delta_a * np.sin(self.pshi_AoD_BI) / self.lambda_c)


        # Compute angle of departure (AoD) for BS-IRS link
        self.pshi_AoD_BI = np.arctan(self.delta_h / self.d_BI) # Why Required, Reapeted 

        # Compute angle of arrival (AoA) for BS-IRS link
        self.phi_AoA_BI = self.pshi_AoD_BI
        self.Azimu_AoA_BI = np.arctan2(self.y_irs - self.y_bs, self.x_irs - self.x_bs)

        # Compute BS steering vector     
        self.F_bs = np.exp(-1j * np.arange(self.M) * 2 * np.pi * self.delta_a * np.sin(self.pshi_AoD_BI) / self.lambda_c)
        # print(self.F_bs)
        self.F_bs = self.F_bs.reshape(-1, 1)  # Reshape to Mx1 for correct dimensions
        n = np.arange(self.N)
        self.F_ris = np.exp(-1j * 2 * np.pi * self.delta_not * (np.arange(self.N)[:, np.newaxis] * np.sin(self.phi_AoA_BI) * np.cos(self.Azimu_AoA_BI) + np.arange(self.N)[:, np.newaxis] * np.cos(self.phi_AoA_BI)) / self.lambda_c)  # Column vector for IRS steering
        

        # Ensure F_ris and F_bs dimensions are compatible for multiplication
        self.H_br_LoS = np.sqrt(self.Qt_BI) * (self.F_ris @ self.F_bs.T)  # LoS component of the channel matrix

        # Compute NLOS components for BS to IRS
        self.H_br_NLOS = np.sqrt(self.Qt_BI) * (
            np.random.randn(self.N, self.M) + 1j * np.random.randn(self.N, self.M)
        )

       

        power_size = 2 * self.K

        channel_size = 2 * (self.N* self.M +self.N* self.K)

        self.action_dim = 2 * self.M * self.K + 2 * self.N
        self.state_dim = power_size + channel_size + self.action_dim
        
        self.H_1 = None
        self.H_2 = None
        self.G = np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K)
        self.Phi = np.eye(self.N, dtype=complex)
        
        norm_G = np.linalg.norm(self.G, 'fro')
        self.G = self.G * (self.SystemPower / norm_G)
        self.state = None
        self.done = None

        self.episode_t = None
        
        self.pa = mt.pow(10,(5-30)/10)
        self.power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        self.af=np.eye(self.N, dtype=complex)

        self.af = np.clip(self.af,0 , np.sqrt( self.pa))
    def _compute_H_2_tilde(self):
        return self.H_2.T @ self.Phi * self.af @ self.H_1 @ self.G












    def reset(self):
        self.episode_t = 0
        # H_1 is channel to base to RIS
        self.H_1 = np.sqrt(self.Rican_BI / (1 + self.Rican_BI)) * self.H_br_LoS + np.sqrt(1 / (1 +self. Rican_BI)) * self.H_br_NLOS  # Combined LoS and NLoS
        
        # User position and channel calculations
        self.H_2 = []
        self.H_BU_ALL = []
        for _ in range(self.K):
            # Generate random user position around RIS
            radius = 10  # Radius around RIS for user positions
            user_radius = radius * np.sqrt(np.random.rand())
            theta = np.pi + np.pi * np.random.rand()
            x_user = self.x_irs + user_radius * np.cos(theta)
            y_user = self.y_irs + user_radius * np.sin(theta)

            # Calculate distances
            d_IU = np.sqrt((self.x_irs - x_user)**2 + (self.y_irs - y_user)**2)  # Distance from IRS to user

            # Calculate large-scale path losses
            Qrk_IU = self.C_0 * d_IU**(-self.alpha_t)  # Path loss for IRS to user link

            # Calculate LOS components for IRS to user
            pshi_AoD_IU = np.arctan(self.h_ris / d_IU)  # AoD from IRS to user
            Azimu_AoD_IU = np.arctan2(self.y_irs - y_user, self.x_irs - x_user)  # Azimuth AoD
            F_risD = np.exp(-1j * 2 * np.pi * self.delta_not * (
                np.arange(self.N)[:, None] * np.sin(pshi_AoD_IU) * np.cos(Azimu_AoD_IU) +
                np.arange(self.N)[:, None] * np.cos(pshi_AoD_IU)
            ) / self.lambda_c)  # Column vector for IRS steering
            h_IU_LoS = np.sqrt(Qrk_IU) * F_risD.T  # LoS component of the channel matrix

            # Calculate NLOS components for IRS to user
            h_IU_NLoS = np.sqrt(Qrk_IU) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))  # NLoS component of the channel matrix

            # Combined channel from IRS to user
            H_IU = np.sqrt(self.Rican_IU / (1 + self.Rican_IU)) * h_IU_LoS + np.sqrt(1 / (1 + self.Rican_IU)) * h_IU_NLoS
            self.H_2.append(H_IU)
        H_combined = np.squeeze(self.H_2)    
        self.H_2=H_combined.T

            

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack(
            (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))

        Phi_real = init_action[:, -2 * self.N:-self.N]
        Phi_imag = init_action[:, -self.N:]

        self.Phi = np.eye(self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        self.Phi *= self.af
        print( np.linalg.norm(self.Phi, 'fro'))

        H_2_tilde = self._compute_H_2_tilde()
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2

            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            
            rho_k = (x / y) * 0.25

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)



        return reward, opt_reward

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)

        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.N:-self.N]
        Phi_imag = action[:, -self.N:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        # print(self.af)
        # print(self.power_t)
       
        # print(self.Phi)
        # print(np.shape(self.Phi))
        # print(np.shape(self.power_t))
        self.power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        self.Phi = np.eye(self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)*self.af
        # print(self.af)
        # print(self.Phi)
        # print(self.power_t[0][0])
        H_2_tilde = self._compute_H_2_tilde()
        print(np.linalg.norm(self.Phi, 'fro'))
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((action, self.power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass
