"""
:Submodule: MaterialClass
:Author: Y. Ben Zineb
:Email: yzineb3@gatech.edu

================================================= V1.0 -- Jul. 6th 2025 ================================================
This submodule defines the MaterialClass. This object is meant to be a container for all the material parameters.
This Class also tracks the history of strain, stress and other high priority parameters throughout the different
increments of a thermomechanical loading. The structure of this material description is inspired by the work of
Chemisky et al. (2011) [https://doi.org/10.1016/j.mechmat.2011.04.003]
A. Duval PhD Thesis (2009) [https://docnum.univ-lorraine.fr/public/SCD_T_2009_0132_DUVAL.pdf]
Y. Chemisky PhD Thesis (2009) [https://docnum.univ-lorraine.fr/public/UPV-M/Theses/2009/Chemisky.Yves.SMZ0943.pdf]
"""

import numpy as np

from src.BasicFunctions import CalculateElasticityTensor, CalculateComplianceTensor, Deviator, EquivalVonMisesStrain, EquivalVonMisesStress, CalculateJ2, CalculateJ3
from src.Wrapper import PackInputs, Solve

class Material:
    """
    Class defining the material properties used in the wrapper to determine the thermomechanical response
    """
    def __init__(self, name, E, nu, alpha, F_epsilon, H_s, H_f, H_twin, H_epsilon_T, epsilon_trac_T,
                 epsilon_trac_T_FA, epsilon_comp_T, Br, Bf, Ms, Af, rf, n, alpha0, alpha1, alpha2, Tinit, br, bf):
        """
        Function to initialize the Material object with material properties
        :param name: string -- Name given to the Material object
        :param E: float -- Young Modulus (MPa)
        :param nu: float -- Poisson Ratio (Between 0 and 1)
        :param alpha: float -- Thermal Dilation factor (K^-1)
        :param F_epsilon: -- Critical value for F_epsilon_bar_T
        :param H_s: float -- Gap between austenite and oriented martensite return temperatures (MPa)
        :param H_f: float -- Transformation pseudo work-hardening (MPa)
        :param H_twin: float -- Twins accommodation pseudo work-hardening (MPa)
        :param H_epsilon_T: float -- Reorientation pseudo work-hardening (MPa)
        :param epsilon_trac_T: float -- Maximum transformation strain in traction
        :param epsilon_trac_T_FA: float -- Maximum transformation strain in traction from formed and auto accommodated martensite
        :param epsilon_comp_T: float -- Maximum transformation strain in compression
        :param Br: float -- Slope of reverse transformation in traction in the (sigma, T) diagram (MPa.K^-1)
        :param Bf: float -- Slope of forward transformation in compression in the (sigma, T) diagram (MPa.K^-1)
        :param Ms: float -- Austenite -> Martensite transformation start temperature (K)
        :param Af: float -- Martensite -> Austenite transformation end temperature (K)
        :param rf: float -- Internal loops amplitude
        :param n: float -- Order coefficient for calculation of saturation strain
        :param Tref: float -- Reference temperature (K)
        """

        self.name = name

        self.n = n

        self.C = CalculateElasticityTensor(E, nu)
        self.S = CalculateComplianceTensor(E, nu)
        self.alpha = alpha
        self.T = Tinit

        if br is not None :
            self.Br = br * epsilon_trac_T
        else:
            self.Br = Br

        if bf is not None:
            self.Bf = bf * epsilon_trac_T
        else:
            self.Bf = Bf

        self.T0 = ((self.Bf * Ms) + (self.Br * Af)) / (self.Bf + self.Br)

        self.H_s = H_s
        self.H_f = H_f
        self.H_twin = H_twin
        self.H_epsilon_T = H_epsilon_T

        self.F_epsilon = F_epsilon
        self.F_f_pred = 0

        self.epsilon_trac_T = epsilon_trac_T
        self.epsilon_trac_T_FA = epsilon_trac_T_FA
        self.epsilon_comp_T = epsilon_comp_T

        self.Ms = Ms
        self.Af = Af

        self.rf = rf

        self.f = 0
        self.f_FA = 0
        self.fdot = 0

        self.epsilon_bar_T = np.zeros((3, 3))
        self.epsilon_bar_twin = np.zeros((3, 3))
        self.sigma = np.zeros((3, 3))
        self.epsilon = alpha * np.eye(3) * (Tinit - self.T0)

        self.hist_dzetaFA = []
        self.dzetaFA = 0

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.hist_epsilon_bar_T = []
        self.hist_epsilon_bar_twin = []
        self.hist_epsilon = []
        self.hist_sigma = []
        self.hist_f = []
        self.hist_f_FA = []
        self.hist_T = []
        self.hist_F_f_pred = []
        self.hist_Tr = []
        self.hist_Or = []
        self.hist_lambda2 = []

        self.gamma = (epsilon_trac_T / epsilon_comp_T) ** n
        self.K = epsilon_trac_T * (1 + ((1 - self.gamma) / (self.gamma + 1))) ** (-1 / n)

        self.F_f_max = self.Bf * (self.T0 - Ms)

        self.B = 2 * self.F_f_max / (Af - Ms)
        self.F_f = - self.B * (self.T - self.T0)


    def get_S(self):
        return self.S


    def ClassifyCase(self, sigma_elast_pred):
        Tr = 0
        Or = 0

        sigma_elast_pred_D = Deviator(sigma_elast_pred)
        epsilonVM = EquivalVonMisesStrain(self.epsilon_bar_T)

        # TODO: How to handle J3 < 0, How to handle epsilon_T_sat at t=0 ? Only add them to f_f when needed (Or=1 or Tr!=0)
        # J2_epsilon_bar_T = CalculateJ2(self.epsilon_bar_T)
        # J3_epsilon_bar_T = CalculateJ3(self.epsilon_bar_T)
        # epsilon_T_sat = self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * J3_epsilon_bar_T / (J2_epsilon_bar_T ** (3 / 2))) ** (1 / self.n)
        # eta_epsilon = 1 - (epsilon_T_sat / self.epsilon_trac_T_FA)
        # epsilon_T_max = epsilon_T_sat * (1 - (self.f_FA / self.f) * eta_epsilon) if self.f != 0 else epsilon_T_sat
        # lambda0 = self.alpha0 * ((self.f - 1) / self.f)
        # lambda1 = self.alpha1 * (self.f / (1 - self.f))
        # lambda2 = self.alpha2 * (self.epsilon_bar_T / (epsilon_T_max - epsilonVM)) * self.epsilon_bar_T

        # TODO: How to handle f = 0 ?
        F_f_pred = (np.einsum("ij, ij", sigma_elast_pred, self.epsilon_bar_T)
                    + self.dzetaFA * np.einsum("ij, ij", sigma_elast_pred, self.epsilon_bar_twin)
                    - self.B * (self.T - self.T0)
                    - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", self.epsilon_bar_T, self.epsilon_bar_T)
                    - self.H_f * self.f
                    - 0.5 * self.H_twin * self.dzetaFA * np.einsum("ij, ij", self.epsilon_bar_twin,
                                                                   self.epsilon_bar_twin))

        F_epsilon_bar_T_pred = (sigma_elast_pred_D
                                - self.H_epsilon_T * self.epsilon_bar_T)


        # F_epsilon_bar_twin_pred = (sigma_elast_pred_D
        #                      - self.H_twin * self.epsilon_bar_twin)

        print(f"Prediction of F_f: {F_f_pred}", f"Previous F_f:{self.F_f}")
        print(f"Delta F_f:{F_f_pred - self.F_f}")

        if F_f_pred - self.F_f > 0:
            right_hand_term = self.F_f_max + (self.Bf - self.B) * (self.T - self.T0) - self.H_s * epsilonVM
            print("F_f - kappaf:", F_f_pred)
            print("Right-hand term:", right_hand_term)
            if F_f_pred > right_hand_term:
                Tr = 1

        elif F_f_pred - self.F_f < 0:
            right_hand_term = -self.F_f_max + (self.Br - self.B) * (self.T - self.T0) - self.H_s * epsilonVM
            print("F_f - kappaf:", F_f_pred )
            print("Right-hand term:", right_hand_term)
            if F_f_pred < right_hand_term:
                Tr = -1

        if EquivalVonMisesStress(F_epsilon_bar_T_pred) > self.F_epsilon:
            Or = 1

        self.F_f_pred = F_f_pred

        return Tr, Or

    def OutputResiduals(self, Tr, Or, sigma_elast_pred):
        residuals = {}

        def R_sigma(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
            S = self.get_S()
            H = CalculateComplianceTensor(self.H_twin, 0.5)
            delta = np.eye(3)

            sigma_next = self.sigma + Delta_Sigma
            sigma_dev_next = Deviator(sigma_next)
            epsilon_bar_twin_next = np.einsum("ijkl, kl -> ij", H, sigma_dev_next)

            f_next = self.f + Delta_f

            epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
            epsilon_T_VM_next = EquivalVonMisesStrain(f_next * epsilon_bar_T_next)

            J2_next = CalculateJ2(epsilon_bar_T_next)
            J3_next = CalculateJ3(epsilon_bar_T_next)

            epsilon_T_sat_next = self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * (J3_next / (J2_next ** (3 / 2)))) ** (1 / self.n)

            f_FA_next = self.f_FA + ((epsilon_T_sat_next - epsilon_T_VM_next) / epsilon_T_sat_next) * Delta_f if Delta_f > 0 else (self.f_FA + (
                        self.f_FA / f_next) * Delta_f)

            term1 = np.einsum("ijkl, kl -> ij", S, sigma_next)
            term2 = self.alpha * delta * (self.T - self.T0)
            term3 = f_next * epsilon_bar_T_next
            term4 = f_FA_next * epsilon_bar_twin_next

            return term1 + term2 + term3 + term4 - self.epsilon

        if Or == 0:
            def R_delta_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T

                f_next = self.f + Delta_f

                epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)

                return f_next * Delta_epsilon_bar_T - (2 / 3) * Delta_lambda_epsilon_T * (epsilon_bar_T_next / epsilon_bar_T_eq_next)
            if Tr == -1:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Br - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return -F_f_next - self.F_f_max + term1 - term2

                def R_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    return EquivalVonMisesStrain(Delta_epsilon_bar_T)

            elif Tr == 0:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    return Delta_f

                def R_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    return EquivalVonMisesStrain(Delta_epsilon_bar_T)

            else:
                def R_f (Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Bf - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return F_f_next - self.F_f_max - term1 - term2

                def R_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    f_next = self.f + Delta_f
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T

                    term1 = Delta_f * epsilon_bar_T_next
                    term2 = f_next * Delta_epsilon_bar_T
                    term3 = Delta_f * Delta_epsilon_bar_T

                    return EquivalVonMisesStrain(term1 + term2 + term3)

        else:
            def R_delta_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T

                sigma_next = self.sigma + Delta_Sigma
                sigma_next_dev = Deviator(sigma_next)

                f_next = self.f + Delta_f
                epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                epsilon_T_eq_next = EquivalVonMisesStrain(f_next * epsilon_bar_T_next)

                J2_next = CalculateJ2(epsilon_bar_T_next)
                J3_next = CalculateJ3(epsilon_bar_T_next)

                epsilon_T_sat_next = self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * (J3_next / (J2_next ** (3 / 2)))) ** (1 / self.n)

                f_FA_next = self.f_FA + ((epsilon_T_sat_next - epsilon_T_eq_next) / epsilon_T_sat_next) * Delta_f if Delta_f > 0 else self.f_FA + (
                        self.f_FA / f_next) * Delta_f

                eta_next = 1 - (self.epsilon_trac_T_FA / self.epsilon_trac_T)

                epsilon_T_max_next = epsilon_T_sat_next * (1 - (f_FA_next / f_next) * eta_next)

                lambda2 = self.alpha2 * (f_next * epsilon_bar_T_next / (epsilon_T_max_next - epsilon_T_eq_next))
                self.lambda2 = lambda2
                F_epsilon_bar_T_next = sigma_next_dev - self.H_epsilon_T * epsilon_bar_T_next - lambda2

                F_epsilon_bar_T_next_eq = EquivalVonMisesStress(F_epsilon_bar_T_next)

                return f_next * Delta_epsilon_bar_T - (3 / 2) * Delta_lambda_epsilon_T * (
                            F_epsilon_bar_T_next / F_epsilon_bar_T_next_eq)

            def R_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T

                sigma_next = self.sigma + Delta_Sigma
                sigma_next_dev = Deviator(sigma_next)

                f_next = self.f + Delta_f
                epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                epsilon_T_eq_next = EquivalVonMisesStrain(f_next * epsilon_bar_T_next)
                J2_next = CalculateJ2(epsilon_bar_T_next)
                J3_next = CalculateJ3(epsilon_bar_T_next)

                epsilon_T_sat_next = self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * (J3_next / J2_next ** (3 / 2))) ** (1 / self.n)
                f_FA_next = self.f_FA + ((epsilon_T_sat_next - epsilon_T_eq_next) / epsilon_T_sat_next) * Delta_f if Delta_f > 0 else self.f_FA + (self.f_FA / f_next) * Delta_f

                eta_next = 1 - (self.epsilon_trac_T_FA / self.epsilon_trac_T)

                epsilon_T_max_next = epsilon_T_sat_next * (1 - (f_FA_next / f_next) * eta_next)
                lambda2 = self.alpha2 * (f_next * epsilon_bar_T_next / (epsilon_T_max_next - epsilon_T_eq_next))
                self.lambda2 = lambda2
                F_epsilon_bar_T_next = sigma_next_dev - self.H_epsilon_T * epsilon_bar_T_next - lambda2

                F_epsilon_bar_T_next_eq = EquivalVonMisesStress(F_epsilon_bar_T_next)

                return F_epsilon_bar_T_next_eq - self.F_epsilon

            if Tr == -1:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Br - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return -F_f_next - self.F_f_max + term1 + term2

            elif Tr == 0:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    return Delta_f

            else:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Bf - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return F_f_next - self.F_f_max - term1 + term2

        residuals["f1"] = R_f
        residuals["f2"] = R_epsilon_bar_T
        residuals["f3"] = R_delta_epsilon_bar_T
        residuals["f4"] = R_sigma


        init_delta_sigma = sigma_elast_pred - self.sigma
        init_delta_f = 1e-8
        init_delta_epsilon_bar_T = 1e-8 * np.eye(3)
        init_delta_lambda_epsilon_T = 1e-8
        if Tr >= 0:
            V0 = PackInputs(init_delta_sigma, init_delta_f, init_delta_epsilon_bar_T, init_delta_lambda_epsilon_T)
        else:
            V0 = PackInputs(init_delta_sigma, -init_delta_f, -init_delta_epsilon_bar_T, -init_delta_lambda_epsilon_T)

        return residuals, V0

    def UpdateState(self, delta_epsilon, delta_T):
        self.hist_T.append(self.T)
        self.T += delta_T

        self.hist_epsilon.append(np.copy(self.epsilon))
        self.epsilon += delta_epsilon

        sigma_elast_pred = self.sigma + np.einsum("ijkl, kl -> ij", self.C, (delta_epsilon + self.alpha * np.eye(3) * (delta_T - self.T0)))

        Tr, Or = self.ClassifyCase(sigma_elast_pred)
        print(f"Tr={Tr}, Or={Or}")

        res_dict, init_guess = self.OutputResiduals(Tr, Or, sigma_elast_pred)
        dsigma, df, depsilonbarT, dalpha3 = Solve(res_dict, init_guess)



        print("f: \n", self.f)
        print("Deltaf:\n", df)

        self.hist_sigma.append(np.copy(self.sigma))
        self.sigma += dsigma

        self.hist_f.append(self.f)
        self.f += df

        self.hist_epsilon_bar_T.append(np.copy(self.epsilon_bar_T))
        self.epsilon_bar_T += depsilonbarT

        self.hist_epsilon_bar_twin.append(np.copy(self.epsilon_bar_twin))
        H = CalculateComplianceTensor(self.H_twin, 0.5)
        self.epsilon_bar_twin = np.einsum("ijkl, kl -> ij", H, Deviator(self.sigma))

        self.hist_dzetaFA.append(self.dzetaFA)

        if df > 0:
            J2 = CalculateJ2(self.epsilon_bar_T)
            J3 = CalculateJ3(self.epsilon_bar_T)

            epsilon_T_sat = (
                    self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * (J3 / (J2 ** (3 / 2)))) ** (1 / self.n))

            self.dzetaFA = (epsilon_T_sat - EquivalVonMisesStrain(self.f * self.epsilon_bar_T)) / epsilon_T_sat
        else:
            self.dzetaFA = self.f_FA / self.f if self.f != 0 else 0

        self.hist_f_FA.append(self.f_FA)
        self.f_FA += self.dzetaFA * df

        F_f_new = (np.einsum("ij, ij", self.sigma, self.epsilon_bar_T)
                   + self.dzetaFA * np.einsum("ij, ij", self.sigma, self.epsilon_bar_twin)
                   - self.B * (self.T - self.T0)  # TODO: Define B
                   - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", self.epsilon_bar_T, self.epsilon_bar_T)
                   - self.H_f * self.f
                   - 0.5 * self.H_twin * self.dzetaFA * np.einsum("ij, ij", self.epsilon_bar_twin,
                                                                  self.epsilon_bar_twin))

        self.F_f = F_f_new
        if Or == 1:
            self.hist_lambda2.append(self.lambda2)
        else:
            self.hist_lambda2.append(np.zeros((3, 3)))
