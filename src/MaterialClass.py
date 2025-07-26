"""
:Submodule: MaterialClass
:Author: Y. Ben Zineb
:Email: yzineb3@gatech.edu

================================================= V1.0 -- Jul. 6th 2025 ================================================
This submodule defines the MaterialClass. This object is meant to be a container for all the material parameters.
This Class also tracks the history of strain, stress and other high priority parameters throughout the different
increments of a thermomechanical loading. The structure of this material description is inspired by the work of
[1] Chemisky et al. (2011) [https://doi.org/10.1016/j.mechmat.2011.04.003]
[2] A. Duval PhD Thesis (2009) [https://docnum.univ-lorraine.fr/public/SCD_T_2009_0132_DUVAL.pdf]
[3] Y. Chemisky PhD Thesis (2009) [https://docnum.univ-lorraine.fr/public/UPV-M/Theses/2009/Chemisky.Yves.SMZ0943.pdf]
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
        :param Tinit: float -- Initial Tenperature (K)
        """

        self.name = name

        self.n = n

        self.C = CalculateElasticityTensor(E, nu)
        self.S = CalculateComplianceTensor(E, nu)
        self.alpha = alpha
        self.T = Tinit

        #  Calculate Br and Bf given the chosen definition
        if br is not None :
            self.Br = br * epsilon_trac_T  # eq (4.5) in [3]
        else:
            self.Br = Br

        if bf is not None:
            self.Bf = bf * epsilon_trac_T  # eq (4.5) in [3]
        else:
            self.Bf = Bf

        self.T0 = ((self.Bf * Ms) + (self.Br * Af)) / (self.Bf + self.Br)  # eq (4.3) in [3]

        self.H_s = H_s
        self.H_f = H_f
        self.H_twin = H_twin
        self.H_epsilon_T = H_epsilon_T

        self.F_epsilon = F_epsilon

        self.epsilon_trac_T = epsilon_trac_T
        self.epsilon_trac_T_FA = epsilon_trac_T_FA
        self.epsilon_comp_T = epsilon_comp_T

        self.Ms = Ms
        self.Af = Af

        self.rf = rf

        self.f = 0
        self.f_FA = 0
        self.f_mem = 0.99
        self.fdot = 0
        self.f_obj = 1

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

        self.gamma = (epsilon_trac_T / epsilon_comp_T) ** n  # eq (2.53) in [1]
        self.K = epsilon_trac_T * (1 + ((1 - self.gamma) / (self.gamma + 1))) ** (-1 / n)  # eq (2.51) in [1]
        self.gammaf = 0.99

        self.F_f_max = self.Bf * (self.T0 - Ms)
        self.F_f_min = self.F_f_max * (1 - rf)
        self.F_f_mem = -self.F_f_max + self.F_f_min


        self.B = 2 * self.F_f_max / (Af - Ms)

        self.F_f = -self.B * (self.T - self.T0)


    def get_S(self):
        """
        Getter Function to extract the compliance tensor
        :return self.S: numpy array -- Compliance tensor of order 4. Shape = (3, 3, 3, 3)
        """
        return self.S


    def ClassifyCase(self, sigma_elast_pred):
        """
        This function uses a predicted Stress increment obtained by considering pure elasticity and assesses whether
        Transformation (Tr) and Orientation (Or) should be considered during a given increment.
        Tr = 1 denotes a direct transformation A --> M
        Tr = -1 denotes an indirect transformation M --> A
        Or = 1 denotes the presence of a preferred orientation
        :param sigma_elast_pred: numpy array -- Predicted Stress tensor at the considered increment supposing pure elasticity. Shape = (3, 3)
        :return Tr: float -- Flag indicating whether a transformation is happening and whether it is direct or indirect
        :return Or: float -- Flag indicating whether the transformation is oriented
        """

        # Default Case: No Transformation and No Orientation
        Tr = 0
        Or = 0

        # Necessary calculations
        sigma_elast_pred_D = Deviator(sigma_elast_pred)
        epsilonVM = EquivalVonMisesStrain(self.epsilon_bar_T)
        kappaf = (1 - self.gammaf) * self.F_f_mem
        Gammaf = (1 - self.gammaf) * self.F_f_min + self.gammaf * self.F_f_max

        # Predict the Transformation force using the predicted stress tensor (see eq (2.41) in [1])
        F_f_pred = (np.einsum("ij, ij", sigma_elast_pred, self.epsilon_bar_T)
                    + self.dzetaFA * np.einsum("ij, ij", sigma_elast_pred, self.epsilon_bar_twin)
                    - self.B * (self.T - self.T0)
                    - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", self.epsilon_bar_T, self.epsilon_bar_T)
                    - self.H_f * self.f
                    - 0.5 * self.H_twin * self.dzetaFA * np.einsum("ij, ij", self.epsilon_bar_twin,
                                                                   self.epsilon_bar_twin))

        # Predict the Orientation force using the predicted stress tensor (see eq (2.42) in [1])
        F_epsilon_bar_T_pred = (sigma_elast_pred_D
                                - self.H_epsilon_T * self.epsilon_bar_T)

        print(f"Prediction of F_f: {F_f_pred}", f"Previous F_f:{self.F_f}")
        print(f"Delta F_f:{F_f_pred - self.F_f}")

        # If the predicted Transformation Force increases from last increment we check for a direct Transformation
        if F_f_pred - self.F_f > 0:
            right_hand_term = Gammaf + (self.Bf - self.B) * (self.T - self.T0) - self.H_s * epsilonVM  # See Step 3 p.59 in [2]
            print("F_f - kappaf:", F_f_pred - kappaf)
            print("Right-hand term:", right_hand_term)
            if F_f_pred - kappaf > right_hand_term:
                Tr = 1  # Direct Transformation happening

        # If the predicted Transformation Force decreases from last increment we check for a direct Transformation
        elif F_f_pred - self.F_f < 0:
            right_hand_term = -Gammaf + (self.Br - self.B) * (self.T - self.T0) - self.H_s * epsilonVM  # See Step 3 p.59 in [2]
            print("F_f - kappaf:", F_f_pred - kappaf)
            print("Right-hand term:", right_hand_term)
            if F_f_pred - kappaf < right_hand_term:
                Tr = -1  # Indirect Transformation happening

        if EquivalVonMisesStress(F_epsilon_bar_T_pred) > self.F_epsilon:  # See Step 3 p.59 in [2]
            Or = 1  # Orientation has to be considered

        return Tr, Or

    def OutputResiduals(self, Tr, Or, sigma_elast_pred):
        """
        Given a Tr and Or flag and the associated elastic prediction of the stress tensor at a given increment, this
        function outputs the system of Residuals that needs to be solved in order to describe the evolution of the
        material throughout the given increment
        :param Tr: float -- Flag indicating whether Transformation is happening and whether it is direct or indirect
        :param Or: float -- Flag indicating whether the transformation is oriented
        :param sigma_elast_pred: numpy array -- Predicted Stress tensor at the considered increment supposing pure elasticity. Shape = (3, 3)
        :return residuals: dict -- Dictionary  containing the four residual functions that need to be minimized
        :return V0: list -- Initial guess for the resolution of the residuals system
        """

        #Initialize the residuals dictionary
        residuals = {}

        # No matter what Tr and Or are, the residual R_sigma is always the same (see eqs. (3.13) to (3.18) in [2])
        def R_sigma(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
            """
            Computes the residual R_sigma as a function of the problem's unknowns
            :param Delta_Sigma: numpy array -- Stress Tensor increase at the current increment. Shape = (3, 3)
            :param Delta_f: float -- Martensite volume fraction increase at the current increment
            :param Delta_epsilon_bar_T: numpy array -- Mean transformation strain increase at the current increment. Shape = (3, 3)
            :param Delta_lambda_epsilon_T: float -- Orientation multiplier increase at the current increment
            :return R_sigma: numpy array -- Elasticity residual. Shape = (3, 3)
            """
            # Necessary computations
            S = self.get_S()
            H = CalculateComplianceTensor(self.H_twin, 0.5)  # Stiffness tensor for twins (see p.59 in [2] for details)
            delta = np.eye(3)  # delta_ij matrix

            # Predict stress and strain at the end of the increment
            sigma_next = self.sigma + Delta_Sigma
            sigma_dev_next = Deviator(sigma_next)
            epsilon_bar_twin_next = np.einsum("ijkl, kl -> ij", H, sigma_dev_next)

            # Predict the martensite volume fraction at the end of the increment
            f_next = self.f + Delta_f

            # Predict the mean transformation strain and its Von Mises equivalent at the end of the increment
            epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
            epsilon_T_VM_next = EquivalVonMisesStrain(f_next * epsilon_bar_T_next)

            # Invariants calculation
            J2_next = CalculateJ2(epsilon_bar_T_next)
            J3_next = CalculateJ3(epsilon_bar_T_next)

            # See eq. (2.50) in [1]
            epsilon_T_sat_next = self.K * (1 + ((1 - self.gamma) / (self.gamma + 1)) * (J3_next / (J2_next ** (3 / 2)))) ** (1 / self.n)

            # See eq. (2.8) in [1]
            f_FA_next = self.f_FA + ((epsilon_T_sat_next - epsilon_T_VM_next) / epsilon_T_sat_next) * Delta_f if Delta_f > 0 else (self.f_FA + (
                        self.f_FA / f_next) * Delta_f)

            # Compute each factor of the residual
            term1 = np.einsum("ijkl, kl -> ij", S, sigma_next)  # Einstein summation S_ijkl sigma_kl
            term2 = self.alpha * delta * (self.T - self.T0)  # Dilation
            term3 = f_next * epsilon_bar_T_next  # Martensite strain
            term4 = f_FA_next * epsilon_bar_twin_next  # Twins strain

            return term1 + term2 + term3 + term4 - self.epsilon

        #  If the transformation is not oriented, we have to assume that the transformation is isotropic
        if Or == 0:
            # The streaming mean transformation strain residual is always the same when Or = 0 (see eqs (3.16) to (3.18) in [2])
            def R_delta_epsilon_bar_T(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                """
                Computes the residual R_delta_epsilon_bar_T as a function of the problem's unknowns
                :param Delta_Sigma: numpy array -- Stress Tensor increase at the current increment
                :param Delta_f: float -- Martensite volume fraction increase at the current increment
                :param Delta_epsilon_bar_T: numpy array -- Mean transformation strain increase at the current increment
                :param Delta_lambda_epsilon_T: float -- Orientation multiplier increase at the current increment
                :return R_delta_epsilon_bar_T: numpy array -- Streaming mean transformation strain residual. Shape = (3, 3)
                """
                epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T

                f_next = self.f + Delta_f

                epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)

                return f_next * Delta_epsilon_bar_T - (2 / 3) * Delta_lambda_epsilon_T * (epsilon_bar_T_next / epsilon_bar_T_eq_next)

            # If the transformation is indirect M --> A (Tr=-1)
            if Tr == -1:
                # Definition of the Transformation Residual for Or = 0; Tr = -1 (eq. (3.17) in [2])
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    gamma_f_next = np.abs(f_next - self.f_mem) / np.abs(0 - self.f_mem)

                    kappa_f = (1 - gamma_f_next) * self.F_f_mem
                    Gamma_f = (1 - gamma_f_next) * self.F_f_min + gamma_f_next * self.F_f_max

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Br - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return -F_f_next + kappa_f - Gamma_f + term1 - term2

                # Definition of the Orientation Residual for Or = 0; Tr = -1 (eq. (3.17) in [2])
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

                    gamma_f_next = np.abs(f_next - self.f_mem) / np.abs(1 - self.f_mem)

                    kappa_f = (1 - gamma_f_next) * self.F_f_mem
                    Gamma_f = (1 - gamma_f_next) * self.F_f_min + gamma_f_next * self.F_f_max

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Bf - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return F_f_next - kappa_f - Gamma_f - term1 + term2

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

                    gamma_f_next = np.abs(f_next - self.f_mem) / np.abs(0 - self.f_mem)

                    kappa_f = (1 - gamma_f_next) * self.F_f_mem
                    Gamma_f = (1 - gamma_f_next) * self.F_f_min + gamma_f_next * self.F_f_max

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Br - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return -F_f_next + kappa_f - Gamma_f + term1 - term2

            elif Tr == 0:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    return Delta_f

            else:
                def R_f(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
                    sigma_next = self.sigma + Delta_Sigma
                    epsilon_bar_T_next = self.epsilon_bar_T + Delta_epsilon_bar_T
                    epsilon_bar_T_eq_next = EquivalVonMisesStrain(epsilon_bar_T_next)
                    f_next = self.f + Delta_f

                    gamma_f_next = np.abs(f_next - self.f_mem) / np.abs(1 - self.f_mem)

                    kappa_f = (1 - gamma_f_next) * self.F_f_mem
                    Gamma_f = (1 - gamma_f_next) * self.F_f_min + gamma_f_next * self.F_f_max

                    F_f_next = (np.einsum("ij, ij", sigma_next, epsilon_bar_T_next)
                                - self.B * (self.T - self.T0)
                                - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", epsilon_bar_T_next, epsilon_bar_T_next)
                                - self.H_f * f_next
                                - self.alpha0 * ((f_next - 1) / f_next)
                                - self.alpha1 * (f_next / (1 - f_next)))

                    term1 = (self.Bf - self.B) * (self.T - self.T0)
                    term2 = self.H_s * epsilon_bar_T_eq_next

                    return F_f_next - kappa_f - Gamma_f - term1 + term2

        # Add the defined residual functions to the output dictionary
        residuals["f1"] = R_f
        residuals["f2"] = R_epsilon_bar_T
        residuals["f3"] = R_delta_epsilon_bar_T
        residuals["f4"] = R_sigma

        # Initial guess for system resolution
        init_delta_sigma = sigma_elast_pred - self.sigma  # Stress increment assuming pure elasticity
        init_delta_f = 1e-8  # Perturbation of the martensite volume fraction
        init_delta_epsilon_bar_T = 1e-8 * np.array([[1, 0, 0], [0, -0.5, 0], [0, 0, -0.5]])  # Perturbation of the mean transformation strain
        init_delta_lambda_epsilon_T = 1e-8  # Perturbation of the orientation multiplier

        # If the transformation is direct hint at positive increments by making the perturbations positive
        if Tr >= 0:
            V0 = PackInputs(init_delta_sigma, init_delta_f, init_delta_epsilon_bar_T, init_delta_lambda_epsilon_T)

        # If the transformation is indirect hint at negative increments by making the perturbations negative
        else:
            V0 = PackInputs(init_delta_sigma, -init_delta_f, -init_delta_epsilon_bar_T, -init_delta_lambda_epsilon_T)

        return residuals, V0

    def UpdateState(self, delta_epsilon, delta_T):
        self.hist_T.append(self.T)
        self.T += delta_T

        self.hist_epsilon.append(np.copy(self.epsilon))
        self.epsilon += delta_epsilon

        sigma_elast_pred = self.sigma + np.einsum("ijkl, kl -> ij", self.C, delta_epsilon)

        Tr, Or = self.ClassifyCase(sigma_elast_pred)
        print(f"Tr={Tr}, Or={Or}")

        if len(self.hist_Tr) > 0:
            if Tr != self.hist_Tr[-1]:
                if Tr >= 0 and self.hist_Tr[-1] == -1:
                    self.f_mem = self.f
                    self.F_f_mem = self.F_f - self.F_f_min
                    self.gammaf = np.abs(self.f - self.f_mem) / np.abs(1 - self.f_mem)
                elif Tr < 0 and self.hist_Tr[-1] == 1:
                    self.f_mem = self.f
                    self.F_f_mem = self.F_f_max - self.F_f
                    self.gammaf = np.abs(self.f - self.f_mem) / np.abs(0 - self.f_mem)
        self.hist_Tr.append(Tr)

        res_dict, init_guess = self.OutputResiduals(Tr, Or, sigma_elast_pred)
        dsigma, df, depsilonbarT, dalpha3 = Solve(res_dict, init_guess)


        print("f: \n", self.f)

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
                   - self.B * (self.T - self.T0)
                   - 0.5 * self.H_epsilon_T * np.einsum("ij, ij", self.epsilon_bar_T, self.epsilon_bar_T)
                   - self.H_f * self.f
                   - 0.5 * self.H_twin * self.dzetaFA * np.einsum("ij, ij", self.epsilon_bar_twin,
                                                                  self.epsilon_bar_twin))

        self.F_f = F_f_new