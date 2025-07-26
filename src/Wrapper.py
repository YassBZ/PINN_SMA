"""
:Submodule: Wrapper
:Author: Y. Ben Zineb
:Email: yzineb3@gatech.edu

================================================= V1.0 -- Jul. 6th 2025 ================================================
Submodule used for the residual minimization at each increment of a thermomechanical loading.
"""


import numpy as np
from scipy.optimize import root, least_squares, fmin, minimize, fsolve

def MakeSymmetric(v):
    return np.array([
        [v[0], v[1], v[2]],
        [v[1], v[3], v[4]],
        [v[2], v[4], v[5]],
    ])

def MakeSymmetricTraceless(v):
    a11, a12, a13, a22, a23 = v
    a33 = - (a11 + a22)
    return np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33],
    ])

def PackInputs(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
    vDelta_Sigma = [Delta_Sigma[0, 0], Delta_Sigma[0, 1], Delta_Sigma[0, 2], Delta_Sigma[1, 1], Delta_Sigma[1, 2], Delta_Sigma[2, 2]]
    vDelta_f = Delta_f
    vDelta_epsilon_bar_T = [Delta_epsilon_bar_T[0, 0], Delta_epsilon_bar_T[0, 1], Delta_epsilon_bar_T[0, 2], Delta_epsilon_bar_T[1, 1], Delta_epsilon_bar_T[1, 2]]
    vDelta_lambda_epsilon_T = Delta_lambda_epsilon_T
    return np.concatenate([vDelta_Sigma, [vDelta_f], vDelta_epsilon_bar_T, [vDelta_lambda_epsilon_T]])

def UnpackInputs(v):
    vDelta_Sigma = v[0:6]
    Delta_f = v[6]
    vDelta_epsilon_bar_T = v[7:12]
    Delta_lambda_epsilon_T = v[12]

    Delta_Sigma = MakeSymmetric(vDelta_Sigma)
    Delta_epsilon_bar_T = MakeSymmetricTraceless(vDelta_epsilon_bar_T)
    return Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T

def SystemResiduals(v, f_dict):
    Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T = UnpackInputs(v)
    res = []

    for key, item in f_dict.items():
        if key in ["f1", "f2"]:
            res.append(item(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T))
        else:
            res.extend(item(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T).ravel())
    return np.array(res)

def Solve(f_dict, v0):
    objective = lambda x: SystemResiduals(x, f_dict)

    sol = least_squares(objective, x0=v0, method="trf", ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=10000)

    Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T = UnpackInputs(sol.x)
    print("Solution found!")
    print("Delta_Sigma:\n", Delta_Sigma[0, 0])
    print("Delta_f =", Delta_f)
    print("Delta_epsilon_bar_T:\n", Delta_epsilon_bar_T)
    print("Delta_lambda_epsilon_T:\n", Delta_lambda_epsilon_T)

    return Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T