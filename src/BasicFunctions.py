"""
:Submodule: BasicFunctions
:Author: Y. Ben Zineb
:Email: yzineb3@gatech.edu

================================================= V1.0 -- Jul. 6th 2025 ================================================
Basic functions meant to be used in the different methods of the PINN-SMA module. The current implementation provides
functions that can perform the following operations:
- Extraction of a tensor's deviatoric part.
- Computation of the equivalent Von Mises Strain and Stress.
- Computation of the compliance tensor using Hooke's generalized law.
"""


import numpy as np

def Deviator(tensor):
    """
    Function that returns the deviatoric part of a tensor
    :param tensor: A tensor of order 2
    :return: The deviatoric part of tensor also of order 2
    """
    if len(tensor.shape) == 2:
        tr = np.trace(tensor)
        deviator = tensor - (1 / 3) * tr * np.eye(tensor.shape[0])

        return deviator
    else:
        print("Tensor must be of order 2")
        return 0

def EquivalVonMisesStress(tensor):
    """
    Function that returns the equivalent von Mises stress given a stress tensor
    :param tensor: numpy array of order 2
    :return: float von Mises equivalent Stress
    """
    J2 = ((1 / 6) * (
                 ((tensor[0, 0] - tensor[1, 1]) ** 2) +
                 ((tensor[1, 1] - tensor[2, 2]) ** 2) +
                 ((tensor[2, 2] - tensor[0, 0]) ** 2)) +
          (tensor[0, 1] ** 2) + (tensor[1, 2] ** 2) + (tensor[2, 0] ** 2))

    return np.sqrt(3 * J2)

def EquivalVonMisesStrain(tensor):
    """
    Function that returns the equivalent von Mises strain given a strain tensor
    :param tensor: numpy array of order 2, Strain tensor
    :return: float von Mises equivalent Strain
    """
    DevStrain = Deviator(tensor)
    Y2 = ((1 / 2) * (
        (DevStrain[0, 0] ** 2) + (DevStrain[1, 1] ** 2) + (DevStrain[2, 2] ** 2) +
        2 * (DevStrain[0, 1] ** 2) + 2 * (DevStrain[1, 2] ** 2) + 2 * (DevStrain[2, 0] ** 2)
    ))

    return np.sqrt((4 / 3) * Y2)


def CalculateElasticityTensor(E, nu):
    """
    Function that applies the generalized Hooke's Law to compute the elasticity tensor from the Young Modulus and
    the Poisson ratio.
    :param E: float, Young Modulus
    :param nu: float, Poisson ratio
    :return: numpy array of order 4, Compliance tensor C_{ijkl}
    """
    G = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    lam = K - (2 / 3) * G

    delta = np.eye(3)
    # Build outer products of Kronecker deltas using broadcasting
    deltaij_deltakl = delta[:, :, None, None] * delta[None, None, :, :]
    deltaik_deltajl = delta[:, None, :, None] * delta[None, :, None, :]
    deltail_deltajk = delta[:, None, None, :] * delta[None, :, :, None]

    C = lam * deltaij_deltakl + G * (deltaik_deltajl + deltail_deltajk)
    return C


def CalculateComplianceTensor(E, nu):
    """
    Function that computes the compliance tensor from the Young's modulus and the Poisson ratio,
    using the generalized Hooke's law for isotropic materials.
    :param E: float, Young's modulus
    :param nu: float, Poisson ratio
    :return: numpy array of order 4, Compliance tensor S_{ijkl}
    """
    delta = np.eye(3)

    # Build outer products of Kronecker deltas using broadcasting
    deltaij_deltakl = delta[:, :, None, None] * delta[None, None, :, :]
    deltaik_deltajl = delta[:, None, :, None] * delta[None, :, None, :]
    deltail_deltajk = delta[:, None, None, :] * delta[None, :, :, None]

    # Compliance tensor formula for isotropic materials
    S = ((1 + nu) / E) * 0.5 * (deltaik_deltajl + deltail_deltajk) - (nu / E) * deltaij_deltakl
    return S

def CalculateJ2(tensor):
    return 0.5 * np.einsum("ij, ij", tensor, tensor)

def CalculateJ3(tensor):
    return (1 / 3) * np.einsum("ij, jk, ki", tensor, tensor, tensor)