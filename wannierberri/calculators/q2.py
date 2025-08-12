import numpy as np
from ..utility import alpha_A, beta_A
from ..formula import Formula
from ..symmetry.point_symmetry import transform_ident, transform_odd
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, speed_of_light
from .. import factors as factors
from .calculator import MultitermCalculator
from .dynamic import DynamicCalculator
from itertools import permutations

def symmetrize_axes(arr, axes_to_symmetrize):
    """
    Symmetrize the array over the specified three axes by averaging over all permutations.
    
    Parameters:
    arr (numpy.ndarray): Input array to symmetrize.
    axes_to_symmetrize (list of int): List of three axes to symmetrize.
    
    Returns:
    numpy.ndarray: Symmetrized array.
    """
    # Validate input
    if len(axes_to_symmetrize) != 3:
        raise ValueError("Exactly three axes must be specified for symmetrization.")
    if len(set(axes_to_symmetrize)) != 3:
        raise ValueError("The axes to symmetrize must be distinct.")
    if any(axis >= arr.ndim or axis < -arr.ndim for axis in axes_to_symmetrize):
        raise ValueError("One or more specified axes are out of bounds for the array.")
    
    # Convert negative axes to positive for easier handling
    axes_to_symmetrize = [axis if axis >= 0 else arr.ndim + axis for axis in axes_to_symmetrize]
    
    # Generate all permutations of the specified axes
    all_permutations = list(permutations(axes_to_symmetrize))
    
    # Initialize the symmetrized array
    symmetrized = np.zeros_like(arr)
    
    # For each permutation, transpose the array and add to the sum
    for perm in all_permutations:
        # Construct the new axis order:
        # - All axes not in `axes_to_symmetrize` remain in their original positions.
        # - The axes in `axes_to_symmetrize` are permuted according to `perm`.
        new_order = []
        for axis in range(arr.ndim):
            if axis in axes_to_symmetrize:
                # This axis is being permuted; find its new position in `perm`
                new_order.append(perm[axes_to_symmetrize.index(axis)])
            else:
                # This axis is not being permuted; keep its original position
                new_order.append(axis)
        symmetrized += np.transpose(arr, new_order)
    
    # Average over all permutations
    symmetrized /= len(all_permutations)
    
    return symmetrized

def log(A, path):
    np.save(path + '.npy', A)
    # with open(path,'w') as f:
    #     f.write(np.array2string(A))
    return


class test_Formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)
        
        log(E, path='E')
        log( (data_K.grid.points_FFT) % 1, 'k')
        log(A, path='A')
        log(V, path='V')
        log(M, path='M')
        log(O, path='O')
        log(Q_P, path='Q_P')
        log(Q_M, path='Q_M')
        log(Q_M, path='O_P')

        self.Imn = A
        self.ndim = 1

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class test(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update()
        self.Formula = test_Formula
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.constant_factor =  factors.factor_cell_volume_to_m        

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return np.ones(self.omega.shape)


class Energy_Formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E_K = data_K.Q2.E_K

        self.Imn = E_K[:,:,None] + E_K[:,None,:]
        self.ndim = 0

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Energy(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update()
        self.Formula = Energy_Formula
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.constant_factor =  factors.factor_cell_volume_to_m        

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return np.ones(self.omega.shape)


###############################################################################
ev_to_J = 1.602176634e-19
A_to_m = 10e-10
hbar_eV_s = 6.582119569e-16
###############################################################################
arg_threshold = 1e-7

def denominator(mod_omega, E1, E2):
    arg = (E2 - E1 - mod_omega)
    select = abs(arg) < arg_threshold
    arg[select] = arg_threshold
    arg = 1. / arg
    arg[select] = 0.    
    return arg

def omega_part_0(omega, smr, E1, E2, sign):
    omega = omega * sign 
    mod_omega = omega + 1.j * smr

    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom

# def omega_part_0(omega, smr, E1, E2, sign):
#     omega = omega * sign
#     mod_omega = omega + 1.j * smr
#     denom = denominator(mod_omega, E1, E2)
#     omega_freq = 1/hbar * omega
#     return omega_freq * denom

def omega_part_1(omega, smr, E1, E2, sign):
    omega = omega * sign
    omega_freq = 1/hbar * omega
    return omega_freq

def omega_part_2(omega, smr, E1, E2, sign):
    omega = omega * sign
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq**2 * denom**2

def omega_part_3(omega, smr, E1, E2, sign):
    omega = omega 
    mod_omega = omega + 1.j * smr * sign
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom

def omega_part_3_test(omega, smr, E1, E2, sign_omega, sign_smr):
    omega = omega * sign_omega
    mod_omega = omega + 1.j * smr * sign_smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom

def omega_part_4(omega, smr, E1, E2, sign):
    omega = omega * sign
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom**2

def omega_part_5(omega, smr, E1, E2, sign):
    return np.ones(omega.shape)

def omega_part_6(omega, smr, E1, E2, sign):
    omega = omega * sign
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom**2

def omega_part_7(omega, smr, E1, E2, sign):
    omega = omega * sign
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom**3

def omega_part_8(omega, smr, E1, E2, sign):
    omega = omega * sign
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom**3


def load(data_K, external_terms, spin):
    E = data_K.Q2.E_K
    dE = data_K.Q2.dE
    ddE = data_K.Q2.ddE
    invEdif = data_K.Q2.invEdif
    Delta = data_K.Q2.Edif
    lev = data_K.Q2.levicivita
    anti_kron = data_K.Q2.anti_kron
    if external_terms:
        A = data_K.Q2.A_H
        dA = data_K.Q2.gender_A_H
        ddA = data_K.Q2.gender2_A_H
        O = data_K.Q2.berry_curvature
        M = data_K.Q2.magnetic_dipole
        V = data_K.Q2.velocity
        ddV = data_K.Q2.gender2_velocity
        Q_P = data_K.Q2.electric_quadrupole
        Q_M = data_K.Q2.magnetic_quadrupole
        O_P = data_K.Q2.electric_octupole
    else:
        A = data_K.Q2.A_H_internal
        dA = data_K.Q2.gender_A_H_internal
        ddA = data_K.Q2.gender2_A_H_internal
        O = data_K.Q2.berry_curvature_internal
        M = data_K.Q2.magnetic_dipole_internal
        V = data_K.Q2.velocity_internal
        ddV = data_K.Q2.gender2_velocity_internal
        Q_P = data_K.Q2.electric_quadrupole_internal
        Q_M = data_K.Q2.magnetic_quadrupole_internal
        O_P = data_K.Q2.electric_octupole_internal
    if spin:
        S = data_K.Q2.S
    else:
        S = np.zeros(V.shape)
    return E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron


########################################################
# MAGNETIC SUSCEPTIBILITY
########################################################


class Mag_sus_TR_even(MultitermCalculator):

    def __init__(self, spin=False, **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(spin=spin)
        # Fermi sea terms
        self.terms.extend([Mag_sus_even_1(**params_terms, **kwargs),
                           Mag_sus_even_2(**params_terms, **kwargs),
                           Mag_sus_even_3(**params_terms, **kwargs)])


class Mag_sus_TR_odd(MultitermCalculator):

    def __init__(self, spin=False, **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(spin=spin)
        # Fermi sea terms
        self.terms.extend([Mag_sus_odd_1(**params_terms, **kwargs),
                           Mag_sus_odd_3(**params_terms, **kwargs)])


##########################################################################################


class Mag_sus_occ_2_spin_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        O = data_K.Q2.berry_curvature
        M = data_K.Q2.magnetic_dipole_spin
        kron = data_K.Q2.kron

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += 1 * hbar * np.einsum('knmi, knml, knm -> knmil', O, M, kron)
        summ += 1 * hbar * np.einsum('knmi, knml, knm -> knmil', M, O, kron)
        summ *= -elementary_charge / ( hbar * speed_of_light)

        self.Imn = np.real(summ)
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        inn3 = np.concatenate((inn1, inn2))
        return self.Imn[ik, inn3].sum(axis=0)[inn3].sum(axis=0)


class Mag_sus_occ_2_spin(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_occ_2_spin_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.sign = 1

    def factor_omega(self, E1, E2):
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


class Mag_sus_occ_2_orb_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        O = data_K.Q2.berry_curvature
        M = data_K.Q2.magnetic_dipole_orb
        kron = data_K.Q2.kron

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += np.einsum('knmi, knml, knm -> knmil', O, M, kron)
        summ += np.einsum('knmi, knml, knm -> knmil', M, O, kron)
        summ *= -elementary_charge / ( hbar * speed_of_light)

        self.Imn = np.real(summ)
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        inn3 = np.concatenate((inn1, inn2))
        return self.Imn[ik, inn3].sum(axis=0)[inn3].sum(axis=0)


class Mag_sus_occ_2_orb(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_occ_2_orb_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.sign = 1

    def factor_omega(self, E1, E2):
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


class Mag_sus_inter_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        O = data_K.Q2.berry_curvature
        M_orb = data_K.Q2.magnetic_dipole_orb
        M_spin = data_K.Q2.magnetic_dipole_spin
        M = M_orb + M_spin
        antikron = data_K.Q2.anti_kron
        invEdif = data_K.Q2.invEdif

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += np.einsum('knmi, knml, knm, knm -> knmil', M, M.conj(), antikron, invEdif)
        summ *= -elementary_charge / ( hbar * speed_of_light)

        self.Imn = np.real(summ)
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        inn3 = np.concatenate((inn1, inn2))
        return self.Imn[ik, inn3].sum(axis=0)[inn3].sum(axis=0)


class Mag_sus_inter(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_inter_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.sign = 1

    def factor_omega(self, E1, E2):
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


class Mag_sus_occ_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        if self.external_terms:
            A = data_K.Q2.A_H
        else:
            A = data_K.Q2.A_H_internal
        lev = data_K.Q2.levicivita
        ddE = data_K.Q2.ddE

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += np.einsum('iab, lcd, knma, kmnc, kndb -> knmil', 
                           lev, lev, A, A, ddE)
        summ += - hbar**2/electron_mass *  np.einsum('iab, lcd, knma, kmnc, db -> knmil', 
                                                   lev, lev, A, A, np.eye(3))
        summ *= elementary_charge**2 / ( 4 * hbar**2 * speed_of_light**2 )

        self.Imn = np.real(summ)
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        inn3 = np.concatenate((inn1, inn2))
        return self.Imn[ik, inn3].sum(axis=0)[inn3].sum(axis=0)


class Mag_sus_occ(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_occ_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.sign = 1


    def factor_omega(self, E1, E2):
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


##########################################################################################

class Mag_sus_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        A = data_K.Q2.A_H_internal
        ddV = data_K.Q2.gender2_velocity_internal
        lev = data_K.Q2.levicivita
        M = data_K.Q2.magnetic_dipole_internal
        invEdif = data_K.Q2.invEdif

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)

        # summ_even += -1/16 * elementary_charge**2 * speed_of_light**(-2) * 1j * np.einsum('qnmdca, qmnb, kcd, lab -> qnmkl', ddV, A, lev, lev)
        summ += 1 * hbar * np.einsum('qmn, qnmk, qmnl -> qnmkl', invEdif, M, M)

        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        print(self.sign, type(self).__name__)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Mag_sus_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        A = data_K.Q2.A_H_internal
        ddV = data_K.Q2.gender2_velocity_internal
        lev = data_K.Q2.levicivita
        M = data_K.Q2.magnetic_dipole_internal
        invEdif = data_K.Q2.invEdif

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)

        # summ_even += -1/16 * elementary_charge**2 * speed_of_light**(-2) * 1j * np.einsum('qnmdca, qmnb, kcd, lab -> qnmkl', ddV, A, lev, lev)
        summ += 1 * hbar * np.einsum('qmn, qmnk, qnml -> qnmkl', invEdif, M, M)

        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1

        print(self.sign, type(self).__name__)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Mag_sus_even_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        O = data_K.Q2.berry_curvature_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += -1/16 * elementary_charge**2 * 1/hbar * speed_of_light**(-2) * np.einsum('qmnk, qnml -> qnmkl', O, O)
        summ += 1/16 * elementary_charge**2 * 1/hbar * speed_of_light**(-2) * np.einsum('qnmk, qmnl -> qnmkl', O, O)
        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_even_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_even_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_1(self.omega, self.smr_fixed_width ,E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
        if E1 - E2 < 1e-3:
            return self.FermiDirac(1000)
        else:
	        return self.FermiDirac(E1)


class Mag_sus_even_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        invEdif = data_K.Q2.invEdif
        dE = data_K.Q2.dE
        M_even = data_K.Q2.magnetic_dipole_internal_TR_even
        M = data_K.Q2.magnetic_dipole_internal
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ_even = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ_odd = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)

        summ_even += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qma, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_even, A, lev)
        summ_even += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qna, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_even, A, lev)

        summ_odd += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qma, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_odd, A, lev)
        summ_odd += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qna, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_odd, A, lev)

        summ = summ_even + summ_odd
        summ_tr = -(summ_even - summ_odd).swapaxes(1,2)

        self.Imn = summ + summ_tr
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_even_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_even_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_2(self.omega, self.smr_fixed_width ,E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Mag_sus_odd_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        invEdif = data_K.Q2.invEdif
        dE = data_K.Q2.dE
        M = data_K.Q2.magnetic_dipole_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ_even = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ_odd = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)

        summ_even += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qma, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_even, A, lev)
        summ_even += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qna, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_even, A, lev)

        summ_odd += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qma, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_odd, A, lev)
        summ_odd += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qna, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M_odd, A, lev)

        summ = summ_even + summ_odd
        summ_tr = -(summ_even - summ_odd).swapaxes(1,2)

        self.Imn = summ + summ_tr
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_odd_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_odd_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_2(self.omega, self.smr_fixed_width ,E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)

#########################################
# Gamma (LOWER CASE)
#########################################

class Gamma_TR_even(MultitermCalculator):

    def __init__(self, spin=False, **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(spin=spin)
        # Fermi sea terms
        self.terms.extend([Gamma_even_1(**params_terms, **kwargs),
                           Gamma_even_2(**params_terms, **kwargs),
                           Gamma_even_3(**params_terms, **kwargs)])


class Gamma_TR_odd(MultitermCalculator):

    def __init__(self, spin=False, **kwargs):
        super().__init__(**kwargs)
        params_terms = dict(spin=spin)
        # Fermi sea terms
        self.terms.extend([Gamma_odd_1(**params_terms, **kwargs),
                           Gamma_odd_2(**params_terms, **kwargs)])

#################

class Gamma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        ddV = data_K.Q2.gender2_velocity_internal
        A = data_K.Q2.A_H_internal
        lev = data_K.Q2.levicivita
        M = data_K.Q2.magnetic_dipole_internal
        Q_P = data_K.Q2.electric_quadrupole_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        
        summ += 1 * np.einsum('knmi, kmnjl -> knmijl', M, Q_P)
        
        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Gamma_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        ddV = data_K.Q2.gender2_velocity_internal
        A = data_K.Q2.A_H_internal
        lev = data_K.Q2.levicivita
        Q_P = data_K.Q2.electric_quadrupole_internal
        M = data_K.Q2.magnetic_dipole_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        
        # summ_even += -1/16 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbaj, kmnl, iab -> knmijl', ddV, A, lev)
        # summ_even += -1/16 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbal, kmnj, iab -> knmijl', ddV, A, lev)
        
        summ -= 1 * np.einsum('kmni, knmjl -> knmijl', M, Q_P)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Gamma_even_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        dE = data_K.Q2.dE
        M_even = data_K.Q2.magnetic_dipole_internal_TR_even
        M = data_K.Q2.magnetic_dipole_internal
        summ_even = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ_odd = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('kmj, knmi, kmnl -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('knj, knmi, kmnl -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('kml, knmi, kmnj -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('knl, knmi, kmnj -> knmijl', dE, M_even, A)

        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('kmj, knmi, kmnl -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('knj, knmi, kmnl -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('kml, knmi, kmnj -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('knl, knmi, kmnj -> knmijl', dE, M_odd, A)

        summ = summ_even + summ_odd
        summ_tr = (summ_even - summ_odd).swapaxes(1,2)

        self.Imn = summ + summ_tr
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_even_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_even_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width ,E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Gamma_odd_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        dE = data_K.Q2.dE
        M_even = data_K.Q2.magnetic_dipole_internal_TR_even
        M = data_K.Q2.magnetic_dipole_internal
        summ_even = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ_odd = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('kmj, knmi, kmnl -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('knj, knmi, kmnl -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('kml, knmi, kmnj -> knmijl', dE, M_even, A)
        summ_even += 1/4 * elementary_charge * 1j * np.einsum('knl, knmi, kmnj -> knmijl', dE, M_even, A)

        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('kmj, knmi, kmnl -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('knj, knmi, kmnl -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('kml, knmi, kmnj -> knmijl', dE, M_odd, A)
        summ_odd += 1/4 * elementary_charge * 1j * np.einsum('knl, knmi, kmnj -> knmijl', dE, M_odd, A)


        summ = summ_even + summ_odd
        summ_tr = (summ_even - summ_odd).swapaxes(1,2)

        self.Imn = summ - summ_tr
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_odd_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_odd_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width ,E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Gamma_even_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        O = data_K.Q2.berry_curvature_internal
        dA = data_K.Q2.gender_A_H_internal
        Q_P = data_K.Q2.electric_quadrupole_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knmi, kmnjl -> knmijl', O, dA)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knmi, kmnlj -> knmijl', O, dA)
        summ += -1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmnjl, knmi -> knmijl', Q_P, O)
        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_even_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_even_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
        if E1 - E2 < 1e-3:
            return self.FermiDirac(1000)
        else:
	        return self.FermiDirac(E1)

#####################################################
# BETA
#####################################################

class Beta_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        # E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)
        Q_M = data_K.Q2.magnetic_quadrupole_internal
        A = data_K.Q2.A_H_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        # summ += -1/8 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbaj, kmnl, iab -> knmijl', ddV, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('km, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        # summ += 1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kn, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, kpm, kpn, knpb, kpmj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, kpm, kpn, kpmb, knpj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, kpm, kpn, knpb, kpmj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, kpm, kpn, kpmb, knpj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        summ += 1 * elementary_charge * np.einsum('knmij, kmnl -> knmijl', Q_M, A)
        
        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Beta_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Beta_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)



class Beta_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        # E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)
        Q_M = data_K.Q2.magnetic_quadrupole_internal
        A = data_K.Q2.A_H_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        # summ += -1/8 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbaj, kmnl, iab -> knmijl', ddV, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('km, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        # summ += 1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kn, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, kpm, kpn, knpb, kpmj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, kpm, kpn, kpmb, knpj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, kpm, kpn, knpb, kpmj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        # summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, kpm, kpn, kpmb, knpj, kmnl, iab -> knmijl', dE, anti_kron, anti_kron, A, A, A, lev)
        summ += 1 * elementary_charge * np.einsum('kmnij, knml -> knmijl', Q_M, A)
        
        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Beta_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Beta_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Beta_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/3 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmni, knmlj -> knmijl', O, dA)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmla, kpm, kpn, kpnb, kmpj, iab -> knmijl', dA, anti_kron, anti_kron, A, A, lev)

        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Beta_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Beta_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
        if E1 - E2 < 1e-3:
            return self.FermiDirac(1000)
        else:
	        return self.FermiDirac(E1)


#####################################################
# DELTA
#####################################################


class Delta_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knmiaj, kmnb, lab -> knmijl', ddA, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)
        

class Delta_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmaj, kmnb, knmi, lab -> knmijl', ddE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knaj, kmnb, knmi, lab -> knmijl', ddE, A, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_6(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_7(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('kmnlj, knmi -> knmijl', Q_M, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_4(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_TR_4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('knmlj, kmni -> knmijl', Q_M, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_TR_4(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_TR_4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_5_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knp, kpm, knmi, kmpj, kpnb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knp, kpm, knmi, kpnj, kmpb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knp, kpm, knmi, kmpj, kpnb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knp, kpm, knmi, kpnj, kmpb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmj, knp, kpm, kmpa, knmi, kpnb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmj, knp, kpm, kpna, knmi, kmpb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knj, knp, kpm, kmpa, knmi, kpnb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knj, knp, kpm, kpna, knmi, kmpb, lab -> knmijl', dE, anti_kron, anti_kron, A, A, V, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kma, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kma, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kna, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kna, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, kmj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, kmj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/2 * 1/electron_mass * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmj, kmnl, knmi -> knmijl', dE, S, A)
        summ += 1/2 * 1/electron_mass * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knj, kmnl, knmi -> knmijl', dE, S, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_5(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_5_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_6_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmnl, knmij -> knmijl', O, dA)
        summ += 1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmia, knp, kpm, kmpb, kpnj, lab -> knmijl', dA, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmia, knp, kpm, kpnb, kmpj, lab -> knmijl', dA, anti_kron, anti_kron, A, A, lev)
        
        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_6(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_6_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
        if E1 - E2 < 1e-3:
            return self.FermiDirac(1000)
        else:
	        return self.FermiDirac(E1) 

##############################################################
# PI
##############################################################


class Pi_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1/6 * elementary_charge * np.einsum('qmnjlk, qnmi -> qnmijlk', O_P, A)

        # jlk = summ
        # jkl = jlk.swapaxes(-1,-2)
        # ljk = summ.swapaxes(-2,-3)
        # lkj = ljk.swapaxes(-1,-2)
        # klj = summ.swapaxes(-1,-3)
        # kjl = klj.swapaxes(-1,-2)
        # summ = jlk + jkl + ljk + lkj + klj + kjl

        self.Imn = summ#pi - pi.swapaxes(-1,-3)

        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Pi_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Pi_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1/6 * elementary_charge * np.einsum('qnmjlk, qmni -> qnmijlk', O_P, A)

        # jlk = summ
        # jkl = jlk.swapaxes(-1,-2)
        # ljk = summ.swapaxes(-2,-3)
        # lkj = ljk.swapaxes(-1,-2)
        # klj = summ.swapaxes(-1,-3)
        # kjl = klj.swapaxes(-1,-2)
        # summ = jlk + jkl + ljk + lkj + klj + kjl

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Pi_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Pi_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += -1/12 * elementary_charge**2 * np.einsum('qmj, qml, qnmi, qmnk -> qnmijlk', dE, dE, A, A)
        summ += -1/12 * elementary_charge**2 * np.einsum('qnj, qnl, qnmi, qmnk -> qnmijlk', dE, dE, A, A)

        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1,-3)
        kjl = klj.swapaxes(-1,-2)
        summ = jlk + jkl + ljk + lkj + klj + kjl

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Pi_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_8(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Pi_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1/24 * elementary_charge**2 * np.einsum('qmj, qmnkl, qnmi -> qnmijlk', dE, dA, A)
        summ += -1/24 * elementary_charge**2 * np.einsum('qnj, qmnkl, qnmi -> qnmijlk', dE, dA, A)
        summ += 1/24 * elementary_charge**2 * np.einsum('qmlj, qnmi, qmnk -> qnmijlk', ddE, A, A)
        summ += -1/24 * elementary_charge**2 * np.einsum('qnlj, qnmi, qmnk -> qnmijlk', ddE, A, A)
        summ += 1/12 * elementary_charge * 1j * np.einsum('qmj, qmnlk, qnmi -> qnmijlk', dE, Q_P, A)
        summ += 1/12 * elementary_charge * 1j * np.einsum('qnj, qmnlk, qnmi -> qnmijlk', dE, Q_P, A)

        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1,-3)
        kjl = klj.swapaxes(-1,-2)
        summ = jlk + jkl + ljk + lkj + klj + kjl

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Pi_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


##############################################################
# SIGMA
##############################################################


class Sigma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmijk, qmnl -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmijl, qmnk -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmjik, qmnl -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmjil, qmnk -> qnmijlk', ddA, A)
        summ += 1 * np.einsum('qnmij, qmnkl -> qnmijlk', Q_P, Q_P)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Sigma_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Sigma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Sigma_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmijk, qmnl -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmijl, qmnk -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmjik, qmnl -> qnmijlk', ddA, A)
        # summ += -1/16 * elementary_charge**2 * np.einsum('qnmjil, qmnk -> qnmijlk', ddA, A)
        summ += 1 * np.einsum('qmnij, qnmkl -> qnmijlk', Q_P, Q_P)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Sigma_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Sigma_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Sigma_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1/4 * elementary_charge * 1j * np.einsum('qmk, qnmij, qmnl -> qnmijlk', dE, Q_P, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('qnk, qnmij, qmnl -> qnmijlk', dE, Q_P, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('qml, qnmij, qmnk -> qnmijlk', dE, Q_P, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('qnl, qnmij, qmnk -> qnmijlk', dE, Q_P, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Sigma_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Sigma_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


############################################################
# Upper Case Gamma
############################################################


class Capital_Gamma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/4 * elementary_charge * speed_of_light * np.einsum('kma, knmij, kmnb, lab -> knmijl', dE, Q_P, A, lev)
        summ += -1/4 * elementary_charge * speed_of_light * np.einsum('kna, knmij, kmnb, lab -> knmijl', dE, Q_P, A, lev)


        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_6(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/16 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knmija, kmnb, lab -> knmijl', ddA, A, lev)
        summ += -1/16 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knmjia, kmnb, lab -> knmijl', ddA, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * np.einsum('kmnl, knmij -> knmijl', M, Q_P)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_TR_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * np.einsum('knml, kmnij -> knmijl', M, Q_P)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_TR_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_TR_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knml, kmnij -> knmijl', O, dA)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knml, kmnji -> knmijl', O, dA)
        summ += -1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knmij, knml -> knmijl', Q_P, O)

        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_4(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
        if E1 - E2 < 1e-3:
            return self.FermiDirac(1000)
        else:
	        return self.FermiDirac(E1)


##############################################################
# OMEGA
##############################################################


class Omega_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('qnmijl, qmnk -> qnmijlk', O_P, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Omega_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Omega_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Omega_TR_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S, anti_kron = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('qmnijl, qnmk -> qnmijlk', O_P, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Omega_TR_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Omega_TR_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class A4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        A = data_K.A_H_internal
        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('qnmi, qnmj, qnml, qnmk -> qnmijlk', A, A, A, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class A4(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = A4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2, self.sign)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class A4_TR_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        A = data_K.A_H_internal

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('qmni, qmnj, qmnl, qmnk -> qnmijlk', A, A, A, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class A4_TR(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        # if 'TR' in type(self).__name__:
        #     self.sign = -1
        # else:
        self.sign_omega = 1
        self.sign_smr = 1
        
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = A4_TR_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3_test(self.omega, self.smr_fixed_width, E1, E2, self.sign_omega, self.sign_smr)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)