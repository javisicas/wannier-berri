import numpy as np
from ..utility import alpha_A, beta_A
from ..formula import Formula
from ..symmetry.point_symmetry import transform_ident, transform_ident
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, speed_of_light
from .. import factors as factors
from .calculator import MultitermCalculator
from .dynamic import DynamicCalculator


def log(A, path):
    with open('path','w') as f:
        f.write(np.array2string(A))


class test_Formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        dcov_A = data_K.dcov_A
        
        self.Imn = dcov_A
        self.ndim = 2

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

def omega_part_0(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom

def omega_part_1(omega, smr, E1, E2):
    omega_freq = 1/hbar * omega
    return omega_freq

def omega_part_2(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq**2 * denom**2

def omega_part_3(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom

def omega_part_4(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return denom**2

def omega_part_5(omega, smr, E1, E2):
    return np.ones(omega.shape)

def omega_part_6(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom**2

def omega_part_7(omega, smr, E1, E2):
    mod_omega = omega + 1.j * smr
    denom = denominator(mod_omega, E1, E2)
    omega_freq = 1/hbar * omega
    return omega_freq * denom**3

def omega_part_8(omega, smr, E1, E2):
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
    return E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S


########################################################
# MAGNETIC SUSCEPTIBILITY
########################################################

class Mag_sus_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += -1 * elementary_charge**2 * speed_of_light**(-2) * 1j * np.einsum('qnmdca, qmnb, kcd, lab -> qnmkl', ddV, A, lev, lev)
        summ += 1 * hbar * np.einsum('qmn, qnmk, qmnl -> qnmkl', invEdif, M, M)

        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width ,E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Mag_sus_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += -1/16 * elementary_charge**2 * 1/hbar * speed_of_light**(-2) * np.einsum('qmnk, qnml -> qnmkl', O, O)
        summ += 1/16 * elementary_charge**2 * 1/hbar * speed_of_light**(-2) * np.einsum('qnmk, qmnl -> qnmkl', O, O)
        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_1(self.omega, self.smr_fixed_width ,E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


class Mag_sus_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3), dtype=complex)
        summ += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qma, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M, A, lev)
        summ += -1/4 * elementary_charge * hbar * 1/speed_of_light * np.einsum('qmn, qna, qnmk, qmnb, lab -> qnmkl', invEdif, dE, M, A, lev)
        self.Imn = summ
        self.ndim = 2

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Mag_sus_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Mag_sus_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_2(self.omega, self.smr_fixed_width ,E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


#########################################
# Gamma (LOWER CASE)
#########################################

class Gamma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/16 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbaj, kmnl, iab -> knmijl', ddV, A, lev)
        summ += -1/16 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbal, kmnj, iab -> knmijl', ddV, A, lev)
        summ += 1 * np.einsum('knmi, kmnjl -> knmijl', M, Q_P)
        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width ,E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Gamma_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/4 * elementary_charge * 1j * np.einsum('kmj, knmi, kmnl -> knmijl', dE, M, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('knj, knmi, kmnl -> knmijl', dE, M, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('kml, knmi, kmnj -> knmijl', dE, M, A)
        summ += 1/4 * elementary_charge * 1j * np.einsum('knl, knmi, kmnj -> knmijl', dE, M, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width ,E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)

    
class Gamma_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knmi, kmnjl -> knmijl', O, dA)
        summ += -1/8 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knmi, kmnlj -> knmijl', O, dA)
        summ += -1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmnjl, knmi -> knmijl', Q_P, O)
        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Gamma_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Gamma_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)

#####################################################
# BETA
#####################################################

class Beta_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/8 * elementary_charge**2 * 1/speed_of_light * np.einsum('knmbaj, kmnl, iab -> knmijl', ddV, A, lev)
        summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('km, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        summ += 1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kn, knmbaj, kmnl, iab -> knmijl', E, ddA, A, lev)
        summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, knpb, kpmj, kmnl, iab -> knmijl', dE, A, A, A, lev)
        summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kma, kpmb, knpj, kmnl, iab -> knmijl', dE, A, A, A, lev)
        summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, knpb, kpmj, kmnl, iab -> knmijl', dE, A, A, A, lev)
        summ += -1/24 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('kna, kpmb, knpj, kmnl, iab -> knmijl', dE, A, A, A, lev)
        summ += 1 * elementary_charge * np.einsum('knmij, kmnl -> knmijl', Q_M, A)
                
        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Beta_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Beta_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Beta_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/3 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmni, knmlj -> knmijl', O, dA)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmla, kpnb, kmpj, iab -> knmijl', dA, A, A, lev)
        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Beta_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Beta_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


#####################################################
# DELTA
#####################################################


class Delta_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmaj, kmnb, knmi, lab -> knmijl', ddE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knaj, kmnb, knmi, lab -> knmijl', ddE, A, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_6(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/6 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_2(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_7(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * np.einsum('kmnlj, knmi -> knmijl', Q_M, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knmi, kmpj, kpna, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kma, knmi, kpnj, kmpb, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knmi, kmpj, kpna, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kna, knmi, kpnj, kmpb, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmj, kmpa, knmi, kpna, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('kmj, kpna, knmi, kmpb, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knj, kmpa, knmi, kpna, lab -> knmijl', dE, A, A, V, lev)
        summ += 1/12 * elementary_charge**2 * 1/speed_of_light * 1j * np.einsum('knj, kpna, knmi, kmpb, lab -> knmijl', dE, A, A, V, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kma, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kma, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kna, kmj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += -1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kna, knj, kmnb, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, kmj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, kmj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knj, kmna, knmi, lab -> knmijl', dE, dE, A, A, lev)
        summ += 1 * np.einsum('kmj, kmnl, knmi -> knmijl', dE, S, A)
        summ += 1 * np.einsum('knj, kmnl, knmi -> knmijl', dE, S, A)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_4(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Delta_5_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1/6 * elementary_charge**2 * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmnl, knmia, knmij, lab -> knmijl', O, dA, dA, lev)
        summ += 1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmia, kmpb, kpnj, lab -> knmijl', dA, A, A, lev)
        summ += 1/12 * elementary_charge**2 * 1/hbar * 1/speed_of_light * np.einsum('knmia, kpnb, kmpj, lab -> knmijl', dA, A, A, lev)

        self.Imn = np.real(summ)
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Delta_5(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Delta_5_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


##############################################################
# PI
##############################################################


class Pi_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1/6 * elementary_charge * np.einsum('qmnjlk, qnmi -> qnmijlk', O_P, A)

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


class Pi_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Pi_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        # manual: added k index to dE, i guess it is fine after symmetrization
        summ += -1/12 * elementary_charge**2 * np.einsum('qmjk, qmlk, qnmi -> qnmijlk', dE[:,:,:,None], dE[:,:,:,None], A)
        summ += -1/12 * elementary_charge**2 * np.einsum('qnjk, qnlk, qnmi -> qnmijlk', dE[:,:,:,None], dE[:,:,:,None], A)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_8(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Pi_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Pi_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


##############################################################
# SIGMA
##############################################################


class Sigma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += -1/16 * elementary_charge**2 * np.einsum('qnmijk, qmnl -> qnmijlk', ddA, A)
        summ += -1/16 * elementary_charge**2 * np.einsum('qnmijl, qmnk -> qnmijlk', ddA, A)
        summ += -1/16 * elementary_charge**2 * np.einsum('qnmjik, qmnl -> qnmijlk', ddA, A)
        summ += -1/16 * elementary_charge**2 * np.einsum('qnmjil, qmnk -> qnmijlk', ddA, A)
        summ += 1 * np.einsum('qnmij, qmnkl -> qnmijlk', Q_P, Q_P)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Sigma_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Sigma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Sigma_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Sigma_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_4(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


############################################################
# Upper Case Gamma
############################################################


class Capital_Gamma_1_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_1_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_6(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_2_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_2_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_0(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_3_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += 1 * np.einsum('kmnl, knmij -> knmijl', M, Q_P)

        self.Imn = summ
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Capital_Gamma_3(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_3_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)


class Capital_Gamma_4_formula(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

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
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Capital_Gamma_4_formula
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_5(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1)


#############################################################3
# OMEGA
##############################################################


class Omega_formula_1(Formula):
    def __init__(self, data_K, spin=True, **parameters):
        super().__init__(data_K, **parameters)
        E, dE, ddE, invEdif, Delta, lev, A, dA, ddA, O, M, V, ddV, Q_P, Q_M, O_P, S = load(data_K, self.external_terms, spin)

        summ = np.zeros((data_K.nk, data_K.num_wann, data_K.num_wann, 3, 3, 3, 3), dtype=complex)
        summ += 1 * elementary_charge * np.einsum('qnmijl, qmnk -> qnmijlk', O_P, A)

        self.Imn = summ
        self.ndim = 4

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Omega_1(DynamicCalculator):
    def __init__(self, spin=True, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(spin=spin))
        self.Formula = Omega_formula_1
        self.constant_factor =  factors.factor_cell_volume_to_m
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def factor_omega(self, E1, E2): #E1 occ E2 unocc I believe
        return omega_part_3(self.omega, self.smr_fixed_width, E1, E2)

    def factor_Efermi(self, E1, E2):
	    return self.FermiDirac(E1) - self.FermiDirac(E2)