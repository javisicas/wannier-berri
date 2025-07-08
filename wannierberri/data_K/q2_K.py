from functools import cached_property
import numpy as np
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, speed_of_light
from ..formula.formula import Matrix_GenDer_ln
from ..formula.covariant import DerDcov, Der2Dcov, Der2A, Dcov

class Q2_K:
    def __init__(self, data_K):
        self.eV_to_J = 1.602176634e-19
        self.A_to_m = 10e-10
        
        self.data_K = data_K
        self.eta = 0.01
        self.dEnm_threshold = 1e-3
        En = self.data_K.E_K
        self.kron = np.array(abs(En[:, :, None] - En[:, None, :]) < self.dEnm_threshold, dtype=int)
        self.anti_kron = ( np.ones((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann))
                         - np.eye(self.data_K.num_wann)[None,:,:])

    @cached_property
    def levicivita(self):
        summ = np.zeros((3,3,3), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            summ[l,a,b] += s
        return summ

    @cached_property
    def E_K(self):
        return self.data_K.E_K * self.eV_to_J

    @cached_property
    def dE(self):
        return np.diagonal(self.data_K.Xbar('Ham', 1), axis1=1, axis2=2).transpose(0,2,1) * self.eV_to_J * self.A_to_m

    @cached_property
    def ddE(self):
        return np.diagonal(self.data_K.Xbar('Ham', 2), axis1=1, axis2=2).transpose(0,3,1,2) * self.eV_to_J * self.A_to_m**2

    @cached_property
    def invEdif(self):
        return self.data_K.dEig_inv / self.eV_to_J

    @cached_property
    def S(self):
        return self.data_K.Xbar('SS') * hbar/2

    @cached_property
    def A_H(self):
        return self.data_K.A_H * self.A_to_m

    @cached_property
    def A_H_internal(self):
        return self.data_K.A_H_internal * self.A_to_m

    @cached_property
    def velocity(self):
        # eq 61, in SI
        E = self.E_K
        A = self.A_H
        dE = self.dE # knma
        dEnm = np.zeros((dE.shape[0],dE.shape[1],dE.shape[1],dE.shape[2]))
        rows, cols = np.diag_indices(dE.shape[1])
        dEnm[:,rows,cols,:] = dE
        V = dEnm + 1j*(E[:,:,None,None] - E[:,None,:,None])* A
        return V / hbar

    @cached_property
    def velocity_internal(self):
        # eq 61, in SI
        E = self.E_K
        A = self.A_H_internal
        dE = self.dE # knma
        dEnm = np.zeros((dE.shape[0],dE.shape[1],dE.shape[1],dE.shape[2]), dtype=complex)
        rows, cols = np.diag_indices(dE.shape[1])
        dEnm[:,rows,cols,:] = dE
        V = dEnm + 1j*(E[:,:,None,None] - E[:,None,:,None])* A
        return V / hbar

    @cached_property
    def berry_curvature(self):
        # in SI (m**2)
        gender_A = self.gender_A_H
        
        Omega = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3,), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            Omega[...,l] += s * gender_A[:,:,:,a,b] 
        return Omega 

    @cached_property
    def berry_curvature_internal(self):
        # in SI (m**2)
        gender_A = self.gender_A_H_internal
        
        Omega = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3,), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            Omega[...,l] += s * gender_A[:,:,:,a,b] 
        return Omega 

    @cached_property
    def gender2_velocity_internal(self):
        # Eq (D4), off diagonal in units of eV*A^3/hbar
        E = self.E_K
        dE = self.dE
        ddE = self.ddE
        A = self.A_H_internal #knmd
        gender_A = self.gender_A_H_internal #knmd(a/c)
        gender2_A = self.gender2_A_H_internal #knmdca
        
        E_dif = E[:,:,None] - E[:,None,:]
        dE_dif = dE[:,:,None,:] - dE[:,None,:,:] #knmc
        ddE_dif = ddE[:,:,None,:,:] - ddE[:,None,:,:,:] #knmac
        
        gender2_V = np.zeros(np.shape(gender2_A), dtype=complex) # knmdca
        gender2_V += 1j * np.einsum('knm, knmdca -> knmdca', E_dif, gender2_A)
        gender2_V += 1j * np.einsum('knmc, knmad -> knmdca', dE_dif, gender_A)
        gender2_V += 1j * np.einsum('knma, knmcd -> knmdca', dE_dif, gender_A)
        gender2_V += 1j * np.einsum('knmca, knmd -> knmdca', ddE_dif, A)
        return gender2_V / hbar

#################################################################
# MULTIPOLAR STUFF
#################################################################

    @cached_property
    def magnetic_dipole(self):
        # Eq (D2), off diag
        V = self.velocity
        dE = self.dE
        A = self.A_H
        lev = self.levicivita
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((V.shape), dtype=complex) # kmnl
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('kmpa, kpnb, lab -> kmnl', A, V, lev)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('kpna, kmpb, lab -> kmnl', A, V, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('kmnl -> kmnl', S)
        return summ

    @cached_property
    def magnetic_dipole_internal(self):
        V = self.velocity_internal
        dE = self.dE
        A = self.A_H_internal
        lev = self.levicivita
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((V.shape), dtype=complex) # kmnl
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('kmpa, kpnb, lab -> kmnl', A, V, lev)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('kpna, kmpb, lab -> kmnl', A, V, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('kmnl -> kmnl', S)
        return summ

    @cached_property
    def electric_quadrupole(self):
        A = self.A_H

        Q_P = elementary_charge/4 * (np.einsum('kmlj, klni -> kmnji', A, A)
                                   + np.einsum('kmli, klnj -> kmnji', A, A))
        return Q_P

    @cached_property
    def electric_quadrupole_internal(self):
        A = self.A_H_internal

        Q_P = elementary_charge/4 * (np.einsum('kmlj, klni -> kmnji', A, A)
                                   + np.einsum('kmli, klnj -> kmnji', A, A))
        return Q_P

    @cached_property
    def magnetic_quadrupole(self, spin):
        A = self.A_H
        dA = self.gender_A_H
        V = self.velocity
        dE = self.dE
        ddE = self.ddE
        lev = self.levicivita
        try:
            S = self.S
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('klmja, knlb, iab -> knmij', dA, V, lev)
        summ += -1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('knlja, klmb, iab -> knmij', dA, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klsa, knlj, ksmb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('knsa, klmj, kslb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('ksla, klmj, knsb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('ksma, knlj, klsb, iab -> knmij', A, A, V, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knmja, iab -> knmij', dE, dA, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knmja, iab -> knmij', dE, dA, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klma, knlj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, knla, klmj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, klma, knlj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, knla, klmj, iab -> knmij', dE, A, A, lev)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('klmi, knlj -> knmij', S, A)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('knli, klmj -> knmij', S, A)
        return summ

    @cached_property
    def magnetic_quadrupole_internal(self):
        A = self.A_H_internal
        dA = self.gender_A_H_internal
        V = self.velocity_internal
        dE = self.dE
        ddE = self.ddE
        lev = self.levicivita
        try:
            S = self.S         
        except:
            S = np.zeros(V.shape)
        
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('klmja, knlb, iab -> knmij', dA, V, lev)
        summ += -1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('knlja, klmb, iab -> knmij', dA, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klsa, knlj, ksmb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('knsa, klmj, kslb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('ksla, klmj, knsb, iab -> knmij', A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('ksma, knlj, klsb, iab -> knmij', A, A, V, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knmja, iab -> knmij', dE, dA, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knmja, iab -> knmij', dE, dA, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klma, knlj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, knla, klmj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, klma, knlj, iab -> knmij', dE, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, knla, klmj, iab -> knmij', dE, A, A, lev)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('klmi, knlj -> knmij', S, A)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('knli, klmj -> knmij', S, A)
        return summ

    @cached_property
    def electric_octupole(self):
        A = self.A_H
        dA = self.gender_A_H
        ddA = self.gender2_A_H

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += np.einsum('qnpk, qpsj, qsml -> qmnjlk', A, A, A)
        summ += +1j/2 * np.einsum('qnsk, qsmlj -> qmnjlk', A, dA)
        summ += -1j/2 * np.einsum('qnskj, qsml -> qmnjlk', dA, A)
        summ += -np.einsum('qmnklj -> qmnjlk', ddA)
        summ *= elementary_charge / 36

        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1-3)
        kjl = klj.swapaxes(-1,-2)
        return 1/6 * (jlk + jkl + ljk + lkj + klj + kjl)

    @cached_property
    def electric_octupole_internal(self):
        A = self.A_H_internal
        dA = self.gender_A_H_internal
        ddA = self.gender2_A_H_internal

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += np.einsum('qnpk, qpsj, qsml -> qmnjlk', A, A, A)
        summ += +1j/2 * np.einsum('qnsk, qsmlj -> qmnjlk', A, dA)
        summ += -1j/2 * np.einsum('qnskj, qsml -> qmnjlk', dA, A)
        summ += -np.einsum('qmnklj -> qmnjlk', ddA)
        summ *= elementary_charge / 36

        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1,-3)
        kjl = klj.swapaxes(-1,-2)

        return jlk + jkl + ljk + lkj + klj + kjl

####################################################################
# GENERELAZIED DERIVATIVES OF A
####################################################################

    @cached_property
    def Edif(self):
        E = self.data_K.E_K
        return E[:,:,None] - E[:,None,:]

    @cached_property
    def dDelta(self):
        # this needs testing, is dE_dif the same as dvnn + dvmm
        dH = self.data_K.Xbar('Ham', 1)
        ddE = self.ddE
        Edif = self.Edif

        result = ddE[:,:,None,:,:] - ddE[:,None,:,:]
        result += -np.einsum('knpb, kpnc, kpn', dH, dH, Edif)
        result += +np.einsum('kmpb, kpmc, kpm', dH, dH, Edif)
        result += +np.einsum('kpnb, knpc, knp', dH, dH, Edif)
        result += -np.einsum('kpmb, kmpc, kmp', dH, dH, Edif)
        return result

    @cached_property
    def Delta(self):
        # this needs testing, is dE_dif the same as dvnn + dvmm
        dE = self.dE
        return dE[:,:,None,:] - dE[:,None,:,:]


    @cached_property
    def gender_A_H(self):
        A_bar = self.data_K.Xbar('AA')
        dA_bar = self.data_K.Xbar('AA', 1)
        dH = self.data_K.Xbar('Ham', 1)
        ddH = self.data_K.Xbar('Ham', 2)

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        inv_Edif = self.data_K.dEig_inv
        inv_Edif_eta = Edif / (Edif ** 2 + self.eta ** 2)
        anti_kron = self.anti_kron
        
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ += 1j * np.einsum('knma, kmmc -> knmac', A_bar, A_bar) #0
        summ -= 1j * np.einsum('knma, knnc -> knmac', A_bar, A_bar) #1
        summ += np.einsum('knmac -> knmac', dA_bar) #2
        summ -= np.einsum('kpm, knpa, kpmc, kmp, knp -> knmac', inv_Edif_eta, A_bar, dH, anti_kron, anti_kron) #3
        summ += np.einsum('knp, kpma, knpc, kmp, knp -> knmac', inv_Edif_eta, A_bar, dH, anti_kron, anti_kron) #4
        summ += np.einsum('knm, kmma, knmc -> knmac', inv_Edif, A_bar, dH) #5
        summ -= np.einsum('knm, knna, knmc -> knmac', inv_Edif, A_bar, dH) #6
        summ += np.einsum('knm, kmmc, knma -> knmac', inv_Edif, A_bar, dH) #7
        summ -= np.einsum('knm, knnc, knma -> knmac', inv_Edif, A_bar, dH) #8
        summ -= 1j * np.einsum('knm, knmac -> knmac', inv_Edif, ddH) #9
        summ += 1j * np.einsum('knm, kpm, knpa, kpmc, kmp, knp -> knmac', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #10
        summ -= 1j * np.einsum('knm, knp, kpma, knpc, kmp, knp -> knmac', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #11
        summ += 1j * np.einsum('knm, knma, knmc -> knmac', inv_Edif**2, dH, Delta) #12
        summ += 1j * np.einsum('knm, knmc, knma -> knmac', inv_Edif**2, dH, Delta) #13
        return summ * self.A_to_m**2

    @cached_property
    def gender_A_H_internal(self):
        dH = self.data_K.Xbar('Ham', 1)
        ddH = self.data_K.Xbar('Ham', 2)

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        inv_Edif = self.data_K.dEig_inv
        inv_Edif_eta = Edif / (Edif ** 2 + self.eta ** 2)
        anti_kron = self.anti_kron

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ -= 1j * np.einsum('knm, knmab -> knmab', inv_Edif, ddH) #0
        summ += 1j * np.einsum('knm, kpm, knpa, kpmb, kmp, knp -> knmab', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #1
        summ -= 1j * np.einsum('knm, knp, kpma, knpb, kmp, knp -> knmab', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #2
        summ -= 1j * np.einsum('knm, knma, kmmb -> knmab', inv_Edif**2, dH, dH) #3
        summ += 1j * np.einsum('knm, knma, knnb -> knmab', inv_Edif**2, dH, dH) #4
        summ += 1j * np.einsum('knm, knmb, knma -> knmab', inv_Edif**2, dH, Delta) #5
        return summ * self.A_to_m**2

    @cached_property
    def gender2_A_H_internal(self):
        # TB: A_bar -> 0, only term is gender of eq 32
        dH = self.data_K.Xbar('Ham', 1)
        ddH = self.data_K.Xbar('Ham', 2)
        dddH = self.data_K.Xbar('Ham', 3)

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        dDelta = self.dDelta
        inv_Edif = self.data_K.dEig_inv
        inv_Edif_eta = Edif / (Edif ** 2 + self.eta ** 2)
        gender_A = self.gender_A_H_internal
        anti_kron = self.anti_kron
        
        # different orders of 1/omega_nm
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)

        # NOTE: Use anti_kronekers for the summations
        summ -= 1j * np.einsum('knm, knmabc -> knmabc', inv_Edif, dddH) #0
        summ -= 1j * np.einsum('knm, knmc, knmab -> knmabc', inv_Edif, Delta, gender_A) #1
        summ += 1j * np.einsum('knm, kpm, knpa, kpmbc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #2
        summ += 1j * np.einsum('knm, kpm, kpmb, knpac, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #3
        summ += 1j * np.einsum('knm, kpm, kpmc, knpab, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #4
        summ -= 1j * np.einsum('knm, kpm, kqp, knqa, kpmb, kqpc, kmp, knp, knq, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #5
        summ -= 1j * np.einsum('knm, kpm, kqm, knpa, kpqb, kqmc, kmp, kmq, knp, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #6
        summ += 1j * np.einsum('knm, kpm, kpq, knpa, kqmb, kpqc, kmp, kmq, knp, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #7
        summ += 1j * np.einsum('knm, kpm, knpa, kmmb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #8
        summ -= 1j * np.einsum('knm, kpm, knpa, kpmb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, Delta, anti_kron, anti_kron) #9
        summ -= 1j * np.einsum('knm, kpm, knpa, kppb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #10
        summ += 1j * np.einsum('knm, knq, kpm, kqpa, kpmb, knqc, kmp, knp, knq, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #11
        summ -= 1j * np.einsum('knm, knp, kpma, knpbc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #12
        summ -= 1j * np.einsum('knm, knp, knpb, kpmac, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #13
        summ -= 1j * np.einsum('knm, knp, knpc, kpmab, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #14
        summ += 1j * np.einsum('knm, knp, kqp, kpma, knqb, kqpc, kmp, knp, knq, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #15
        summ += 1j * np.einsum('knm, knp, kqm, kpqa, knpb, kqmc, kmp, kmq, knp, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #16
        summ -= 1j * np.einsum('knm, knp, kpq, kqma, knpb, kpqc, kmp, kmq, knp, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #17
        summ -= 1j * np.einsum('knm, knp, kpm, kmma, knpb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #18
        summ -= 1j * np.einsum('knm, knp, kpm, knna, kpmb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #19
        summ += 1j * np.einsum('knm, knp, kpm, kppa, knpb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #20
        summ += 1j * np.einsum('knm, knp, kpm, kppa, kpmb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #21
        summ -= 1j * np.einsum('knm, knp, knq, kpma, kqpb, knqc, kmp, knp, knq, kpq -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron, anti_kron, anti_kron) #22
        summ += 1j * np.einsum('knm, knp, kpma, knnb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #23
        summ += 1j * np.einsum('knm, knp, kpma, knpb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, Delta, anti_kron, anti_kron) #24
        summ -= 1j * np.einsum('knm, knp, kpma, kppb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #25
        summ += 1j * np.einsum('knm, knma, knmbc -> knmabc', inv_Edif**2, dH, dDelta) #26
        summ += 1j * np.einsum('knm, knmb, knmac -> knmabc', inv_Edif**2, dH, dDelta) #27
        summ -= 1j * np.einsum('knm, knmc, kmmab -> knmabc', inv_Edif**2, dH, ddH) #28
        summ += 1j * np.einsum('knm, knmc, knnab -> knmabc', inv_Edif**2, dH, ddH) #29
        summ += 1j * np.einsum('knm, knmac, knmb -> knmabc', inv_Edif**2, ddH, Delta) #30
        summ += 1j * np.einsum('knm, knmbc, knma -> knmabc', inv_Edif**2, ddH, Delta) #31
        summ -= 1j * np.einsum('knm, kpm, knpa, kpmc, knmb, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #32
        summ -= 1j * np.einsum('knm, kpm, knpb, kpmc, knma, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #33
        summ += 1j * np.einsum('knm, knp, kpma, knpc, knmb, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #34
        summ += 1j * np.einsum('knm, knp, kpmb, knpc, knma, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #35
        summ += 1j * np.einsum('knm, kmma, knmc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #36
        summ -= 1j * np.einsum('knm, knma, knmb, knmc -> knmabc', inv_Edif**3, dH, Delta, Delta) #37
        summ -= 1j * np.einsum('knm, knna, knmc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #38
        summ += 1j * np.einsum('knm, kmmb, knmc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #39
        summ -= 1j * np.einsum('knm, knmb, knma, knmc -> knmabc', inv_Edif**3, dH, Delta, Delta) #40
        summ -= 1j * np.einsum('knm, knnb, knmc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #41
        return summ  * self.A_to_m**3
    
#########################################################
# MISC
#########################################################

    def off_diag(self, A):
        shape_dif = A.ndim - 3 #self.kron.ndim
        kron = self.kron.reshape(self.kron.shape + (1,) * shape_dif)
        A_diag = kron * A
        return A - A_diag
        