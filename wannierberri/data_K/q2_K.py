from functools import cached_property
import numpy as np
# from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom, speed_of_light
from ..formula.formula import Matrix_GenDer_ln
from ..formula.covariant import DerDcov, Der2Dcov, Der2A, Dcov

speed_of_light = 3e10 #cm/s
elementary_charge = 4.8032e-10 #cm^3/2 g^1/2 s^-1
hbar = 1.0546e-27 #cm^2 g s^-1
electron_mass = 9.1093837139e-28 #g
    
class Q2_K:
    def __init__(self, data_K):
        self.eV_to_erg = 1.602176633e-12
        self.A_to_cm = 10e-8
        
        self.data_K = data_K
        self.eta = 0.04
        self.dEnm_threshold = 1e-3
        En = self.data_K.E_K
        self.kron = np.array(abs(En[:, :, None] - En[:, None, :]) < self.dEnm_threshold, dtype=int)
        self.anti_kron = ( np.ones((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann)) - self.kron)

    def symmetrize_axes(self, arr, axes_to_symmetrize):
        from itertools import permutations
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
                    new_order.append(perm[axes_to_symmetrizendex(axis)])
                else:
                    # This axis is not being permuted; keep its original position
                    new_order.append(axis)
            symmetrized += np.transpose(arr, new_order)
        
        # Average over all permutations
        symmetrized /= len(all_permutations)
        
        return symmetrized

    @cached_property
    def levicivita(self):
        summ = np.zeros((3,3,3), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            summ[l,a,b] += s
        return summ

    @cached_property
    def E_K(self):
        return self.data_K.E_K * self.eV_to_erg

    @cached_property
    def dE(self):
        return np.diagonal(self.data_K.Xbar('Ham', 1), axis1=1, axis2=2).transpose(0,2,1) * self.eV_to_erg * self.A_to_cm

    @cached_property
    def ddE(self):
        # PhysRevB.75.195121 28
        dH = self.data_K.Xbar('Ham', 1) * self.eV_to_erg * self.A_to_cm
        ddH = self.data_K.Xbar('Ham', 2) * self.eV_to_erg * self.A_to_cm**2

        sc_eta = 0.04
        E_K = self.E_K
        dEig = E_K[:, :, None] - E_K[:, None, :]
        dEig_inv_Pval = dEig / (dEig ** 2 + sc_eta ** 2)
        D_H_Pval = -dH * dEig_inv_Pval[:, :, :, None]

        dHD_part = np.einsum('knla, klmb -> knmab', dH, D_H_Pval)
        ddE = ddH + dHD_part + np.conjugate(dHD_part.swapaxes(1,2))
        return np.diagonal(ddE, axis1=1, axis2=2).transpose(0,3,1,2) 

    @cached_property
    def invEdif(self):
        return self.data_K.dEig_inv / self.eV_to_erg

    # @cached_property
    # def S(self):
    #     return self.data_K.Xbar('SS') * hbar/2

    @cached_property
    def A_H(self):
        return self.data_K.A_H * self.A_to_cm

    @cached_property
    def A_H_internal(self):
        A_int = self.data_K.A_H_internal
        return A_int * self.A_to_cm

    @cached_property
    def velocity(self):
        # eq 61, in SI
        E = self.E_K
        A = self.A_H
        dE = self.dE # knma
        dEnm = np.zeros((dE.shape[0],dE.shape[1],dE.shape[1],dE.shape[2]), dtype=complex)
        rows, cols = np.diag_indices(dE.shape[1])
        dEnm[:,rows,cols,:] = dE
        V = dEnm + 1j*(E[:,:,None,None] - E[:,None,:,None])* A
        return V / hbar

    @cached_property
    def velocity_internal(self):
        # eq 61
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
        gender_A = self.gender_A_H
        
        Omega = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3,), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            Omega[...,l] += s * gender_A[:,:,:,b,a] 
        return Omega 

    @cached_property
    def berry_curvature_internal(self):
        # in SI (m**2)
        gender_A = self.gender_A_H_internal
        
        Omega = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3,), dtype=complex)
        for s,l,a,b in [(1,0,1,2), (1,1,2,0), (1,2,0,1),
                        (-1,0,2,1), (-1,1,0,2), (-1,2,1,0)]:
            Omega[...,l] += s * gender_A[:,:,:,b,a] 
        return Omega 

    @cached_property
    def gender2_velocity_internal(self):
        # Eq (D4), SI
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
    def magnetic_dipole_spin(self):
        # Eq (D2), off diag
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        summ += 1 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('kmnl -> kmnl', S)
        return summ

    @cached_property
    def magnetic_dipole_orb(self):
        # Eq (D2), off diag
        V = self.velocity
        dE = self.dE
        A = self.A_H
        lev = self.levicivita
        anti_kron = self.anti_kron

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        # summ += 1/2 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kmpa, kpnb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/2 * elementary_charge * 1/speed_of_light * np.einsum('kmp, kmpa, kpnb, lab -> kmnl', anti_kron, A, V, lev)
        return summ

    @cached_property
    def magnetic_dipole_inter(self):
        # Eq (D2), off diag
        dE = self.dE
        A = self.A_H
        M_spin = self.magnetic_dipole_spin
        M_orb = self.magnetic_dipole_orb
        lev = self.levicivita
        
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        summ += M_spin + M_orb
        summ += elementary_charge / (2 * speed_of_light) * 1/hbar * np.einsum('knb, knma, lab -> knml', dE, A, lev)
        return summ

    @cached_property
    def magnetic_dipole(self):
        # Eq (D2), off diag
        V = self.velocity
        dE = self.dE
        A = self.A_H
        lev = self.levicivita
        anti_kron = self.anti_kron
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kmpa, kpnb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kpna, kmpb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('kmnl -> kmnl', S)
        return summ

    @cached_property
    def magnetic_dipole(self):
        # Eq (D2), off diag
        V = self.velocity
        dE = self.dE
        A = self.A_H
        lev = self.levicivita
        anti_kron = self.anti_kron
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kmpa, kpnb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kpna, kmpb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
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
        anti_kron = self.anti_kron
        try:
            S = self.data_K.Xbar('SS') * hbar/2         
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3), dtype=complex)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kmpa, kpnb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/4 * elementary_charge * 1/speed_of_light * np.einsum('knp, kpm, kpna, kmpb, lab -> kmnl', anti_kron, anti_kron, A, V, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1/2 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, kmna, lab -> kmnl', dE, A, lev)
        summ += 1 * elementary_charge * 1/electron_mass * 1/speed_of_light * S
        return summ

    @cached_property
    def electric_quadrupole(self):
        A = self.A_H
        anti_kron = self.anti_kron

        Q_P = elementary_charge/4 * (np.einsum('kmpj, kpni, kmp, knp -> kmnji', A, A, anti_kron, anti_kron)
                                   + np.einsum('kmpi, kpnj, kmp, knp -> kmnji', A, A, anti_kron, anti_kron))
        return Q_P

    @cached_property
    def electric_quadrupole_internal(self):
        A = self.A_H_internal
        anti_kron = self.anti_kron

        Q_P = elementary_charge/4 * (np.einsum('kmpj, kpni, kmp, knp -> kmnji', A, A, anti_kron, anti_kron)
                                   + np.einsum('kmpi, kpnj, kmp, knp -> kmnji', A, A, anti_kron, anti_kron))
        return Q_P

    @cached_property
    def magnetic_quadrupole(self, spin):
        A = self.A_H
        dA = self.gender_A_H
        V = self.velocity
        dE = self.dE
        ddE = self.ddE
        lev = self.levicivita
        anti_kron = self.anti_kron
        try:
            S = self.S
        except:
            S = np.zeros(V.shape)

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ += 1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('klmja, klm, knl, knlb, iab -> knmij', dA, anti_kron, anti_kron, V, lev)
        summ += -1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('knlja, klm, knl, klmb, iab -> knmij', dA, anti_kron, anti_kron, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klm, kns, ksl, knsa, klmj, kslb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klm, kns, ksl, ksla, klmj, knsb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('kls, knl, ksm, klsa, knlj, ksmb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('kls, knl, ksm, ksma, knlj, klsb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knmja, iab -> knmij', dE, dA, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knmja, iab -> knmij', dE, dA, lev)
        summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klm, knl, klma, knlj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klm, knl, knla, klmj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, klm, knl, klma, knlj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, klm, knl, knla, klmj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('klmi, klm, knl, knlj -> knmij', S, anti_kron, anti_kron, A)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('knli, klm, knl, klmj -> knmij', S, anti_kron, anti_kron, A)
        return summ


    @cached_property
    def magnetic_quadrupole_internal(self):
        A = self.A_H_internal
        dA = self.gender_A_H_internal
        V = self.velocity_internal
        dE = self.dE
        ddE = self.ddE
        lev = self.levicivita
        anti_kron = self.anti_kron
        try:
            S = self.S         
        except:
            S = np.zeros(V.shape)
        
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        # summ += 1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('klmja, klm, knl, knlb, iab -> knmij', dA, anti_kron, anti_kron, V, lev)
        # summ += -1/12 * elementary_charge * 1/speed_of_light * 1j * np.einsum('knlja, klm, knl, klmb, iab -> knmij', dA, anti_kron, anti_kron, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klm, kns, ksl, knsa, klmj, kslb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('klm, kns, ksl, ksla, klmj, knsb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('kls, knl, ksm, klsa, knlj, ksmb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        summ += 1/12 * elementary_charge * 1/speed_of_light * np.einsum('kls, knl, ksm, ksma, knlj, klsb, iab -> knmij', anti_kron, anti_kron, anti_kron, A, A, V, lev)
        # summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmb, knmja, iab -> knmij', dE, dA, lev)
        # summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knb, knmja, iab -> knmij', dE, dA, lev)
        # summ += -1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('kmaj, knmb, iab -> knmij', ddE, A, lev)
        # summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * 1j * np.einsum('knaj, knmb, iab -> knmij', ddE, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klm, knl, klma, knlj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('klb, klm, knl, knla, klmj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('kmb, klm, knl, klma, knlj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/12 * elementary_charge * 1/hbar * 1/speed_of_light * np.einsum('knb, klm, knl, knla, klmj, iab -> knmij', dE, anti_kron, anti_kron, A, A, lev)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('klmi, klm, knl, knlj -> knmij', S, anti_kron, anti_kron, A)
        summ += 1/2 * elementary_charge * 1/electron_mass * 1/speed_of_light * np.einsum('knli, klm, knl, klmj -> knmij', S, anti_kron, anti_kron, A)
        return summ

    @cached_property
    def electric_octupole(self):
        A = self.A_H
        dA = self.gender_A_H
        ddA = self.gender2_A_H
        anti_kron = self.anti_kron

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)
        summ += -1/36 * elementary_charge * np.einsum('qmnklj -> qnmjlk', ddA)
        summ += -1/72 * elementary_charge * 1j * np.einsum('qmsjl, qns, qsm, qsnk -> qnmjlk', dA, anti_kron, anti_kron, A)
        summ += 1/72 * elementary_charge * 1j * np.einsum('qsnkl, qns, qsm, qmsj -> qnmjlk', dA, anti_kron, anti_kron, A)
        summ += 1/36 * elementary_charge * np.einsum('qmp, qps, qsn, qmpj, qsnk, qpsl -> qnmjlk', anti_kron, anti_kron, anti_kron, A, A, A)
        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1-3)
        kjl = klj.swapaxes(-1,-2)
        return jlk + jkl + ljk + lkj + klj + kjl

    @cached_property
    def electric_octupole_internal(self):
        A = self.A_H_internal
        dA = self.gender_A_H_internal
        ddA = self.gender2_A_H_internal
        anti_kron = self.anti_kron

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)
        # summ += -1/36 * elementary_charge * np.einsum('qmnklj -> qnmjlk', ddA)
        summ += -1/72 * elementary_charge * 1j * np.einsum('qmsjl, qns, qsm, qsnk -> qnmjlk', dA, anti_kron, anti_kron, A)
        summ += 1/72 * elementary_charge * 1j * np.einsum('qsnkl, qns, qsm, qmsj -> qnmjlk', dA, anti_kron, anti_kron, A)
        summ += 1/36 * elementary_charge * np.einsum('qmp, qps, qsn, qmpj, qsnk, qpsl -> qnmjlk', anti_kron, anti_kron, anti_kron, A, A, A)
        
        jlk = summ
        jkl = jlk.swapaxes(-1,-2)
        ljk = summ.swapaxes(-2,-3)
        lkj = ljk.swapaxes(-1,-2)
        klj = summ.swapaxes(-1,-3)
        kjl = klj.swapaxes(-1,-2)
        return jlk + jkl + ljk + lkj + klj + kjl

        # return self.symmetrize_axes(summ, [-3, -2, -1])

####################################################################
# GENERELAZIED DERIVATIVES OF A
####################################################################

    @cached_property
    def Edif(self):
        # J
        E = self.E_K
        return E[:,:,None] - E[:,None,:]

    @cached_property
    def dDelta(self):
        D = self.data_K.D_H * self.A_to_cm
        dH = self.data_K.Xbar('Ham', 1) * self.eV_to_erg * self.A_to_cm
        ddH = self.data_K.Xbar('Ham', 2) * self.eV_to_erg * self.A_to_cm**2
        Edif = self.Edif
        anti_kron = self.anti_kron

        nn = np.zeros((self.data_K.nk, self.data_K.num_wann, 3, 3), dtype=complex)
        nn += 1j * np.einsum('knp, knpc, kpna -> knac', anti_kron**2, D, dH) #0
        nn -= 1j * np.einsum('knp, kpnc, knpa -> knac', anti_kron**2, D, dH) #1
        nn += np.einsum('knnac -> knac', ddH) #2
        return nn[:,:,None,:] - nn[:,None,:,:]

    @cached_property
    def Delta(self):
        # this needs testing, is dE_dif the same as dvnn + dvmm
        dE = self.dE
        return dE[:,:,None,:] - dE[:,None,:,:]


    @cached_property
    def gender_A_H(self):
        A_bar = self.data_K.Xbar('AA') * self.A_to_cm
        dA_bar = self.data_K.Xbar('AA', 1) * self.A_to_cm**2
        dH = self.data_K.Xbar('Ham', 1) * self.eV_to_erg * self.A_to_cm
        ddH = self.data_K.Xbar('Ham', 2) * self.eV_to_erg * self.A_to_cm**2

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        inv_Edif = self.invEdif
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
        return summ

    @cached_property
    def gender_A_H_internal(self):
        dH = self.data_K.Xbar('Ham', 1) * self.eV_to_erg * self.A_to_cm
        ddH = self.data_K.Xbar('Ham', 2) * self.eV_to_erg * self.A_to_cm**2

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        inv_Edif = self.invEdif
        inv_Edif_eta = Edif / (Edif ** 2 + self.eta ** 2)
        anti_kron = self.anti_kron

        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3), dtype=complex)
        summ -= 1j * np.einsum('knm, knmab -> knmab', inv_Edif, ddH) #0
        summ += 1j * np.einsum('knm, kpm, knpa, kpmb, kmp, knp -> knmab', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #1
        summ -= 1j * np.einsum('knm, knp, kpma, knpb, kmp, knp -> knmab', inv_Edif, inv_Edif_eta, dH, dH, anti_kron, anti_kron) #2
        summ -= 1j * np.einsum('knm, knma, kmmb -> knmab', inv_Edif**2, dH, dH) #3
        summ += 1j * np.einsum('knm, knma, knnb -> knmab', inv_Edif**2, dH, dH) #4
        summ += 1j * np.einsum('knm, knmb, knma -> knmab', inv_Edif**2, dH, Delta) #5
        return summ

    @cached_property
    def gender2_A_H_internal(self):
        # TB: A_bar -> 0, only term is gender of eq 32
        dH = self.data_K.Xbar('Ham', 1) * self.eV_to_erg * self.A_to_cm
        ddH = self.data_K.Xbar('Ham', 2) * self.eV_to_erg * self.A_to_cm**2
        dddH = self.data_K.Xbar('Ham', 3) * self.eV_to_erg * self.A_to_cm**3
        D = self.data_K.D_H * self.A_to_cm

        Edif = self.Edif
        Delta = self.Delta # or self.dE_dif
        dDelta = self.dDelta
        inv_Edif = self.invEdif
        inv_Edif_eta = Edif / (Edif ** 2 + self.eta ** 2)
        gender_A = self.gender_A_H_internal
        anti_kron = self.anti_kron
        
        # different orders of 1/omega_nm
        summ = np.zeros((self.data_K.nk, self.data_K.num_wann, self.data_K.num_wann, 3, 3, 3), dtype=complex)

        # NOTE: Use anti_kronekers for the summations
        summ += -1 * np.einsum('knm, kpm, kmp, kppc, knpa, kpmb, knp, kpp -> knmabc', inv_Edif, inv_Edif_eta, anti_kron**2, D, dH, dH, anti_kron, anti_kron) #0
        summ += 1 * np.einsum('knm, kpm, knp, kppc, knpa, kpmb, kmp, kpp -> knmabc', inv_Edif, inv_Edif_eta, anti_kron**2, D, dH, dH, anti_kron, anti_kron) #1
        summ += 1 * np.einsum('knm, knp, kmp, kppc, kpma, knpb, knp, kpp -> knmabc', inv_Edif, inv_Edif_eta, anti_kron**2, D, dH, dH, anti_kron, anti_kron) #2
        summ += -1 * np.einsum('knm, knp, knp, kppc, kpma, knpb, kmp, kpp -> knmabc', inv_Edif, inv_Edif_eta, anti_kron**2, D, dH, dH, anti_kron, anti_kron) #3
        summ += -1 * 1j * np.einsum('knm, knmabc -> knmabc', inv_Edif, dddH) #4
        summ += 1 * 1j * np.einsum('knm, kpm, knpa, kpmbc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #5
        summ += 1 * 1j * np.einsum('knm, kpm, kpmb, knpac, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #6
        summ += 1 * 1j * np.einsum('knm, kpm, kpmc, knpab, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #7
        summ += 1 * 1j * np.einsum('knm, kpm, knpa, kmmb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #8
        summ += 1 * 1j * np.einsum('knm, kpm, knpa, kpmb, kmmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #9
        summ += -1 * 1j * np.einsum('knm, kpm, knpa, kpmb, kppc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #10
        summ += -1 * 1j * np.einsum('knm, kpm, kmp, knpa, kppb, kpmc, knp, kpp -> knmabc', inv_Edif, inv_Edif_eta**2, anti_kron**2, dH, dH, dH, anti_kron, anti_kron) #11
        summ += -1 * 1j * np.einsum('knm, kpm, knpa, kppb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #12
        summ += -1 * 1j * np.einsum('knm, knp, kpma, knpbc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #13
        summ += -1 * 1j * np.einsum('knm, knp, knpb, kpmac, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #14
        summ += -1 * 1j * np.einsum('knm, knp, knpc, kpmab, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, dH, ddH, anti_kron, anti_kron) #15
        summ += -1 * 1j * np.einsum('knm, knp, kpm, kmma, knpb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #16
        summ += -1 * 1j * np.einsum('knm, knp, kpm, knna, kpmb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #17
        summ += 1 * 1j * np.einsum('knm, knp, kpm, kmp, kppa, knpb, kpmc, knp, kpp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, anti_kron**2, dH, dH, dH, anti_kron, anti_kron) #18
        summ += 1 * 1j * np.einsum('knm, knp, kpm, kppa, knpb, kpmc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #19
        summ += 1 * 1j * np.einsum('knm, knp, kpm, knp, kppa, kpmb, knpc, kmp, kpp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, anti_kron**2, dH, dH, dH, anti_kron, anti_kron) #20
        summ += 1 * 1j * np.einsum('knm, knp, kpm, kppa, kpmb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #21
        summ += 1 * 1j * np.einsum('knm, knp, kpma, knnb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #22
        summ += 1 * 1j * np.einsum('knm, knp, kpma, knpb, knnc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #23
        summ += -1 * 1j * np.einsum('knm, knp, kpma, knpb, kppc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #24
        summ += -1 * 1j * np.einsum('knm, knp, knp, kpma, kppb, knpc, kmp, kpp -> knmabc', inv_Edif, inv_Edif_eta**2, anti_kron**2, dH, dH, dH, anti_kron, anti_kron) #25
        summ += -1 * 1j * np.einsum('knm, knp, kpma, kppb, knpc, kmp, knp -> knmabc', inv_Edif, inv_Edif_eta**2, dH, dH, dH, anti_kron, anti_kron) #26
        summ += 1 * 1j * np.einsum('knm, knma, knmbc -> knmabc', inv_Edif**2, dH, dDelta) #27
        summ += 1 * 1j * np.einsum('knm, knmb, knmac -> knmabc', inv_Edif**2, dH, dDelta) #28
        summ += -1 * 1j * np.einsum('knm, kmmc, knmab -> knmabc', inv_Edif**2, dH, ddH) #29
        summ += -1 * 1j * np.einsum('knm, knmc, kmmab -> knmabc', inv_Edif**2, dH, ddH) #30
        summ += 1 * 1j * np.einsum('knm, knmc, knnab -> knmabc', inv_Edif**2, dH, ddH) #31
        summ += 1 * 1j * np.einsum('knm, knnc, knmab -> knmabc', inv_Edif**2, dH, ddH) #32
        summ += 1 * 1j * np.einsum('knm, knmac, knmb -> knmabc', inv_Edif**2, ddH, Delta) #33
        summ += 1 * 1j * np.einsum('knm, knmbc, knma -> knmabc', inv_Edif**2, ddH, Delta) #34
        summ += 1 * 1j * np.einsum('knm, kpm, knpa, kpmb, kmmc, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #35
        summ += -1 * 1j * np.einsum('knm, kpm, knpa, kpmb, knnc, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #36
        summ += -1 * 1j * np.einsum('knm, kpm, knpa, kpmc, knmb, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #37
        summ += -1 * 1j * np.einsum('knm, kpm, knpb, kpmc, knma, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #38
        summ += -1 * 1j * np.einsum('knm, knp, kpma, knpb, kmmc, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #39
        summ += 1 * 1j * np.einsum('knm, knp, kpma, knpb, knnc, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, dH, anti_kron, anti_kron) #40
        summ += 1 * 1j * np.einsum('knm, knp, kpma, knpc, knmb, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #41
        summ += 1 * 1j * np.einsum('knm, knp, kpmb, knpc, knma, kmp, knp -> knmabc', inv_Edif**2, inv_Edif_eta, dH, dH, Delta, anti_kron, anti_kron) #42
        summ += 1 * 1j * np.einsum('knm, kmma, knmc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #43
        summ += 2 * 1j * np.einsum('knm, knma, kmmc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #44
        summ += -2 * 1j * np.einsum('knm, knma, knnc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #45
        summ += -1 * 1j * np.einsum('knm, knna, knmc, knmb -> knmabc', inv_Edif**3, dH, dH, Delta) #46
        summ += 1 * 1j * np.einsum('knm, kmmb, knmc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #47
        summ += 2 * 1j * np.einsum('knm, knmb, kmmc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #48
        summ += -2 * 1j * np.einsum('knm, knmb, knnc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #49
        summ += -1 * 1j * np.einsum('knm, knnb, knmc, knma -> knmabc', inv_Edif**3, dH, dH, Delta) #50
        return summ

#########################################################
# MISC
#########################################################

    def off_diag(self, A):
        shape_dif = A.ndim - 3 #self.kron.ndim
        kron = self.kron.reshape(self.kron.shape + (1,) * shape_dif)
        A_diag = kron * A
        return A - A_diag
        