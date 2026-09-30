import numpy as np
import qu_alg_qu_sim_tensor_networks.mpo_builder as qmpo
import qu_alg_qu_sim_tensor_networks.mps as mps
# class MPStools:
#     def __init__(self, mps):
#         # Transpose each B tensor once and store the new MPS
#         Bs = [np.transpose(B, (1, 0, 2)) for B in mps.Bs]
#         self.psi = qmpo.MPS(Bs)

#     def expectation_value(self, mpo):
#         return self.compute_expectation(mpo, self.psi)

#     def norm(self):
#         C = np.tensordot(self.psi.Bs[0], np.conj(self.psi.Bs[0]), axes=([0, 1], [0, 1]))
#         for i in range(1, self.psi.L):
#             C = np.tensordot(C, self.psi.Bs[i], axes=([0], [1]))
#             C = np.tensordot(C, np.conj(self.psi.Bs[i]), axes=([0, 1], [1, 0]))
#         return np.squeeze(C)

#     def compute_expectation(self, mpo, psi):
#         C = np.tensordot(mpo.Ms[0], psi.Bs[0], axes=([0], [0]))
#         C = np.tensordot(C, np.conj(psi.Bs[0]), axes=([0], [0]))
#         C = np.transpose(C, (2, 0, 4, 3, 1, 5))  # (a0 w0 b0 a1 w1 b1)

#         for i in range(1, mpo.L):
#             C = np.tensordot(C, psi.Bs[i], axes=(-3, 1))
#             C = np.tensordot(C, mpo.Ms[i], axes=([-2, -4], [0, 2]))
#             C = np.tensordot(C, np.conj(psi.Bs[i]), axes=([-2, -4], [0, 1]))
#         return C.squeeze()
        


def expectation_value(mpo, mps):
    """Compute ⟨psi|MPO|psi⟩ for an MPS in (virtual, physical, virtual) leg order.

    Transposes the MPS tensors from (vL, i, vR) to (i, vL, vR) before contraction
    to match the internal MPO convention.

    Parameters
    ----------
    mpo : MPO
        Matrix product operator.
    mps : MPS
        Matrix product state with tensors in (vL, i, vR) order.

    Returns
    -------
    complex or float
        Expectation value ⟨psi|MPO|psi⟩.
    """
    Bs = []
    for B in mps.Bs:
        Bs.append(np.transpose(B, (1,0,2)))
    psi = qmpo.MPS(Bs)
    O = compute_expectation(mpo, psi)
    return O


def norm_psi(psi, transpose=True):
    """Compute the norm ⟨psi|psi⟩ of an MPS by full contraction.

    Parameters
    ----------
    psi : MPS
        Matrix product state.
    transpose : bool, optional
        If True (default), transpose each tensor from (vL, i, vR) to (i, vL, vR)
        before contraction to match the internal MPO-style convention.

    Returns
    -------
    complex
        Scalar norm ⟨psi|psi⟩ (should be real and positive for a valid state).
    """
    if transpose:
         Bs = []
         for B in psi.Bs:
             Bs.append(np.transpose(B, (1,0,2)))
    
    phi = qmpo.MPS(Bs)
    C = np.tensordot(phi.Bs[0], np.conj(phi.Bs[0]), axes=([0,1],[0,1]) )  #(a1, b1)
    for i in range(1,phi.L):
        #print(C.shape, phi.Bs[i].shape)
        C = np.tensordot(C, phi.Bs[i], axes=([0],[1]) )  #(b1, i, a2)
        C = np.tensordot(C, np.conj(phi.Bs[i]), axes=([0,1],[1,0]))
    return np.squeeze(C)


def dvv_vdv_mps_swap(psii):
    """Convert MPS tensors from (i, vL, vR) to (vL, i, vR) leg ordering.

    Parameters
    ----------
    psii : MPS (qmpo.MPS)
        MPS with tensors stored as (physical, vL, vR).

    Returns
    -------
    qmpo.MPS
        New MPS with tensors stored as (vL, physical, vR).
    """
    Bs = []
    for B in psii.Bs:
        B = np.transpose(B, (1,0,2))
        Bs.append(B)

    psi_sp=qmpo.MPS(Bs, psii.Ss)
    #print(psi_sp.L)
    return psi_sp


def vvdd_ddvv_mpo_swap(wmpo):
    """This transpose the indices of tensors in MPO and output is physical-physicl-virtual-virtual (ddvv)

    Args:
        wmpo (_type_): _description_

    Returns:
        _type_: _description_
    """
    Ms = []
    for i in range(wmpo.L):
        Ms.append(np.transpose(wmpo.Ms[i], (2,3,0,1)))
    model = qmpo.MPO(Ms, Ss=None, bonds=None)
    return model




def compute_expectation(mpo: qmpo.MPO, mps: qmpo.MPS):
    """Compute ⟨psi|MPO|psi⟩ by full tensor-network contraction, site by site.

    Expects MPS tensors in (physical, vL, vR) order and MPO tensors in
    (physical_out, physical_in, vL, vR) order (the mpo_builder convention).

    Parameters
    ----------
    mpo : qmpo.MPO
        Matrix product operator.
    mps : qmpo.MPS
        Matrix product state with tensors in (i, vL, vR) order.

    Returns
    -------
    complex
        Scalar expectation value ⟨psi|MPO|psi⟩.
    """
    #print(mpo.Ms[4].shape)
    C = np.tensordot(mpo.Ms[0], mps.Bs[0], axes=([0],[0])) #(t w0 w1 a0 a1  ) 
    C = np.tensordot(C, np.conj(mps.Bs[0]), axes=([0],[0])) #(w0 w1 a0 a1 b0 b1)
    C = np.transpose(C, (2,0,4,3,1,5))  #(a0 w0 b0 a1 w1 b1)
    #print(C.shape)
    for i in range(1, mpo.L):
        #print(mps.Bs[i].shape)
        C = np.tensordot(C, mps.Bs[i], axes=(-3,1)) #(a0 w0 b0 w1 b1 i a11)
        C = np.tensordot(C, mpo.Ms[i], axes=([-2, -4],[0,2])) #(a0 w0 b0 b1 a11, t, w11 )
        #print(i, C.shape,np.conj(mps.Bs[i]).shape )
        C = np.tensordot(C, np.conj(mps.Bs[i]), axes=([-2, -4],[0,1]))  #(a0 w0 b0 a11 w11 b11 )
        #print(C.shape, i)
    return C.squeeze()


def op_bond_expectation_value(phi, op, i, d=2, VdVForm=True):
    """Compute the single-site expectation value ⟨phi|op_i|phi⟩ at site i.

    Parameters
    ----------
    phi : MPS
        Matrix product state (vL, i, vR) or (i, vL, vR) depending on VdVForm.
    op : np.ndarray
        Local operator (d x d matrix).
    i : int
        Site index.
    d : int, optional
        Physical dimension (default 2).
    VdVForm : bool, optional
        If True (default), transpose phi tensors to (vL, physical, vR) before use.

    Returns
    -------
    complex
        Single-site expectation value (scalar).
    """
    #sz=np.array([[1/2.,0.],[0.,-1/2.]])  
    if VdVForm:
        psi = dvv_vdv_mps_swap(phi)
    sBB = np.tensordot(np.diag(psi.Ss[i]), psi.Bs[i], axes=(1,1)) # (a_i, i, a_i+1)
    C = np.tensordot(sBB, op, axes=(1,0))  #(a_i, a_i+1, i)
    sBB = np.conj(sBB)
    op_i = np.squeeze(np.tensordot(sBB,C,axes=([0,2,1],[0,1,2]))).item()
    return op_i


def mpo_on_mps(op, psi, L, d=2, bc='finite'):
    """Apply an MPO to an MPS, returning a new MPS with enlarged bond dimension.

    The output bond dimension is the product of the MPO and MPS bond dimensions
    (no truncation is performed).

    Parameters
    ----------
    op : MPO
        Matrix product operator with tensors of shape (i*, i, vL, vR).
    psi : MPS
        Input MPS with tensors of shape (vL, i, vR).
    L : int
        Number of sites.
    d : int, optional
        Physical dimension (default 2).
    bc : str, optional
        Boundary condition, 'finite' or 'infinite' (default 'finite').

    Returns
    -------
    MPS
        Resulting MPS with tensors of shape (vL_op*vL_psi, i, vR_op*vR_psi).
    """
    Bs=[]
    for i in range(L):
        B = np.tensordot(op.Ms[i], psi.Bs[i], axes=([0],[1])) #(s', bl, br, al, ar)
        a_l = psi.Bs[i].shape[0]
        a_r = psi.Bs[i].shape[2]
        b_l = op.Ms[i].shape[2]
        b_r = op.Ms[i].shape[3]
        B = np.transpose(B, (1, 3, 0, 2, 4))
        B = np.reshape(B, (a_l*b_l, d, a_r*b_r))
        Bs.append(B)
        S = np.ones([1], dtype=float)
    Ss = [S.copy() for i in range(L)]
    return mps.MPS(Bs, Ss, bc=bc)   


def expectation_value(mpo, mps):
    # compute_expectation contracts with NumPy; coerce the MPS tensors to the
    # host so this works after a GPU (cupy) DMRG run (no-op on CPU).
    from qu_alg_qu_sim_tensor_networks.backend import to_numpy
    Bs = []
    for B in mps.Bs:
        Bs.append(np.transpose(to_numpy(B), (1,0,2)))
    psi = qmpo.MPS(Bs)
    O = compute_expectation(mpo, psi)
    return O



def two_mps_overlap_dvv(psi_1, psi_2, L, d=2):
    """Compute the overlap ⟨psi_2|psi_1⟩ for MPS in (physical, vL, vR) leg order.

    Parameters
    ----------
    psi_1 : MPS
        Ket MPS with tensors of shape (i, vL, vR).
    psi_2 : MPS
        Bra MPS with tensors of shape (i, vL, vR).
    L : int
        Number of sites.
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    complex
        Scalar overlap ⟨psi_2|psi_1⟩.
    """
    Blist = psi_1.Bs
    SBlist = psi_2.Bs
    B_dag = np.conj(SBlist[0].T)    #( a_i+1, a_i, i )
    BB = np.tensordot(Blist[0], B_dag , axes=([0],[2]) ) # (a_i, a_i+1, a_i+1, a_i)
    C = BB
    i = 1
    while(i<L):
        B_dag = np.conj(SBlist[i].T)    #( a_i+2, a_i+1, i+1 )
        BB = np.tensordot(Blist[i], B_dag , axes=([0],[2]) ) # (a_i+1, a_i+2, a_i+2, a_i+1)
        C = np.tensordot(C, BB, axes=([1,2],[0,3]))
        C = np.transpose(C, (0,2,3,1))
        i = i+1
    return np.squeeze(C)


def two_mps_overlap_vdv(psi_1, psi_2, L, d=2):
    """Compute the overlap ⟨psi_2|psi_1⟩ for MPS in (vL, physical, vR) leg order.

    Parameters
    ----------
    psi_1 : MPS
        Ket MPS with tensors of shape (vL, i, vR).
    psi_2 : MPS
        Bra MPS with tensors of shape (vL, i, vR).
    L : int
        Number of sites.
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    complex
        Scalar overlap ⟨psi_2|psi_1⟩.
    """
    Blist = psi_1.Bs
    SBlist = psi_2.Bs
    B_dag = np.conj(SBlist[0].T)    #( a_i+1,i, a_i,)
    BB = np.tensordot(Blist[0], B_dag , axes=([1],[1]) ) # (a_i, a_i+1, a_i+1, a_i)
    C = BB
    i = 1
    while(i<L):
        B_dag = np.conj(SBlist[i].T)    #( a_i+2, i+1, a_i+1 )
        BB = np.tensordot(Blist[i], B_dag , axes=([1],[1]) ) # (a_i+1, a_i+2, a_i+2, a_i+1)
        C = np.tensordot(C, BB, axes=([1,2],[0,3]))
        C = np.transpose(C, (0,2,3,1))
        i = i+1
    return np.squeeze(C)


# def two_mps_overlap(psi_1, psi_2, L, d=2):
#     Blist = psi_1.Bs
#     SBlist = psi_2.Bs
#     print(Blist[2].shape, SBlist[2].shape)
#     B_dag = np.conj(SBlist[0].T)    #( a_i+1, a_i, i )
#     #print("here",B_dag.shape)
#     BB = np.tensordot(Blist[0], B_dag , axes=([0],[2]) ) # (a_i, a_i+1, a_i+1, a_i)
#     C = BB
#     #print(C.shape)
#     i = 1
#     while(i<L):
#         #print(f"i={i}")
#         B_dag = np.conj(SBlist[i].T)    #( a_i+2, a_i+1, i+1 )
#         BB = np.tensordot(Blist[i], B_dag , axes=([0],[2]) ) # (a_i+1, a_i+2, a_i+2, a_i+1)
#         C = np.tensordot(C, BB, axes=([1,2],[0,3]))
#         C = np.transpose(C, (0,2,3,1))
#         i = i+1
#     return np.squeeze(C)#*np.conj(C))**0.5


# def density_matrix_dvv_style(psi, L, s, l_m, l_A, l_B):
#     """Create the density matrix for a specific region of the MPS.
#     The tensors are in physical-virtual-virtua order

#     Args:
#         psi (_type_): _description_
#         L (_type_): _description_
#         s (_type_): _description_
#         l_m (_type_): _description_
#         l_A (_type_): _description_
#         l_B (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     Blis = psi.Bs
#     Ss = psi.Ss
    
#     if s != 0: 
#        D = np.tensordot(Blis[0] , np.conj(Blis[0]) , axes =([0,1],[0,1]))     #(a_i+1, b_i+1)
#        region_A = np.arange(s, s+l_A)
#        print("here 1")
#     else:
#          D = np.tensordot(Blis[0] , np.conj(Blis[0]) , axes =([1],[1]))      #(p_i, a_i+1, pp_i, b_i+1)
#          k = range(len(D.shape))
#          k[-3],k[-2] = k[-2],k[-3]
#          D = np.transpose(D, k)                                            #(p_i, pp_i, a_i+1, b_i+1)
#          region_A = np.arange(s+1, s+l_A)

#     region_l = np.arange(1,s)
#     region_m = np.arange(s+l_A, s+l_A+l_m)
#     region_B = np.arange(s+l_A+l_m, s+l_A+l_m+l_B)
#     region_r = np.arange(s+l_A+l_m+l_B, L)
#     print("here")
#     for i in region_l:
#            D = np.tensordot(D, Blis[i] , axes=([-2],[1]))                   #(p_0, pp_0, b_1, p_l, a_l+1)
#            D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3,-2],[1,0]))     #(p_0, pp_0, a_l+1, b_l+1)                                      
#     for i in region_A:
#              D = np.tensordot(D, Blis[i], axes = ([-2],[1]))                #(p_0, pp_0, b_l+1, p_k, a_k+1)
#              D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3],[1]))        #(p_0, pp_0, p_k, a_k+1, pp_k, b_k+1)
#              k = range(len(D.shape))
#              k[-3],k[-2] = k[-2],k[-3]
#              D = np.transpose(D, k)
#     for i in region_m:
#            D = np.tensordot(D, Blis[i] , axes=([-2],[1]))    #(a_i+1, p_i+1, a_i+2)
#            D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3,-2],[1,0]))   #(a_i+1, a_i+2)
#     for i in region_B:
#        if i != L-1:
#              D = np.tensordot(D, Blis[i], axes = ([-2],[1]))                #(p_0, pp_0, b_l+1, p_k, a_k+1)
#              D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3],[1]))        #(p_0, pp_0, p_k, a_k+1, pp_k, b_k+1)
#              k = range(len(D.shape))
#              k[-3],k[-2] = k[-2],k[-3]
#              D = np.transpose(D, k)
#        else:
#            D = np.tensordot(D, Blis[i] , axes=([-2],[1]))                  #(b_i+1, p_i+1, a_i+2)
#            D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3,-1],[1,2]))   #(a_i+1, a_i+2)

#     for i in region_r:
#         #print i
#         if i != L-1:
#            D = np.tensordot(D, Blis[i] , axes=([-2],[1]))    #(a_i+1, p_i+1, a_i+2)
#            D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3,-2],[1,0]))   #(a_i+1, a_i+2)           
#         else:
#            D = np.tensordot(D, Blis[i] , axes=([-2],[1]))                  #(b_i+1, p_i+1, a_i+2)
#            D = np.tensordot(D, np.conj(Blis[i]), axes= ([-3,-2,-1],[1,0,2]))   #(a_i+1, a_i+2)

#     #print D.shape
#     return D


def swap_last_axes(tensor, a=-3, b=-2):
    perm = list(range(tensor.ndim))
    perm[a], perm[b] = perm[b], perm[a]
    return np.transpose(tensor, perm)

def density_matrix_dvv_style(psi, L, s, l_m, l_A, l_B):
    """Create the density matrix for a specific region of the MPS.
    The tensors are in physical-virtual-virtual (d-v-v) order.

    Args:
        psi: MPS object with attributes Bs (tensors) and Ss (Schmidt values)
        L: Total number of MPS sites
        s: Start site for region A
        l_m: Length of middle region (between A and B)
        l_A: Length of region A
        l_B: Length of region B

    Returns:
        numpy.ndarray: Reduced density matrix for the region A
    """
    Blis = psi.Bs
    Ss = psi.Ss

    if s != 0: 
        D = np.tensordot(Blis[0], np.conj(Blis[0]), axes=([0, 1], [0, 1]))  # (a_i+1, b_i+1)
        region_A = np.arange(s, s + l_A)
    else:
        D = np.tensordot(Blis[0], np.conj(Blis[0]), axes=([1], [1]))        # (p_i, a_i+1, pp_i, b_i+1)
        D = swap_last_axes(D)                                              # (p_i, pp_i, a_i+1, b_i+1)
        region_A = np.arange(s + 1, s + l_A)

    region_l = np.arange(1, s)
    region_m = np.arange(s + l_A, s + l_A + l_m)
    region_B = np.arange(s + l_A + l_m, s + l_A + l_m + l_B)
    region_r = np.arange(s + l_A + l_m + l_B, L)

    for i in region_l:
        D = np.tensordot(D, Blis[i], axes=([-2], [1]))
        D = np.tensordot(D, np.conj(Blis[i]), axes=([-3, -2], [1, 0]))

    for i in region_A:
        D = np.tensordot(D, Blis[i], axes=([-2], [1]))
        D = np.tensordot(D, np.conj(Blis[i]), axes=([-3], [1]))
        D = swap_last_axes(D)

    for i in region_m:
        D = np.tensordot(D, Blis[i], axes=([-2], [1]))
        D = np.tensordot(D, np.conj(Blis[i]), axes=([-3, -2], [1, 0]))

    for i in region_B:
        if i != L - 1:
            D = np.tensordot(D, Blis[i], axes=([-2], [1]))
            D = np.tensordot(D, np.conj(Blis[i]), axes=([-3], [1]))
            D = swap_last_axes(D)
        else:
            D = np.tensordot(D, Blis[i], axes=([-2], [1]))
            D = np.tensordot(D, np.conj(Blis[i]), axes=([-3, -1], [1, 2]))

    for i in region_r:
        if i != L - 1:
            D = np.tensordot(D, Blis[i], axes=([-2], [1]))
            D = np.tensordot(D, np.conj(Blis[i]), axes=([-3, -2], [1, 0]))
        else:
            D = np.tensordot(D, Blis[i], axes=([-2], [1]))
            D = np.tensordot(D, np.conj(Blis[i]), axes=([-3, -2, -1], [1, 0, 2]))

    return D


def transpose_list(ll, partial = False):
    k_list = [];
    if partial==0:
       for i in range(ll):
           k_list.append(2*i)
       for i in range(ll):
           k_list.append(2*i+1)
    else:
        for i in range(int(ll/2)):
            k_list.append(i)
        for i in range(int(ll/2)):
            k_list.append(i + int((3*ll)/2))
        for i in range(int(ll/2)):
            k_list.append(i + ll)
        for i in range(int(ll/2)):
            k_list.append(i+int(ll/2))
    return k_list


from typing import Dict, List, Tuple

# ---------- Pauli basis (d=2) ----------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# stacked single-qubit operators for magic estimator
ops = np.stack([I, X, Y, Z], axis=0)  # (4,2,2)


def site_operator_transfer_einsum(B: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """
    Return E[k, l, L, r, R] = sum_{s,t} B[l,s,r] * ops[k,s,t] * conj(B[L,t,R]).
    Shapes:
      B   : (chi_l, d, chi_r)
      ops : (K, d, d)
      E   : (K, chi_l, chi_l, chi_r, chi_r)
    """
    E = np.einsum("kst,asb,AtB->kaAbB", ops, B, np.conjugate(B), optimize=True)
    return E


def site_operator_transfer_tdot(B, ops):
    """
    E[k, l, L, r, R] = sum_{s,t} B[l,s,r] * ops[k,s,t] * conj(B[L,t,R])
    Shapes assumed:
      B   : (chi_l, d, chi_r)
      ops : (K, d, d)
    Returns:
      E   : (K, chi_l, chi_l, chi_r, chi_r)
    """
    S = np.tensordot(ops, B, axes=([1], [1]))      # (K, d, chi_l, chi_r)
    S = np.transpose(S, (0, 2, 1, 3))              # (K, chi_l, d, chi_r)
    E = np.tensordot(S, np.conjugate(B), axes=([2], [1]))  # (K, chi_l, chi_r, chi_l, chi_r)
    return np.transpose(E, (0, 1, 3, 2, 4))        # (K, chi_l, chi_l, chi_r, chi_r)


def operator_transfer_list(Bs: List[np.ndarray], ops: np.ndarray, *, use="einsum") -> List[np.ndarray]:
    """
    Build the operator-transfer tensor for every site in the MPS.
    Returns a list [E^(0), E^(1), ..., E^(L-1)], each shaped (K, chi_l, chi_l, chi_r, chi_r).
    """
    fn = site_operator_transfer_einsum if use == "einsum" else site_operator_transfer_tdot
    return [fn(B, ops) for B in Bs]


def _herm(X: np.ndarray) -> np.ndarray:
    # Hermitize last two indices (χ×χ)
    return 0.5 * (X + X.swapaxes(-1, -2).conj())


def _safe_probs(p: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """Normalize a length-4 nonnegative vector to a proper PDF (sum=1) robustly."""
    p = np.real_if_close(p)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= tol:
        p = np.full_like(p, 0.25, dtype=float)
    else:
        p = (p / s).astype(float)
    # force exact sum==1 by adjusting argmax
    k = int(np.argmax(p))
    p[k] += 1.0 - float(p.sum())
    return p


def _left_contract_env_with_transfer(Lenv: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Lenv: (χL, χL) Hermitian environment
    E   : (4, χL, χL, χR, χR) per-site operator-transfer
    returns S[k] = Σ_{ℓ,ℓ'} Lenv[ℓ,ℓ'] E[k, ℓ,ℓ', :, :] ∈ C^{χR×χR}, k=0..3
    """
    S = np.tensordot(Lenv, E, axes=([0, 1], [1, 2]))  # (4, χR, χR)
    return _herm(S)


def _update_env(Lenv: np.ndarray, S_k: np.ndarray, prob_k: float) -> np.ndarray:
    """L ← S_k / sqrt(2 * π), keeping L Hermitian."""
    denom = np.sqrt(max(2.0 * prob_k, 1e-300))
    L_new = S_k / denom
    return _herm(L_new)


def _sample_one_chain(Bs: List[np.ndarray],
                      E_list: List[np.ndarray],
                      rng: np.random.Generator) -> Tuple[float, List[int]]:
    """
    Do one sequential draw σ = (k_0,...,k_{L-1}) with sitewise probs
      π_i(k) = 0.5 * Tr(S_{i,k} S_{i,k}^†).
    Returns (Π(σ), digits), where Π(σ) = ∏_i π_i(k_i).
    """
    chiL0 = Bs[0].shape[0]
    Lenv = np.eye(chiL0, dtype=complex)
    Lenv = _herm(Lenv)

    Pi = 1.0
    digits = []
    for E in E_list:
        S = _left_contract_env_with_transfer(Lenv, E)                # (4, χR, χR)
        Pi_mat = np.tensordot(S, S.conj(), axes=([1, 2], [1, 2]))    # (4,4)
        probs_raw = 0.5 * np.real_if_close(np.diag(Pi_mat))          # (4,)
        probs = _safe_probs(probs_raw)

        k = int(rng.choice(4, p=probs))
        digits.append(k)

        Pi *= float(probs[k])
        Lenv = _update_env(Lenv, S[k], probs[k])

    return float(Pi), digits

def get_pauli_string(digits):
    """
    Maps a list of digits (0-3) to a single string of Pauli operators.
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z
    """
    # Mapping based on standard quantum operator indexing
    mapping = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    
    # join() combines the mapped characters into one string
    return "".join(mapping.get(d, '?') for d in digits)

# # Example usage:
# input_list = [0, 1, 3, 2]
# print(get_pauli_string(input_list))  # Output: "IXZY"

def estimate_magic_perfect_from_mps(
    Bs: List[np.ndarray],
    ops: np.ndarray,
    *,
    n_samples: int = 10_000,
    seed: int = 0
) -> Dict[str, float]:
    """
    Perfect-sampling estimator of stabilizer Rényi magic from an MPS.

    Bs: list of MPS cores B_i of shape (χ_{i-1}, d=2, χ_i), preferably right-canonical.
    ops: (4,2,2) stacked single-qubit Paulis in order [I,X,Y,Z].

    Returns M1, M2 in nats + standard errors and diagnostics.
    """
    E_list = operator_transfer_list(Bs, ops, use="tdot")  # each E_i: (4, χL, χL, χR, χR)

    rng = np.random.default_rng(seed)
    PIs = np.empty(n_samples, dtype=float)
    #digits_list = []
    samples = {}
    for t in range(n_samples):
        Pi, _digits = _sample_one_chain(Bs, E_list, rng)
        PIs[t] = Pi
        #digits_list.append(get_pauli_string(_digits))
        samples[get_pauli_string(_digits)] = Pi
    n = len(Bs)
    ln2n = n * np.log(2.0)
    eps = 1e-300

    mean_P = float(PIs.mean())
    M2 = -np.log(mean_P + eps) - ln2n

    logs = np.log(np.clip(PIs, eps, None))
    M1 = -float(logs.mean()) - ln2n

    m = len(PIs)
    se_M2 = np.sqrt(PIs.var(ddof=1) / m) / max(mean_P, eps)
    se_M1 = np.sqrt(logs.var(ddof=1) / m)

    return {
        "M1_nats": M1, "SE_M1_nats": float(se_M1),
        "M2_nats": M2, "SE_M2_nats": float(se_M2),
        "mean_sumPi2": mean_P, "N": m, "samples": samples,
    }



if __name__ == "__main__":
    print(transpose_list(5))