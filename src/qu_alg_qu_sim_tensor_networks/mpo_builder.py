import numpy as np
# qutip (and qutip.qip, which moved to the separate `qutip-qip` package in
# QuTiP v5) are only needed by the QFT / pointer helpers far below, not by the
# MPO / TDVP / bridge path. Import lazily so this module stays usable without
# qutip installed (e.g. a --no-deps Colab install).
try:
    import qutip as qt
except Exception:  # pragma: no cover - optional dependency (QFT/pointer helpers only)
    qt = None
try:
    import qutip.qip.algorithms.qft as QFT
except Exception:  # pragma: no cover - optional dependency
    QFT = None


from qu_alg_qu_sim_tensor_networks import mps
from qu_alg_qu_sim_tensor_networks.tdvp import TDVPEngine
from qu_alg_qu_sim_tensor_networks import dmrg



class MPS:
    """Matrix product state with physical-first tensor convention (i, vL, vR).

    This MPS class uses the tensor leg ordering (physical, virtual_left, virtual_right),
    which differs from the `mps.MPS` class that uses (vL, i, vR). It is used
    internally by `mpo_builder` routines.

    Parameters
    ----------
    Bs : list of np.ndarray, optional
        Site tensors of shape (d, vL, vR).
    Ss : list of np.ndarray, optional
        Schmidt values at each bond.
    bonds : list of int, optional
        Bond dimensions; inferred from Bs if not provided.
    d : int, optional
        Physical dimension (default 2).
    bc : str, optional
        Boundary condition, 'finite' or 'infinite' (default 'finite').
    """
    def __init__(self, Bs=None, Ss=None, bonds=None, d=2, bc="finite"):
        self.L = None if Bs is None else len(Bs)
        self.d = d
        self.Bs = Bs
        self.Ss = Ss
        self.bonds = bonds
        
    @classmethod
    def initial_state_f(cls, L, d=2, dtype=complex):
        """Create a ferromagnetic product state: all sites in |0⟩.

        Parameters
        ----------
        L : int
            Number of sites.
        d : int, optional
            Physical dimension (default 2).
        dtype : dtype, optional
            Array data type (default complex).

        Returns
        -------
        MPS
            Product MPS with all spins in the first basis state |0⟩.
        """
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[np.mod(i, 1), 0, 0] = 1.0
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)
    
    
    @classmethod
    def initial_state_x_f(cls, L, d=2, dtype=complex):
        """Create a ferromagnetic product state polarized in the X direction: all sites in |+⟩.

        Parameters
        ----------
        L : int
            Number of sites.
        d : int, optional
            Physical dimension (default 2).
        dtype : dtype, optional
            Array data type (default complex).

        Returns
        -------
        MPS
            Product MPS with all spins in the equal superposition (1/√2)(|0⟩+|1⟩).
        """
        B_list = []
        s_list = []
        chi_vec = []
        for i in range(L):
            B = np.zeros((d, 1, 1), dtype=dtype)
            B[0, 0, 0] = 1.0 / np.sqrt(2.0)
            B[1, 0, 0] = 1.0 / np.sqrt(2.0)
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B_list[i].shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

    @classmethod
    def from_product_state(cls, product_state, L, d=2, dtype=complex):
        """Create a product MPS from a sequence of basis-state indices.

        Parameters
        ----------
        product_state : iterable of int
            Basis state index for each site (0-indexed).
        L : int
            Number of sites.
        d : int, optional
            Physical dimension (default 2).
        dtype : dtype, optional
            Array data type (default complex).

        Returns
        -------
        MPS
            Product MPS where site i is in the computational basis state |product_state[i]⟩.
        """
        B_list = []
        s_list = []
        chi_vec= []
        for k in product_state:
            B=np.zeros((d, 1, 1), dtype=dtype)
            #print(f"{k}")
            B[k,0,0] = 1
            s = np.zeros(1)
            s[0] = 1.0
            B_list.append(B)
            s_list.append(s)
            chi_vec.append(B.shape[1])
        s_list.append(s)
        chi_vec.append(1)
        return cls(B_list, s_list, chi_vec)

class MPO:
    """
    Create matrix product density operator (MPO)
    """

    def __init__(self, Ms=None, Ss=None, bonds=None, d=2, bc="finite"):
        self.L = None if Ms is None else len(Ms)
        self.d = d
        self.Ms = Ms
        self.Ss = Ss
        self.bc = bc
        if bonds:
            self.bonds=bonds 
        else: 
            self.update_bonds()
        self.init_H_mpo()
        
        
    def init_H_mpo(self):
        """Set H_mpo as an alias for the site tensor list Ms (used by DMRG/TDVP engines)."""
        self.H_mpo=self.Ms


    def update_bonds(self):
        """Infer and set bond dimensions from the MPO site tensors Ms."""
        bonds=[1]
        for w in self.Ms:
            #print(w.shape)
            bonds.append(w.shape[-1])
        self.bonds=bonds

        
        
def right_normalization_mpo(rho, truncation=True, epsilon=1e-7, chi_max=200, d=2):
    """Checks if a tensor is right-normalized. For MPS only. It could be expanded for MPO.

    Args:
        M_list (_type_): _description_
        chi_vec (_type_): _description_
        d (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    Blist = []
    L = len(rho.Ms)
  
    discarded=0
    for i in range(L):
        
        idx = L - 1 - i
        # shapes
        w0 = rho.Ms[idx].shape[2]
        s = rho.Ms[idx].shape[0]
        t = rho.Ms[idx].shape[1]
        w1 = rho.Ms[idx].shape[3]

        # reshape and transpose
        B_tmp = np.reshape(np.transpose(rho.Ms[idx], (2, 0, 1, 3)), (w0, s * t * w1))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        if truncation:
            tmp = np.min([np.sum(Y > epsilon), chi_max])
        else:
            tmp = len(Y)
    
        # for next B
        X = X[:, : tmp]
        Y = Y[: tmp]
        
        norm = np.linalg.norm(Y)
        #print("Y",Y)
        discarded += np.sum(Y[chi_max :] ** 2) / sum(Y ** 2)
        Z = Z[:tmp, :]
        new_ind = int(Z.shape[1] / (d * d))
        B_new = np.reshape(Z, (tmp, d, d, new_ind))
        
        # add blist
        Blist.append(np.transpose(B_new, (1, 2, 0, 3)))

    
        #Ss.append(Y/norm)
        # XY dot
        Umat = np.dot(X, np.diag(Y))
        
        rho.Ms[idx - 1] = np.tensordot(rho.Ms[idx - 1], Umat, axes=(3, 0))
        #if idx ==0:
            #print(Umat)
    #normalized_rho = MPO(Blist[::-1], rho.Ss, rho.bonds)
    #Ss.append(np.array([1.0]))
    Blist[-1]=Blist[-1]/Umat.squeeze()
    return Blist[::-1], discarded#, Ss[::-1]#, normalized_rho




def left_normalization_mpo(rho, truncation=True, epsilon=1e-7, chi_max=200, d=2):
    """Left-normalize an MPO by sweeping left-to-right with SVD.

    Parameters
    ----------
    rho : MPO
        Input MPO with tensors of shape (d, d, vL, vR). Modified in-place.
    truncation : bool, optional
        If True (default), truncate singular values below ``epsilon`` up to ``chi_max``.
    epsilon : float, optional
        Singular value threshold (default 1e-7).
    chi_max : int, optional
        Maximum bond dimension to retain (default 200).
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    Alist : list of np.ndarray
        Left-canonical MPO tensors.
    norm_mpo : np.ndarray
        Residual norm matrix at the right boundary.
    """
    Alist = []
    L = len(rho.Ms)
    #print(f"L={L}; left normalization")
    
    for idx in range(L):
        s = rho.Ms[idx].shape[0]
        t = rho.Ms[idx].shape[1]
        w0 = rho.Ms[idx].shape[2]
        #print(s,t,w0)
        gamma   = s*t*w0
        alpha = rho.Ms[idx].shape[3]
        B_tmp = np.reshape(rho.Ms[idx], (gamma, alpha))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        if truncation:
            tmp = np.min([np.sum(Y > epsilon), chi_max])
        else:
            tmp = len(Y)
            
        #print(tmp, X.shape)
        A_new = np.reshape(X[:, :tmp], (s, t, w0, tmp))
        Y = Y[:tmp]
        Z = Z[:tmp, :]
        T = np.dot(np.diag(Y), Z)
        #print(T, "\n-----------")
        Alist.append(A_new)
        #print(idx)
        if idx < L-1:
            rho.Ms[idx+1] = np.transpose(np.tensordot(T, rho.Ms[idx+1], axes=(1, 2)), (1,2,0,3))
        else:
            #print(T, idx, A_new.shape)
            norm_mpo = T.squeeze()
        
    
    return Alist, norm_mpo



def left_isometric_mpo_ws(ws_L, truncation=True, epsilon=1e-7, chi_max=200, d=2):
    """Left-isometrize a raw list of MPO tensors (not an MPO object).

    Parameters
    ----------
    ws_L : list of np.ndarray
        MPO site tensors, each of shape (d, d, vL, vR).
    truncation : bool, optional
        If True (default), truncate singular values below ``epsilon`` up to ``chi_max``.
    epsilon : float, optional
        Singular value threshold (default 1e-7).
    chi_max : int, optional
        Maximum bond dimension to retain (default 200).
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    Alist : list of np.ndarray
        Left-isometric MPO tensors.
    T_l : np.ndarray
        Leftover transfer matrix at the right boundary (to be absorbed into the bridge tensor).
    """
    Alist = []
    Blist = []
    L = len(ws_L)
    for idx in range(L):
        s = ws_L[idx].shape[0]
        t = ws_L[idx].shape[1]
        w0 = ws_L[idx].shape[2]
        #print(s,t,w0)
        gamma   = s*t*w0
        alpha = ws_L[idx].shape[3]
        B_tmp = np.reshape(ws_L[idx], (gamma, alpha))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        if truncation:
            tmp = np.min([np.sum(Y > epsilon), chi_max])
        else:
            tmp = len(Y)
            
        #print(tmp, X.shape)
        A_new = np.reshape(X[:, :tmp], (s, t, w0, tmp))
        Y = Y[:tmp]
        Z = Z[:tmp, :]
        T = np.dot(np.diag(Y), Z)
        #print(T, "\n-----------")
        Alist.append(A_new)
    T_l = T
    return Alist, T_l



def right_isometric_mpo_ws(ws_R, truncation=True, epsilon=1e-7, chi_max=200, d=2):
    """Right-isometrize a raw list of MPO tensors (not an MPO object).

    Parameters
    ----------
    ws_R : list of np.ndarray
        MPO site tensors, each of shape (d, d, vL, vR).
    truncation : bool, optional
        If True (default), truncate singular values below ``epsilon`` up to ``chi_max``.
    epsilon : float, optional
        Singular value threshold (default 1e-7).
    chi_max : int, optional
        Maximum bond dimension to retain (default 200).
    d : int, optional
        Physical dimension (default 2).

    Returns
    -------
    Blist : list of np.ndarray
        Right-isometric MPO tensors (in reverse order, use Blist[::-1]).
    T_r : np.ndarray
        Leftover transfer matrix at the left boundary.
    """
    Blist = []
    L = len(ws_R)
    discarded = 0
    for i in range(L):
        
        idx = L - 1 - i
        # shapes
        w0 = ws_R[idx].shape[2]
        s = ws_R[idx].shape[0]
        t = ws_R[idx].shape[1]
        w1 = ws_R[idx].shape[3]

        # reshape and transpose
        B_tmp = np.reshape(np.transpose(ws_R[idx], (2, 0, 1, 3)), (w0, s * t * w1))
        X, Y, Z = np.linalg.svd(B_tmp, full_matrices=True, compute_uv=True)
        if truncation:
            tmp = np.min([np.sum(Y > epsilon), chi_max])
        else:
            tmp = len(Y)
    
        # for next B
        X = X[:, : tmp]
        Y = Y[: tmp]
        
        norm = np.linalg.norm(Y)
        
        discarded += np.sum(Y[chi_max :] ** 2) / sum(Y ** 2)
        Z = Z[:tmp, :]
        new_ind = int(Z.shape[1] / (d * d))
        B_new = np.reshape(Z, (tmp, d, d, new_ind))
        
        # add blist
        Blist.append(np.transpose(B_new, (1, 2, 0, 3)))
        # XY dot
        Umat = np.dot(X, np.diag(Y))
        
        #ws_R[idx - 1] = np.tensordot(ws_R[idx - 1], Umat, axes=(3, 0))
    T_r = Umat
    return Blist, T_r
    
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def _choose_rank(s: np.ndarray, truncation: bool, epsilon: float, chi_max: int) -> int:
    """Pick kept rank given singular values."""
    if not truncation:
        return len(s)
    r = int(np.sum(s > epsilon))
    r = min(max(r, 1), chi_max)  # keep at least 1
    return r


def left_canonicalize_mpo_half(
    ws: List[np.ndarray],
    truncation: bool = True,
    epsilon: float = 1e-7,
    chi_max: int = 200,
    *,
    check: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Left-canonicalize an MPO half-chain by sweeping left -> right.

    Input:
        ws[i]: MPO tensor of shape (d, d, a_l, a_r).

    Output:
        ws_left: list of left-canonical tensors with shapes (d, d, a_l', a_r')
                 such that each tensor, viewed as a matrix
                     W_mat = reshape(W, (d*d*a_l', a_r'))
                 satisfies (W_mat)^† W_mat = I_{a_r'}  (columns orthonormal).

        T_l: leftover "transfer" matrix on the *right boundary* (the cut):
             shape (a_r_last_new, a_r_last_old_of_input_chain).
             If you want a completely canonical chain you would absorb T_l into
             whatever comes next (e.g., the bridge / middle tensor).

        info: dict with diagnostics (kept dims, discarded weight estimate, etc.)

    Notes:
        - This is the standard gauge propagation: at each step we absorb the
          incoming transfer matrix M into the *left bond* of the next tensor.
        - For a half-chain ending at the cut, you usually want to KEEP T_l.
    """
    if len(ws) == 0:
        raise ValueError("ws is empty")

    ws_left: List[np.ndarray] = []
    kept_ranks = []
    discarded_weights = []

    d0, d1, aL0, _ = ws[0].shape
    if d0 != d1:
        raise ValueError(f"Expected (d,d,...) physical dims but got ({d0},{d1},...)")
    d = d0

    # Incoming transfer matrix acting on the left bond.
    # Shape (a_l_in, a_l_current). Initially identity.
    M = np.eye(aL0, dtype=ws[0].dtype)

    for i, W in enumerate(ws):
        d0, d1, a_l, a_r = W.shape
        if d0 != d or d1 != d:
            raise ValueError(f"Site {i}: inconsistent physical dim (got {d0},{d1}, expected {d},{d})")
        if M.shape[1] != a_l:
            raise ValueError(f"Site {i}: incoming M has shape {M.shape} but tensor left bond is {a_l}")

        # Absorb gauge into left bond: (a_l_in,a_l) x (d,d,a_l,a_r) -> (d,d,a_l_in,a_r)
        W_eff = np.tensordot(M, W, axes=(1, 2))            # (a_l_in, d, d, a_r)
        W_eff = np.transpose(W_eff, (1, 2, 0, 3))          # (d, d, a_l_in, a_r)
        a_l_in = W_eff.shape[2]

        # Reshape to matrix with left index grouped: (d*d*a_l_in, a_r)
        A = np.reshape(W_eff, (d * d * a_l_in, a_r))

        # SVD: A = U S Vh
        U, S, Vh = np.linalg.svd(A, full_matrices=False)

        r = _choose_rank(S, truncation, epsilon, chi_max)
        kept_ranks.append(r)

        # Discarded weight estimate (Frobenius norm fraction)
        if truncation and r < len(S):
            disc = float(np.sum(S[r:] ** 2) / np.sum(S ** 2))
        else:
            disc = 0.0
        discarded_weights.append(disc)

        U_r = U[:, :r]                          # (d*d*a_l_in, r)  columns orthonormal
        S_r = S[:r]                             # (r,)
        Vh_r = Vh[:r, :]                        # (r, a_r)

        # Left-canonical tensor
        W_left = np.reshape(U_r, (d, d, a_l_in, r))   # (d,d,a_l_in,r)
        ws_left.append(W_left)

        # Transfer to next site on its left bond: M := (S Vh)
        M = (S_r[:, None] * Vh_r)               # (r, a_r)

        if check:
            # Check left isometry: U_r^† U_r = I_r
            G = U_r.conj().T @ U_r
            err = np.linalg.norm(G - np.eye(r), ord="fro")
            if err > 1e-8:
                raise RuntimeError(f"Left isometry check failed at site {i}: ||U^†U-I||_F={err:e}")

    T_l = M  # leftover at the cut

    info = dict(
        kept_ranks=kept_ranks,
        discarded_weights=discarded_weights,
        discarded_weight_total=float(np.sum(discarded_weights)),
    )
    return ws_left, T_l, info


def right_canonicalize_mpo_half(
    ws: List[np.ndarray],
    truncation: bool = True,
    epsilon: float = 1e-7,
    chi_max: int = 200,
    *,
    check: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Right-canonicalize an MPO half-chain by sweeping right -> left.

    Input:
        ws[i]: MPO tensor of shape (d, d, a_l, a_r).

    Output:
        ws_right: list of right-canonical tensors in the SAME order as input.
                  Each tensor, viewed as a matrix
                      W_mat = reshape(W, (a_l', d*d*a_r'))
                  satisfies W_mat W_mat^† = I_{a_l'}  (rows orthonormal).

        T_r: leftover "transfer" matrix on the *left boundary* (the cut):
             shape (a_l_old_of_input_chain, a_l_new_at_cut).
             This is what you would contract with the left-half leftover to
             form a bridge matrix at the cut.

        info: dict with diagnostics.

    Notes:
        - Gauge propagation: at each step we absorb incoming M into the *right bond*
          of the current tensor, then SVD, then pass U S leftwards.
        - For a half-chain starting at the cut, you usually want to KEEP T_r.
    """
    if len(ws) == 0:
        raise ValueError("ws is empty")

    d0, d1, _, aR_last = ws[-1].shape
    if d0 != d1:
        raise ValueError(f"Expected (d,d,...) physical dims but got ({d0},{d1},...)")
    d = d0

    ws_right_rev: List[np.ndarray] = []
    kept_ranks = []
    discarded_weights = []

    # Incoming transfer matrix acting on the right bond.
    # Shape (a_r_current, a_r_out). Initially identity.
    M = np.eye(aR_last, dtype=ws[-1].dtype)

    for step, idx in enumerate(range(len(ws) - 1, -1, -1)):
        W = ws[idx]
        d0, d1, a_l, a_r = W.shape
        if d0 != d or d1 != d:
            raise ValueError(f"Site {idx}: inconsistent physical dim (got {d0},{d1}, expected {d},{d})")
        if M.shape[0] != a_r:
            raise ValueError(f"Site {idx}: incoming M has shape {M.shape} but tensor right bond is {a_r}")

        # Absorb gauge into right bond: (d,d,a_l,a_r) x (a_r,a_r_out) -> (d,d,a_l,a_r_out)
        W_eff = np.tensordot(W, M, axes=(3, 0))          # (d,d,a_l,a_r_out)
        a_r_out = W_eff.shape[3]

        # Reshape to matrix with right indices grouped: (a_l, d*d*a_r_out)
        B = np.reshape(W_eff, (a_l, d * d * a_r_out))

        # SVD: B = U S Vh
        U, S, Vh = np.linalg.svd(B, full_matrices=False)

        r = _choose_rank(S, truncation, epsilon, chi_max)
        kept_ranks.append(r)

        if truncation and r < len(S):
            disc = float(np.sum(S[r:] ** 2) / np.sum(S ** 2))
        else:
            disc = 0.0
        discarded_weights.append(disc)

        U_r = U[:, :r]                   # (a_l, r)
        S_r = S[:r]                      # (r,)
        Vh_r = Vh[:r, :]                 # (r, d*d*a_r_out)  rows orthonormal

        # Right-canonical tensor from Vh_r
        W_right = np.reshape(Vh_r, (r, d, d, a_r_out))   # (r,d,d,a_r_out)
        W_right = np.transpose(W_right, (1, 2, 0, 3))    # (d,d,r,a_r_out)
        ws_right_rev.append(W_right)

        # Transfer to previous site on its right bond: M := (U S)
        M = U_r * S_r[None, :]           # (a_l, r)

        if check:
            # Check right isometry: Vh_r Vh_r^† = I_r
            G = Vh_r @ Vh_r.conj().T
            err = np.linalg.norm(G - np.eye(r), ord="fro")
            if err > 1e-8:
                raise RuntimeError(f"Right isometry check failed at site {idx}: ||VV^†-I||_F={err:e}")

    # Reverse back to original order
    ws_right = list(reversed(ws_right_rev))
    T_r = M  # leftover at the cut (left boundary)

    info = dict(
        kept_ranks=list(reversed(kept_ranks)),  # align with site order if you like
        discarded_weights=list(reversed(discarded_weights)),
        discarded_weight_total=float(np.sum(discarded_weights)),
    )
    return ws_right, T_r, info

    


def compress_mpo(wmpo, epsilon=1e-9, chi_max=200):
    Ms, norm_mpo = left_normalization_mpo(wmpo, truncation=False, epsilon=epsilon, chi_max=chi_max)
    wmpo = MPO(Ms, Ss=None, bonds=None)
    Ms, discarded = right_normalization_mpo(wmpo, truncation=True, epsilon=epsilon, chi_max=chi_max) 
    wmpo = MPO(Ms, Ss=None, bonds=None)
    return wmpo, norm_mpo, discarded










def spin_operator(N):
    # pre-allocate operators
    s0 = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    s0_list = []
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(s0)

        op_list[n] = s0
        s0_list.append(qt.tensor(op_list))
        
        
        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))
            
        op_list[n] = sy
        sy_list.append(qt.tensor(op_list))
            
        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))
            
            
    return s0_list, sx_list, sy_list, sz_list


def pointer_operator(N):
    # pre-allocate operators
    si = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    p = (si - sz)/2


    p_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = p
        p_list.append((2**(-n-1))*qt.tensor(op_list))
             
    return p_list




def get_P_x(psi, num_total_sites, num_pointers):
    qft = QFT(num_pointers)
    dms = qft.dims
    shape = qft.shape
    rho = qt.Qobj(reduced_density_matrix_pointer(psi, num_total_sites, num_pointers), dims=dms)#, shape=shape)
    rho_com = qft.dag()*rho*qft
    #print(f"t={t:.2f}")
    Pxs = []
    for x in range(2**num_pointers):
        Px_op = projector_p_x(x, num_pointers) 
        
        px = Px_op.dag()*rho_com*Px_op
        #print("here",px)
        #Pxs.append(np.trace(px).real)  #old qutip version
        Pxs.append(px.real)
        #print(f"    P={np.trace(px).real:.3f}, x={x}")
    #print("==="*15)
    return Pxs
    
    
    
    
def reduced_density_matrix_pointer(Psi, n_sys, n_ps, as_matrix = True):
    """Compute the reduce density matrix of pointers only

    Args:
        Psi (_type_): _description_
        n_sys (_type_): _description_
        n_ps (_type_): _description_
        as_matrix (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    Ms = Psi.Bs
    rho = np.tensordot(Ms[0], np.conj(Ms[0]), axes=([0,1],[0,1]))  #(a_i, a_i*)
    for i in range(1, n_sys):
        #print(rho.shape)
        rho = np.tensordot(rho, Ms[i], axes=([-2],[0]))  #(a_i*, s_i+1, a_i+2)
        rho = np.tensordot(rho, np.conj(Ms[i]), axes=([-3, 1],[0, 1]))  #(a_i, a_i*)
    #print("1s", rho.shape)
    for j in range(n_sys, n_sys+n_ps):
        #print("a",j, rho.shape,  Ms[j].shape)
        rho = np.tensordot(rho, Ms[j], axes=([-2],[0]))  #(a_i*, s_i+1, a_i+2)
        rho = np.tensordot(rho, np.conj(Ms[j]), axes=([-3],[0])) #( s_i+1, a_i+2, s'_i+1, a*_i+2)
        sh = list(range(len(rho.shape)))
        #print(sh)
        #print("b",j, rho.shape)
        sh[-2], sh[-3] = sh[-3], sh[-2]
        rho = np.transpose(rho, sh)

    sh = list(range(len(rho.shape)))
    sh[-2], sh[-3] = sh[-2], sh[-3]
    rho = np.transpose(rho, sh)
    rho = np.squeeze(rho)
    #print(rho.shape)
    ordering = tuple(range(0, 2*(n_ps),2))+tuple(range(1, 2*(n_ps)+1,2))
    
    #print(ordering, len(rho.shape))
    if as_matrix:
        rho = np.transpose(rho, ordering)
        rho = np.reshape(rho, (2**n_ps, 2**n_ps))
    return rho


loc_sz = lambda x : qt.basis(2, 0) if (x == '0') else qt.basis(2, 1)

Z = lambda z, n_pointer: qt.tensor([loc_sz(i) for i in get_bin(z).zfill(n_pointer)])

get_bin = lambda x: format(x, 'b')



def projector_p_x(xp, n_pointer, d=2):
    """
    x: pointer state in decimal rep.
    """
    config = get_bin(xp).zfill(n_pointer)
    state = qt.tensor([loc_sz(i) for i in config])
    P_x = state#*state.dag()
    return P_x    





def form_initial_pointer_state(num_pointers):
    up_state = qt.tensor([qt.basis(2,0) for i in range(num_pointers)])
    qft = QFT.qft(num_pointers)
    zero_state_p = qft*up_state
    zero_state_p = np.array(zero_state_p)
    zero_state_p = list(zero_state_p.squeeze())
    psi_p = ptn.MPS.from_vector(2, num_pointers, zero_state_p)
    Bs_p = psi_p.A

    Bs_pm = []
    for B in Bs_p:
        Bs_pm.append(np.transpose(B, (1,0,2)))
    Ss_p = [np.array([1.]),] *num_pointers
    return Bs_pm, Ss_p


def system_pointer_psi(psi_sys, Bsp, Ssp):
    Ss_sys = psi_sys.Ss
    Ss_p = Ssp
    Ss = Ss_sys + Ss_p
    Bs = psi_sys.Bs+Bsp
    bonds = [1]
    for b in Bs:
        bonds.append(b.shape[-1])
    #print(bonds, [b.shape for b in Bs ])
    #rBs = []
    #for b in Bs:
    #    rBs.append(np.transpose(b, (1,0,2)))
    #rBs = qmpo.right_normalization_mps(rBs, bonds)
    #Bs=[]
    #for b in rBs:
     #   Bs.append(np.transpose(b, (1,0,2)))
    
    return mps.MPS(Bs, Ss, bc="finite")


def replace_pointers(psi_sys, Bsp, Ssp, N_sys, N_ps):
    Ss_sys = psi_sys.Ss[0:N_sys]
    Ss_p = Ssp
    Ss = Ss_sys + Ss_p
    Bs = psi_sys.Bs[:N_sys]+Bsp
    bonds = [1]
    for b in Bs:
        bonds.append(b.shape[-1])
    #print(bonds, [b.shape for b in Bs ])
    #rBs = []
    #for b in Bs:
    #    rBs.append(np.transpose(b, (1,0,2)))
    #rBs = qmpo.right_normalization_mps(rBs, bonds)
    #Bs=[]
    #for b in rBs:
     #   Bs.append(np.transpose(b, (1,0,2)))
    
    return MPS(Bs, Ss)




from qu_alg_qu_sim_tensor_networks import mpo_builder as mpo
import numpy as np
import numpy as np
from typing import List, Optional


import numpy as np
from typing import List, Optional


def mpo_dagger_tensors(Ms: List[np.ndarray]) -> List[np.ndarray]:
    """
    MPO dagger for tensors with convention (out, in, a_l, a_r).
    Returns tensors for G† with same convention (out, in, a_l, a_r):
        (G†)_{out,in} = conj(G_{in,out})
    """
    return [np.conjugate(M).transpose(1, 0, 2, 3) for M in Ms]


def build_left_envs_Gdag_O_G_tensordot_left_half(
    mps_Bs: List[np.ndarray],
    G_Ms_left: List[np.ndarray],
    O_Ms_left: List[np.ndarray],
    cut: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Left environments for <psi| G† O G |psi> when G and O are given only on the LEFT HALF.

    Inputs:
      mps_Bs: full MPS list, length Ltot, B[i] shape (chiL, d, chiR)
      G_Ms_left: MPO tensors for sites [0..cut-1], length cut
      O_Ms_left: MPO tensors for sites [0..cut-1], length cut
      cut: global cut index; if None, inferred as len(G_Ms_left)

    Output:
      L_envs: list of length (Ltot+1).
        L_envs[0] = (1,1,1,1,1)
        L_envs[i] defined for i in [0..cut], each with open legs:
          (chi_bra_i, b'_i, a_i, b_i, chi_ket_i)
      For i > cut, entries are None.

    Notes:
      - This is the SAME contraction pattern as the "full-length" left builder,
        but it only iterates over the left-half sites and uses global indices directly.
      - If your G_Ms_left / O_Ms_left correspond to sites 0..cut-1, then local index = global index.
    """
    Ltot = len(mps_Bs)
    if cut is None:
        cut = len(G_Ms_left)

    Gd_left = mpo_dagger_tensors(G_Ms_left)
    dtype = np.result_type(mps_Bs[0].dtype, G_Ms_left[0].dtype, O_Ms_left[0].dtype, complex)

    L_envs = [None] * (Ltot + 1)
    env = np.ones((1, 1, 1, 1, 1), dtype=dtype)  # (lb, bpL, aL, bL, lk)
    L_envs[0] = env
    print("length",len(G_Ms_left))
    for i in range(0, cut):
        B  = mps_Bs[i]           # (lk, t, rk)
        Bc = np.conjugate(B)     # (lb, s, rb)

        G  = G_Ms_left[i]        # (v, t, bL, bR)
        O  = O_Ms_left[i]        # (u, v, aL, aR)
        Gd = Gd_left[i]          # (s, u, bpL, bpR)

        # env(lb, bpL, aL, bL, lk) --contract-- Bc(lb, s, rb) over lb
        tmp = np.tensordot(env, Bc, axes=(0, 0))
        # tmp: (bpL, aL, bL, lk, s, rb)

        # contract with Gd(s, u, bpL, bpR) over (bpL, s)
        tmp = np.tensordot(tmp, Gd, axes=([0, 4], [2, 0]))
        # tmp: (aL, bL, lk, rb, u, bpR)

        # contract with O(u, v, aL, aR) over (aL, u)
        tmp = np.tensordot(tmp, O, axes=([0, 4], [2, 0]))
        # tmp: (bL, lk, rb, bpR, v, aR)

        # contract with G(v, t, bL, bR) over (bL, v)
        tmp = np.tensordot(tmp, G, axes=([0, 4], [2, 0]))
        # tmp: (lk, rb, bpR, aR, t, bR)

        # contract with B(lk, t, rk) over (lk, t)
        tmp = np.tensordot(tmp, B, axes=([0, 4], [0, 1]))
        # tmp: (rb, bpR, aR, bR, rk)

        env = tmp
        L_envs[i + 1] = env

    return L_envs


def build_right_envs_Gdag_O_G_tensordot_right_half(
    mps_Bs: List[np.ndarray],
    G_Ms_right: List[np.ndarray],
    O_Ms_right: List[np.ndarray],
    cut: int,
) -> List[np.ndarray]:
    """
    Right environments for <psi| G† O G |psi> when G and O are given only on the RIGHT HALF.

    Inputs:
      mps_Bs: full MPS list, length Ltot, B[i] shape (chiL, d, chiR)
      G_Ms_right: MPO tensors for sites [cut..Ltot-1], length (Ltot-cut)
      O_Ms_right: MPO tensors for sites [cut..Ltot-1], length (Ltot-cut)
      cut: global cut index

    Output:
      R_envs: list of length (Ltot+1).
        R_envs[Ltot] = (1,1,1,1,1)
        R_envs[i] defined for i in [cut..Ltot], each with open legs:
          (chi_bra_i, b'_i, a_i, b_i, chi_ket_i)
      For i < cut, entries are None.
    """
    Ltot = len(mps_Bs)

    Gd_right = mpo_dagger_tensors(G_Ms_right)
    dtype = np.result_type(mps_Bs[0].dtype, G_Ms_right[0].dtype, O_Ms_right[0].dtype, complex)

    R_envs = [None] * (Ltot + 1)
    env = np.ones((1, 1, 1, 1, 1), dtype=dtype)  # (rb, bpR, aR, bR, rk)
    R_envs[Ltot] = env

    for i in range(Ltot - 1, cut - 1, -1):
        j = i - cut  # local right-half index

        B  = mps_Bs[i]           # (lk, t, rk)
        Bc = np.conjugate(B)     # (lb, s, rb)

        G  = G_Ms_right[j]       # (v, t, bL, bR)
        O  = O_Ms_right[j]       # (u, v, aL, aR)
        Gd = Gd_right[j]         # (s, u, bpL, bpR)

        # env(rb, bpR, aR, bR, rk) --contract-- B(lk, t, rk) over rk
        tmp = np.tensordot(env, B, axes=(4, 2))
        # tmp: (rb, bpR, aR, bR, lk, t)

        # contract with G(v, t, bL, bR) over (bR, t)
        tmp = np.tensordot(tmp, G, axes=([3, 5], [3, 1]))
        # tmp: (rb, bpR, aR, lk, v, bL)

        # contract with O(u, v, aL, aR) over (aR, v)
        tmp = np.tensordot(tmp, O, axes=([2, 4], [3, 1]))
        # tmp: (rb, bpR, lk, bL, u, aL)

        # contract with Gd(s, u, bpL, bpR) over (bpR, u)
        tmp = np.tensordot(tmp, Gd, axes=([1, 4], [3, 1]))
        # tmp: (rb, lk, bL, aL, s, bpL)

        # contract with Bc(lb, s, rb) over (rb, s)
        tmp = np.tensordot(Bc, tmp, axes=([2, 1], [0, 4]))
        # tmp: (lb, lk, bL, aL, bpL)

        # reorder to (lb, bpL, aL, bL, lk)
        env = np.transpose(tmp, (0, 4, 3, 2, 1))
        R_envs[i] = env

    return R_envs


import numpy as np
from typing import List, Optional


def mpo_dagger_tensors(Ms: List[np.ndarray]) -> List[np.ndarray]:
    """
    MPO dagger for tensors with convention (out, in, a_l, a_r).
    Returns tensors for G† with same convention (out, in, a_l, a_r):
        (G†)_{out,in} = conj(G_{in,out})
    """
    return [np.conjugate(M).transpose(1, 0, 2, 3) for M in Ms]


def build_left_envs_Gdag_O_G_einsum(
    mps_Bs: List[np.ndarray],
    G_Ms: List[np.ndarray],
    O_Ms: List[np.ndarray],
    cut: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Left environments for <psi| G† O G |psi> built with einsum.

    Conventions:
      MPS: B[i]  shape (χL, d, χR)
      MPO: W[i]  shape (d_out, d_in, aL, aR)

    Output:
      L_envs[i] is contraction of sites [0..i-1] leaving open:
        (χ_bra_i,  b'_i,  a_i,  b_i,  χ_ket_i)
      where b' is the virtual bond of G†, a is virtual bond of O, b is virtual bond of G.
    """
    L = len(mps_Bs)
    if cut is None:
        cut = L

    Gd_Ms = mpo_dagger_tensors(G_Ms)
    dtype = np.result_type(mps_Bs[0].dtype, G_Ms[0].dtype, O_Ms[0].dtype, complex)

    # env indices: (L, p, a, b, K)
    env = np.ones((1, 1, 1, 1, 1), dtype=dtype)
    L_envs = [env]

    for i in range(cut):
        B  = mps_Bs[i]           # (K, t, T)
        Bc = np.conjugate(B)     # (L, s, R)
        Gd = Gd_Ms[i]            # (s, u, p, P)
        O  = O_Ms[i]             # (u, v, a, A)
        G  = G_Ms[i]             # (v, t, b, B)

        # Result env: (R, P, A, B, T)
        env = np.einsum(
            "L p a b K,  L s R,  s u p P,  u v a A,  v t b B,  K t T  ->  R P A B T",
            env, Bc, Gd, O, G, B,
            optimize=True
        )
        L_envs.append(env)

    return L_envs


def build_right_envs_Gdag_O_G_einsum(
    mps_Bs: List[np.ndarray],
    G_Ms: List[np.ndarray],
    O_Ms: List[np.ndarray],
    cut: int,
) -> List[np.ndarray]:
    """
    Right environments for <psi| G† O G |psi> built with einsum.

    Output:
      R_envs[i] is contraction of sites [i..L-1] leaving open on the LEFT:
        (χ_bra_i,  b'_i,  a_i,  b_i,  χ_ket_i)
    """
    L = len(mps_Bs)

    Gd_Ms = mpo_dagger_tensors(G_Ms)
    dtype = np.result_type(mps_Bs[0].dtype, G_Ms[0].dtype, O_Ms[0].dtype, complex)

    R_envs = [None] * (L + 1)

    # env indices: (R, P, A, B, T) at the right boundary
    env = np.ones((1, 1, 1, 1, 1), dtype=dtype)
    R_envs[L] = env

    for i in range(L - 1, cut - 1, -1):
        B  = mps_Bs[i]           # (K, t, T)
        Bc = np.conjugate(B)     # (L, s, R)
        Gd = Gd_Ms[i]            # (s, u, p, P)
        O  = O_Ms[i]             # (u, v, a, A)
        G  = G_Ms[i]             # (v, t, b, B)

        # env: (R,P,A,B,T) -> new env: (L,p,a,b,K)
        env = np.einsum(
            "L s R,  s u p P,  u v a A,  v t b B,  K t T,  R P A B T  ->  L p a b K",
            Bc, Gd, O, G, B, env,
            optimize=True
        )
        R_envs[i] = env

    return R_envs

def squeeze_product_state_env(env5: np.ndarray) -> np.ndarray:
    """
    For product-state MPS (chi_bra=chi_ket=1), squeeze:
        (1, b', a, b, 1) -> (b', a, b)
    """
    return env5[0, :, :, :, 0]
