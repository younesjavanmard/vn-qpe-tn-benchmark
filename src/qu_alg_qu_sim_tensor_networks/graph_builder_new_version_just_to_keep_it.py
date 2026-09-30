# graph_builder.py  — rustworkx + parallelized make_Ws, same public API

from typing import List, Dict, Tuple
import os
import numpy as np
import networkx as nx
# Prefer rustworkx; fall back to networkx for plotting
try:
    import rustworkx as rx
    _HAVE_RX = True
except Exception:
    _HAVE_RX = False
    import networkx as nx  # type: ignore

from concurrent.futures import ThreadPoolExecutor

# Toggle internal parallelism for make_Ws (keeps function signature)
_PARALLEL_MAKE_WS = True
_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 0)

# ---------------------------------------
#  Pauli cache & utilities (unchanged API)
# ---------------------------------------

class Pauli:
    """
    Lightweight Pauli cache with O(1) lookup.
    """
    def __init__(self, ksi: float = 1.0):
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1],[1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0],[0, -1]], dtype=np.complex128)
        Sp = 0.5 * (X + 1j*Y)
        Sm = 0.5 * (X - 1j*Y)

        self.I, self.X, self.Y, self.Z = I, X, Y, Z
        self.Sp, self.Sm = Sp, Sm
        self._ksi = float(ksi)

        self._map = {
            "I": I, "X": X, "Y": Y, "Z": Z,
            "p": (I - Z)/2,          # Sm*Sp
            "q": (I + Z)/2,          # Sp*Sm
            "R": Sp,                 # raising
            "L": Sm,                 # lowering
            "S": (I - X)/2,          # projector
        }

    def get_pauli(self, symbol: str) -> np.ndarray:
        if symbol == "O":
            k = self._ksi
            return (0.5*(1.0/k + k))*self.I + (0.5*(-1.0/k + k))*self.Z + self.X
        try:
            return self._map[symbol]
        except KeyError as e:
            raise ValueError(f"Unknown single-site op '{symbol}'") from e


def strings_to_ops_convertor(string: str, N: int, s0, sx, sy, sz, sp, sm):
    pauli = {"I": s0, "X": sx, "Y": sy, "Z": sz, "R": sp, "L": sm}
    h = 1
    for i, s in enumerate(string):
        if s in pauli:
            h = h * pauli[s][i]
        elif s == "p":
            h = h * (s0[i] - sz[i]) / 2
        elif s == "q":
            h = h * (s0[i] + sz[i]) / 2
        elif s == "S":
            h = h * (s0[i] - sx[i]) / 2
        elif s == "O":
            h = h * (0.5*(1.0 + 1.0))*s0[i] + (0.5*(-1.0 + 1.0))*sz[i] + sx[i]
        else:
            raise ValueError(f"Unknown op '{s}' in string '{string}'")
    return h


# ----------------------
#  Small helper speedups
# ----------------------

def list_to_index(mli: List) -> List[int]:
    """
    O(n) stable factorization: assign a contiguous integer ID
    to each first-seen unique element in mli.
    """
    lookup: Dict = {}
    out: List[int] = []
    nxt = 0
    for item in mli:
        idx = lookup.get(item)
        if idx is None:
            idx = nxt
            lookup[item] = idx
            nxt += 1
        out.append(idx)
    return out


def sum_list_string(chars: List[str]) -> str:
    return "".join(chars)


# ----------------------
#  Original helpers kept
# ----------------------

def get_possible_transitions(es, s):
    e1 = set([e[0:s] for e in es])
    e_dict={}
    eslist=list(es)
    for e in e1:
        e2 = [i[-1] for i in eslist if i[0:s]==e]
        e_dict[e]=e2
    return e_dict


# ----------------------
#  Graph constructors
# ----------------------

def initializing(H: List[str], H_site: int, num_total_sites: int):
    edgeset_0 = sorted(set(h[H_site] for h in H))
    paths = {}
    j = H_site
    w = j + 1
    for n, op in enumerate([term[H_site] for term in H]):
        i = edgeset_0.index(op)     # edgeset_0 <= 10 → fine to index
        paths[n] = {f"n{j}_{j}": j, (H_site, j, i): op, f"n{w}_{op}": i}
    return paths


def get_left_G(H: List[str], num_total_sites: int):
    graph_i = initializing(H, 0, num_total_sites)
    half_chin_loc = num_total_sites // 2 + (num_total_sites % 2)

    for site in range(1, half_chin_loc):
        sslice = slice(0, site+1)
        hs_terms = [h[sslice] for h in H]
        hs_terms_idx = list_to_index(hs_terms)

        for n, path in graph_i.items():
            vals = list(path.values())
            node_v = vals[::2][-1]
            curr = H[n][sslice]
            path[(site, node_v, hs_terms_idx[n])] = curr[-1]
            path[f"n{hs_terms_idx[n]}_{curr}"] = hs_terms_idx[n]
    return graph_i


def get_right_G(H: List[str], num_total_sites: int):
    graph_f = {n:{} for n in range(len(H))}
    half_chin_loc = num_total_sites // 2 + (num_total_sites % 2)

    op_set = {"I":0, "X":1, "Y":2, "Z":3, "p":4, "q":5, "R":6, "L":7, "S":8, "O":9}

    for site in range(half_chin_loc, num_total_sites):
        sslice = slice(site, num_total_sites)
        hs_terms = [h[sslice] for h in H]
        hs_terms_idx = list_to_index(hs_terms)

        for n, term in enumerate(hs_terms):
            graph_f[n][f"n{hs_terms_idx[n]}_{term}"] = hs_terms_idx[n]
            graph_f[n][(site, op_set[H[n][site]], hs_terms_idx[n])] = H[n][site]

    for n in range(len(H)):
        graph_f[n][f"n{num_total_sites}_{0}"] = 0

    return graph_f


# ----------------------
#  Fast S assembly (O(T))
# ----------------------

def _left_state_id(graph_L: Dict[int, Dict], n: int) -> int:
    vals = list(graph_L[n].values())
    return vals[::2][-1]

def _right_state_id(graph_R: Dict[int, Dict], n: int) -> int:
    vals = list(graph_R[n].values())
    return vals[::2][0]


def _compute_lambda_coefs_matrix_fast(graph_L, graph_R, H, coefsym_dict, coefs_dict):
    num_terms = len(H)
    L_ids = np.fromiter((_left_state_id(graph_L, i)  for i in range(num_terms)), dtype=np.int64)
    R_ids = np.fromiter((_right_state_id(graph_R, i) for i in range(num_terms)), dtype=np.int64)
    JL = int(L_ids.max()) + 1 if L_ids.size else 1
    JR = int(R_ids.max()) + 1 if R_ids.size else 1

    M   = np.zeros((JL, JR), dtype=np.complex128)
    Msym= Msym = np.full((JL, JR), np.nan)#np.zeros((JL, JR), dtype=np.float64)

    for i in range(num_terms):
        iL = int(L_ids[i]); jR = int(R_ids[i])
        term = H[i]
        c = coefs_dict[term]
        M[iL, jR] += c
        Msym[iL, jR] = coefsym_dict[term]

    return M, Msym


def compute_lambda_coefs_matrix(graph_L, graph_R, H, num_total_sites, coefsym_dict, coefs_dict):
    return _compute_lambda_coefs_matrix_fast(graph_L, graph_R, H, coefsym_dict, coefs_dict)

def compute_lambda_coefs_matrix_1(graph_L, graph_R, H, num_total_sites, coefsym_dict, coefs_dict):
    return _compute_lambda_coefs_matrix_fast(graph_L, graph_R, H, coefsym_dict, coefs_dict)


# ----------------------
#  Graph visualization
# ----------------------

def make_graph(paths, Hamiltonian, start_site=0, final_site=1):
    """
    Build a graph for visualization. If rustworkx is available, returns a
    rustworkx.PyDiGraph; otherwise returns a networkx.DiGraph.
    API (name/signature) is unchanged.
    """
    # Gather layers & edges
    labels = {}
    for site in range(start_site, final_site):
        nodes_layer_0=[]
        nodes_layer_1=[]
        edges_0_to_1=[]
        vals_all = list(paths.values())
        for i in range(len(vals_all)):
            keys_i = list(vals_all[i].keys())
            vals_i = list(vals_all[i].values())
            nodes_layer_0.append(keys_i[::2][site])
            nodes_layer_1.append(keys_i[::2][site+1])
            edges_0_to_1.append(vals_i[1::2][site])
        edge_pair = [k for k in zip(nodes_layer_0, nodes_layer_1)]
        labels.update({key: val for key, val in zip(edge_pair, edges_0_to_1)})

    # if _HAVE_RX:
    #     G = rx.PyDiGraph()
    #     node_index = {}  # map label->node_id
    #     def nid(label):
    #         if label in node_index:
    #             return node_index[label]
    #         idx = G.add_node(label)
    #         node_index[label] = idx
    #         return idx

    #     for (u_lab, v_lab), e_lab in labels.items():
    #         u = nid(u_lab); v = nid(v_lab)
    #         G.add_edge(u, v, e_lab)

    #     # 'pos' and 'labels' for rustworkx are not standard; return dummies for compat
    #     pos = {}   # not used by your pipeline
    #     return G, pos, labels
    # else:
        G = nx.DiGraph()
        pos = {}
        graph_length = max(len(list(val)[1::2]) for val in paths.values())
        for site in range(start_site, final_site):
            nodes_layer_0=[]
            nodes_layer_1=[]
            edges_0_to_1=[]
            vals_all = list(paths.values())
            for i in range(len(vals_all)):
                keys_i = list(vals_all[i].keys())
                vals_i = list(vals_all[i].values())
                nodes_layer_0.append(keys_i[::2][site])
                nodes_layer_1.append(keys_i[::2][site+1])
                edges_0_to_1.append(vals_i[1::2][site])

            layer_0=sorted(set(nodes_layer_0))
            layer_1=sorted(set(nodes_layer_1))
            G.add_nodes_from(layer_0)
            edge_pair = [k for k in zip(nodes_layer_0, nodes_layer_1)]
            local_labels = {key: val for key, val in zip(edge_pair, edges_0_to_1)}
            for (u, v), lab in local_labels.items():
                G.add_edge(u, v, label=lab)
            pos.update({n: (2*site, -i) for i, n in enumerate(layer_0)})

        G.add_nodes_from(layer_1)
        tot_sites = max(len(h) for h in Hamiltonian)
        if final_site == tot_sites:
            pos.update({n: (3*site + 2*tot_sites, -len(layer_0)+1) for i, n in enumerate(layer_1)})
        else:
            pos.update({n: (3*site+2, -i) for i, n in enumerate(layer_1)})
        return G, pos, labels


# ----------------------
#  MPO tensor builders
# ----------------------

def _make_W_site(site: int,
                 paths_items: List[Tuple[int, Dict]],
                 coefs_dict: Dict[str, complex],
                 coefsym_dict: Dict[str, int]):
    """
    Helper to build a single site's (Wsym, W) — used by make_Ws.
    Kept separate so we can thread it safely (no cross-talk).
    """
    local_d = 2
    pauli = Pauli()

    nodes_layer_0=[]
    nodes_layer_1=[]
    edges_0_to_1=[]
    for _, pth in paths_items:
        keys_i = list(pth.keys())
        vals_i = list(pth.values())
        nodes_layer_0.append(keys_i[::2][site])
        nodes_layer_1.append(keys_i[::2][site+1])
        edges_0_to_1.append(vals_i[1::2][site])

    layer_0=sorted(set(nodes_layer_0))
    layer_1=sorted(set(nodes_layer_1))
    index_0= list_to_index(nodes_layer_0)
    index_1= list_to_index(nodes_layer_1)

    connections = list(zip(index_0, index_1, nodes_layer_0, nodes_layer_1, edges_0_to_1))

    Wsym = np.empty((len(layer_0), len(layer_1)), dtype=object)
    Wsym.fill(None)
    W = np.zeros((local_d, local_d, len(layer_0), len(layer_1)), dtype=np.complex128)

    # collapse duplicates
    for xr, yc, _, _, label in set(connections):
        Wsym[xr, yc] = label
        if isinstance(label, str) and len(label) == 1:
            W[:, :, xr, yc] = pauli.get_pauli(label)
        else:
            # fallback to original convention (rare path)
            if isinstance(label, tuple) and len(label) == 2:
                term_sym_idx, op_char = label
                # resolve term string back from symbol index
                # (linear scan is OK: only at graph build time)
                term = next(k for k, v in coefsym_dict.items() if v == term_sym_idx)
                W[:, :, xr, yc] = coefs_dict[term] * pauli.get_pauli(op_char)
            else:
                # If ever a different encoding sneaks in, attempt direct op char
                W[:, :, xr, yc] = pauli.get_pauli(str(label))
    return Wsym, W


def make_Ws(paths, start_site, final_site, coefs_dict, coefsym_dict):
    """
    Same signature. Now optionally parallel over sites (threaded).
    """
    paths_items = list(paths.items())
    n_sites = max(0, final_site - start_site)
    if _PARALLEL_MAKE_WS and n_sites > 1 and _MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, n_sites)) as ex:
            futures = [ex.submit(_make_W_site, site, paths_items, coefs_dict, coefsym_dict)
                       for site in range(start_site, final_site)]
            results = [f.result() for f in futures]
    else:
        results = [_make_W_site(site, paths_items, coefs_dict, coefsym_dict)
                   for site in range(start_site, final_site)]

    Wsym_s, Ws = zip(*results) if results else ([], [])
    return list(Wsym_s), list(Ws)


def build_mpo_tensors(H: List[str],
                      num_total_sites: int,
                      coefs_dict: Dict[str, complex],
                      coefsym_dict: Dict[str, int]):
    """
    Same signature; internal S-build O(T) and same SVD glue.
    """
    graph_L = get_left_G(H, num_total_sites)
    graph_R = get_right_G(H, num_total_sites)

    if num_total_sites % 2 == 0:
        left_half = num_total_sites // 2
        right_half = num_total_sites // 2
    else:
        left_half = num_total_sites // 2 + 1
        right_half = num_total_sites // 2

    wsyms_L, ws_L = make_Ws(graph_L, 0, left_half, coefs_dict, coefsym_dict)
    wsyms_R, ws_R = make_Ws(graph_R, 0, right_half, coefs_dict, coefsym_dict)

    S, _ = compute_lambda_coefs_matrix_1(graph_L, graph_R, H, num_total_sites, coefsym_dict, coefs_dict)

    x, y, z = np.linalg.svd(S, full_matrices=False)
    keep = y > 1e-8
    if not np.any(keep):
        keep = np.array([True] + [False]*(y.shape[0]-1))
    x = x[:, keep]; y = y[keep]; z = z[keep, :]

    ws = ws_L
    ws[-1] = np.tensordot(ws[-1], x, axes=(-1, 0))  # (..., chiL) x (chiL, r) -> (..., r)

    yz = np.tensordot(np.diag(y), z, axes=(-1, 0))  # (r,r) x (r,chiR) -> (r,chiR)
    wR_new = np.transpose(np.tensordot(yz, ws_R[0], axes=(-1, 2)), (1, 2, 0, 3))
    ws.append(wR_new)
    for i in range(1, len(ws_R)):
        ws.append(ws_R[i])

    return ws






def build_mpo_tensors_in_details(H: List[str],
                                 num_total_sites: int,
                                 coefs_dict: Dict[str, complex],
                                 coefsym_dict: Dict[str, int],
                                 chi_max=1500,
                                 epsilon=1e-8):
    """
    Same returns: ws , ws_L, ws_R, S
    """
    import copy
    import numpy as np # Ensure numpy is imported if not globally available

    graph_L = get_left_G(H, num_total_sites)
    graph_R = get_right_G(H, num_total_sites)

    if num_total_sites % 2 == 0:
        left_half = num_total_sites // 2
        right_half = num_total_sites // 2
    else:
        left_half = num_total_sites // 2 + 1
        right_half = num_total_sites // 2

    wsyms_L, ws_L = make_Ws(graph_L, 0, left_half, coefs_dict, coefsym_dict)
    wsyms_R, ws_R = make_Ws(graph_R, 0, right_half, coefs_dict, coefsym_dict)
    print(f"shapes of left and right : {ws_L[-1].shape} and {ws_R[0].shape}")

    S, S_sym = compute_lambda_coefs_matrix_1(graph_L, graph_R, H, num_total_sites, coefsym_dict, coefs_dict)
    print(f"shape of S = {S.shape}")
    
    # 1. Economic SVD (Critical for non-square matrices)
    x, y, z = np.linalg.svd(S, full_matrices=False)
    
    # 2. Determine Rank (Count valid singular values)
    keep = np.sum(y > epsilon)

    # 3. Apply Hard Limit (chi_max)
    keep = min(keep, chi_max)

    # 4. Safety: Always keep at least 1 singular value to prevent crash
    keep = max(keep, 1)

    # 5. Apply Truncation using SLICING (:)
    # ERROR FIX: x[:, keep] grabs ONE column. x[:, :keep] grabs TOP 'keep' columns.
    x = x[:, :keep]
    y = y[:keep]
    z = z[:keep, :]

    ws = copy.deepcopy(ws_L)
    
    # Contract Left side
    ws[-1] = np.tensordot(ws[-1], x, axes=(-1, 0))
    
    # Contract Center (Diagonal * Right Vectors)
    # y is guaranteed to be 1D array here, so np.diag works
    yz = np.tensordot(np.diag(y), z, axes=(-1, 0))
    
    # Contract Right side
    wR_new = np.transpose(np.tensordot(yz, ws_R[0], axes=(-1, 2)), (1, 2, 0, 3))
    
    ws.append(wR_new)
    for i in range(1, len(ws_R)):
        ws.append(ws_R[i])

    return ws, ws_L, ws_R, S, S_sym



def build_mpo_tensors_in_details_truncated(
    H, num_total_sites, coefs_dict, coefsym_dict,
    chi_max=1000, epsilon=1e-8, *, normalize_halves=True, verbose=True
):
    import copy
    import numpy as np

    def ortho_keep(s, eps, chi):
        k = int(np.sum(s > eps))
        k = min(k, int(chi))
        return max(k, 1)

    def left_normalize_list(Ms, trunc=True, eps=1e-12, chi=10**9):
        Ms = [np.array(T, copy=True) for T in Ms]
        out, discarded, tail = [], 0.0, None
        for i in range(len(Ms)):
            W = Ms[i]
            d1, d2, wL, wR = W.shape
            B = W.reshape(d1 * d2 * wL, wR)
            U, s, Vh = np.linalg.svd(B, full_matrices=False)
            k = ortho_keep(s, eps, chi) if trunc else len(s)
            discarded += (np.sum(s[k:] ** 2) / (np.sum(s ** 2) + 1e-30))
            A = U[:, :k].reshape(d1, d2, wL, k)
            R = (np.diag(s[:k]) @ Vh[:k, :])  # (k, wR)
            out.append(A)
            if i < len(Ms) - 1:
                Wn = Ms[i + 1]
                tmp = np.tensordot(R, Wn, axes=(1, 2))      # (k,d,d,wnR)
                Ms[i + 1] = np.transpose(tmp, (1, 2, 0, 3)) # (d,d,k,wnR)
            else:
                tail = R
        return out, tail, discarded

    def right_normalize_list(Ms, trunc=True, eps=1e-12, chi=10**9):
        Ms = [np.array(T, copy=True) for T in Ms]
        out_rev, discarded, head = [], 0.0, None
        L = len(Ms)
        for i in range(L):
            idx = L - 1 - i
            W = Ms[idx]
            d1, d2, wL, wR = W.shape
            B = np.transpose(W, (2, 0, 1, 3)).reshape(wL, d1 * d2 * wR)
            U, s, Vh = np.linalg.svd(B, full_matrices=False)
            k = ortho_keep(s, eps, chi) if trunc else len(s)
            discarded += (np.sum(s[k:] ** 2) / (np.sum(s ** 2) + 1e-30))

            Vh_k = Vh[:k, :]
            Wnew = np.transpose(Vh_k.reshape(k, d1, d2, wR), (1, 2, 0, 3))  # (d,d,k,wR)
            out_rev.append(Wnew)

            Umat = U[:, :k] @ np.diag(s[:k])  # (wL,k)
            if idx > 0:
                Wp = Ms[idx - 1]
                Ms[idx - 1] = np.tensordot(Wp, Umat, axes=(3, 0))  # (d,d,wPrev,k)
            else:
                head = Umat
        return out_rev[::-1], head, discarded

    graph_L = get_left_G(H, num_total_sites)
    graph_R = get_right_G(H, num_total_sites)

    left_half = num_total_sites // 2 if (num_total_sites % 2 == 0) else (num_total_sites // 2 + 1)
    right_half = num_total_sites // 2

    _, ws_L = make_Ws(graph_L, 0, left_half,  coefs_dict, coefsym_dict)
    _, ws_R = make_Ws(graph_R, 0, right_half, coefs_dict, coefsym_dict)

    S_raw, S_sym = compute_lambda_coefs_matrix_1(graph_L, graph_R, H, num_total_sites, coefsym_dict, coefs_dict)
    aL_old, aR_old = S_raw.shape

    if verbose:
        print(f"[raw] ws_L[-1]={ws_L[-1].shape}  ws_R[0]={ws_R[0].shape}  S={S_raw.shape}")

    # SVD truncate S_raw
    U, s, Vh = np.linalg.svd(S_raw, full_matrices=False)
    k0 = ortho_keep(s, epsilon, chi_max)
    U = U[:, :k0]      # (aL_old,k0)
    s = s[:k0]         # (k0,)
    Vh = Vh[:k0, :]    # (k0,aR_old)

    ws_L = [np.array(T, copy=True) for T in ws_L]
    ws_R = [np.array(T, copy=True) for T in ws_R]

    ws_L[-1] = np.tensordot(ws_L[-1], U, axes=(3, 0))                 # (...,k0)
    tmp = np.tensordot(Vh, ws_R[0], axes=(1, 2))                      # (k0,d,d,wR)
    ws_R[0] = np.transpose(tmp, (1, 2, 0, 3))                         # (d,d,k0,wR)

    S = np.diag(s).astype(np.complex128)
    S_sym = None

    if verbose:
        print(f"[svd] kept k0={k0}  ws_L[-1]={ws_L[-1].shape}  ws_R[0]={ws_R[0].shape}  S={S.shape}")

    # If we normalize halves, we absorb TL/TR into S, and that changes the cut basis.
    TL = np.eye(k0, dtype=np.complex128)
    TR = np.eye(k0, dtype=np.complex128)
    if normalize_halves:
        ws_L, TL, discL = left_normalize_list(ws_L, trunc=True, eps=epsilon, chi=chi_max)
        S = TL @ S
        ws_R, TR, discR = right_normalize_list(ws_R, trunc=True, eps=epsilon, chi=chi_max)
        S = S @ TR
        if verbose:
            print(f"[norm] ws_L[-1]={ws_L[-1].shape}  ws_R[0]={ws_R[0].shape}  S={S.shape}")
            print(f"[norm] discarded_L={discL:.3e}  discarded_R={discR:.3e}")

    # Maps old bridge basis -> new bridge basis
    # X_new = A_left @ X_old @ A_right
    A_left  = (TL @ U.conj().T).astype(np.complex128)                 # (kL_final, aL_old)
    A_right = (Vh.conj().T @ TR).astype(np.complex128)                # (aR_old, kR_final)

    # Build full ws (absorb S into first right tensor)
    wR0_full = np.transpose(np.tensordot(S, ws_R[0], axes=(1, 2)), (1, 2, 0, 3))
    ws = copy.deepcopy(ws_L)
    ws.append(wR0_full)
    ws.extend(ws_R[1:])

    if verbose:
        print(f"[full] ws len={len(ws)}  ws[0]={ws[0].shape}  ws[-1]={ws[-1].shape}")

    return ws, ws_L, ws_R, S, S_sym, A_left, A_right, S_raw  # S_raw returned only to know old dims



