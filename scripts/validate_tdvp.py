import numpy as np, time
from qu_alg_qu_sim_tensor_networks.lattice.geometry import TriangularLattice
from qu_alg_qu_sim_tensor_networks.lattice import models
from qu_alg_qu_sim_tensor_networks import mps
import vnqpe_tn as V
from qu_alg_qu_sim_tensor_networks.p2tdvp import ParallelTDVP
lat=TriangularLattice(2,3,boundary="periodic"); H=models.heisenberg(lat); n=lat.num_sites; r=3
w,sm=V.system_mpo(H); psi,E=V.run_dmrg(sm,w,chi=3); m=V.coupled_model(sm,r); vec=V.mps_to_vector(psi)
dt, nst = 0.05, 20; ts=dt*np.arange(1,nst+1)
Pex=V.exact_pointer_distribution(H,vec,r,ts)
_,Pser,_,_=V.evolve(V.attach_pointer(psi,r),m,n,r,tmax=dt*nst,dt=dt,chi_max=64,eps=1e-14)
print(f"package serial TDVPIntegrator : max|P-Pexact| = {np.abs(Pser-Pex).max():.2e}")
def run(p, dt=dt, nst=nst):
    eng=ParallelTDVP(V.attach_pointer(psi,r), m.H_mpo, chi_max=64, n_parts=p)
    P=[]
    for k in range(nst):
        eng.step(dt); P.append(V.pointer_distribution(V.pointer_rdm(mps.MPS(eng.mps_tensors(),[np.ones(1)]*eng.L),n,r),r))
    return np.array(P), eng
for p in (1,2,4):
    P,eng=run(p)
    print(f"p2TDVP n_parts={p}: max|P-Pexact| = {np.abs(P-Pex).max():.2e}   max|P-P_serialpkg| = {np.abs(P-Pser).max():.2e}  parts={eng.parts}")
# time-step convergence of p=1 and p=4 against exact
for p in (1,4):
    for d in (0.05,0.025,0.0125):
        nn=int(round(1.0/d)); P,_=run(p,d,nn); Pe=V.exact_pointer_distribution(H,vec,r,d*np.arange(1,nn+1))
        print(f"  n_parts={p} dt={d}: max|P-Pexact| = {np.abs(P-Pe).max():.2e}")
