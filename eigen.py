'''Calculate dominant eigenvalues of symmetric matrices the with Lanczos algorithm'''
from torch.autograd import grad
import torch as th
import numpy as np

class HVP(object):
    """Operator returning a Hessian vector product."""

    def __init__(self, f, x, fd=False, h=1e-1, order='O1', **kwargs):
        """Initialize the class with a function f evaluated at point x."""
        super().__init__()

        self.fd = fd
        self.h  = h
        self.order = order

        if not callable(f) and fd:
            raise TypeError('f must be calleable')

        if callable(f):
            self.fval = f(x)
            self.f = f
        else:
            self.fval = self.f = f


        if not fd:
            dx = grad(self.fval,x, create_graph =True)
            self.dx = dx[0]
        else:
            self.dx = None

        self.x = x

    def __call__(self, v):

        if not self.fd:
            dxpv = self.dx*v
        else:
            # TODO: Check that v is a unit vector
            x = self.x
            h = self.h
            f = self.f
            if self.order == 'O1':
                dxpv = (f(self.x+h*v) - self.fval)/h 
            elif self.order == 'O2':
                dxpv = (f(x+0.5*h*v) - f(x-0.5*h*v))/h
        dxpv = dxpv.sum()

        dx2 = grad(dxpv,self.x, retain_graph = True)
        dx2 = dx2[0]

        return  dx2



def batch_symeigs(operator, shape, device=th.device('cpu'), maxiters=20, tol=1e-4, **kwargs):
    '''Compute dominant eigenvalues of a batched symmetric operator, using Lanczos iteration [1].

    Arguments:
        operator (function): A *symmetric* operator, mapping PyTorch tensors to
            PyTorch tensors of the same size
        shape (tuple): the shape of input and output tensors of the operator. The first dimension
            is the batch dimension.

    Optional arguments:
        device (torch.device): which device to create torch.tensors on (default: 'cpu')
        maxiters (int): maximum number of Lanczos iterations (default: 20)
        tol (float): early stopping criteria (default: 1e-4)

    Returns:
        E (torch.tensor): for each batch in inputs, the dominant eigenvalues of the operator
    
    [1] Golub & Van Loan, "Matrix Computations", 2012.'''


    Bsz = shape[0]
    Ndim = 1
    for s in shape[1:]:
        Ndim *=s
    maxiters = min(Ndim,maxiters)

    
    # Initialization
    q = th.randn((Bsz,Ndim),device=device)
    q = q.div_(q.norm(2,-1,keepdim=True))

    qlast = th.zeros((Bsz,Ndim),device=device)
    qlast2 = th.zeros((Bsz,Ndim),device=device)
    alpha = th.zeros((Bsz,maxiters+1), device=device)
    beta = th.zeros((Bsz,maxiters+1), device=device)
    beta[:, 0] = 1.
    betalast = th.ones(Bsz, device=device)
    T = th.zeros((Bsz,maxiters,maxiters),device=device)

    for i in range(1,maxiters+1):

        Aq = operator(q.view(shape)).detach()
        Aq = Aq.view(Bsz,-1)


        r = Aq - betalast.view(Bsz,1)*qlast
        alpha_ = th.einsum('ij,ij->i',q,r)
        r = r - alpha_.view(Bsz,1)*q

        alpha[:,i] = alpha_
        T[:,i-1,i-1] = alpha_

        # Compute eigenvalues of T. Terminate if the min and max eigenvalues
        # haven't changed significantly.
        #
        # TODO: Consider letting the user define their own stopping criteria.
        #
        # TODO: batch QR algorithm for eigenvalues, or at least open a feature request on PyTorch.
        # Since T is symmetric tridiagonal, this can be done in O(n^2)
        # operations, for each matrix in the batch.
        #
        # See for example
        # [2] K Gates & WB Gragg. "Notes on TQR algorithms." J Computational
        #      and Applied Mathematics, 86: 195-203, 1997.
        if i>1:
            mine, maxe = minE.clone(), maxE.clone()
            E = th.zeros(Bsz,i, device=device)
            for bs, t in enumerate(T[:,:i,:i]):
                try:
                    e, _ = t.symeig()
                    minE[bs], maxE[bs] = e.min(), e.max()
                except RuntimeError:
                    # Sometimes MAGMA fails to find eigs
                    e = th.full((len(t),),np.nan,device=device)

                E[bs,:] = e
            mintol = (mine-minE).abs().max().item()
            maxtol = (maxe-maxE).abs().max().item()


            tol_ = max(mintol,maxtol)
            if tol_ < tol:
                return E
        else:
            mine, maxe = alpha_.clone(), alpha_.clone()
            minE, maxE = alpha_.clone(), alpha_.clone()

        # Orthonormalize the newest Lanczos vectors q with respect to previous
        # two vectorss (for numerical stability)
        #
        # TODO: implement [3] to detect spurious eigenvalues
        # [3] JK Cullum ad RA Willoughby. "Computing eigenvalues of very large symmetric matrices --
        #     an implementation of a Lanczos algorithm with no reorthogonalization."
        #     J Comput Phys, 44:329-358, 1981
        rorth = r - th.einsum('ij,ij->i',r,qlast).view(Bsz,1)*qlast 
        rorth = rorth - th.einsum('ij,ij->i',rorth,qlast2).view(Bsz,1)*qlast2
        betalast = rorth.norm(2,-1)
        beta[:,i] = betalast

        # Update off diagonals of T
        if i<maxiters:
            T[:,i-1,i] = betalast
            T[:,i,i-1] = betalast

        
        qlast2 = qlast.clone()
        qlast = q.clone()
        q = rorth/betalast.view(Bsz,1)


    return E

def dominant_hessian_eigs(f, x, **kwargs):

    operator = HVP(f,x, **kwargs)
    eigs = batch_symeigs(operator, x.shape, device=x.device, **kwargs)

    return eigs.min(dim=-1)[0], eigs.max(dim=-1)[0]

def test():
    import time

    has_cuda= th.cuda.is_available()

    d = 100
    bsz = 64
    M = th.randn(bsz,d,d).cuda()
    if has_cuda:
        M = M.cuda()
    M = 0.5*(M+M.transpose(1,2))

    Etrue = th.zeros(bsz,d, device=M.device)
    for i, m in enumerate(M):
        e,_ = m.symeig(eigenvectors=False)
        Etrue[i] = e

    x = th.randn(bsz,d,1,device=M.device)
    x.requires_grad_()

    f = lambda x: (0.5*x.transpose(1,2).matmul(M.matmul(x))).sum()

    operator = HVP(f,x, fd=True,h=0.1)
    t = time.time()
    Efd = batch_symeigs(operator, x.shape, device=x.device, maxiters=d)
    t = time.time()-t
    print('With finite differences:')
    print('\tmax eig error: %.4e'%((Efd.max(dim=-1)[0]-Etrue.max(dim=-1)[0]).abs().max()))
    print('\tmin eig error: %.4e'%((Efd.min(dim=-1)[0]-Etrue.min(dim=-1)[0]).abs().max()))
    print('\ttime: %.4g'%t)

    operator = HVP(f,x, fd=False)
    t = time.time()
    E = batch_symeigs(operator, x.shape, device=x.device, maxiters=200)
    t = time.time()-t
    print('\nWith autograd:')
    print('\tmax eig error: %.4e'%((E.max(dim=-1)[0]-Etrue.max(dim=-1)[0]).abs().max()))
    print('\tmin eig error: %.4e'%((E.min(dim=-1)[0]-Etrue.min(dim=-1)[0]).abs().max()))
    print('\ttime: %.4g'%t)

if __name__=='__main__':
    test()
