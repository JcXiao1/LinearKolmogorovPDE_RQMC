from abc import ABC, abstractmethod
import torch
import numpy as np
from rng import Normdf_inv,Generate_sobol

class PDE(ABC):
    def __init__(self,params):
        super().__init__()
        self.params = params

    @abstractmethod
    def sde(x,w):
        pass
    
    @abstractmethod
    def solution(x):
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


PDE_params = {
    "BSmodel_params":{
        "x": [4.5,5.5],
        "t": 1,
        "N": 1,
        "mu" : -0.05,
        "K":5.5,
    },
    "HeatEquation_params":{
        "x": [0,1],
        "N": 1,
        "t": 1,
    }
}




class BSmodel(PDE):
    def __init__(
                self, 
                dimension, 
                params = PDE_params["BSmodel_params"], 
                device = torch.device('cuda:0')
                ):
        super().__init__(params)
        self.dim = dimension
        self.device = device
        self.std = np.sqrt(self.params["t"]/self.params["N"])
        beta = torch.linspace(0.1+1/(2*self.dim),0.6,steps=self.dim)
        Cmatirx = torch.linalg.cholesky((0.5*torch.ones((self.dim,self.dim))).fill_diagonal_(1),upper=False)
        self.sigma = beta.reshape(-1,1)*Cmatirx
        self.sigma = self.sigma.to(device)
        t = self.params["t"]
        mu = self.params["mu"]
        self.drift = torch.exp((mu-0.5*torch.norm(self.sigma,p=2,dim=1)**2)*t)
        self.drift = self.drift.to(device)

    #x denote the standard uniform sample 
    def region_uniform(self,x):
        lower,upper = self.params["x"]
        return (upper - lower)*x + lower
    
    def sde(self,x,w):
        w = self.std * w
        mu = self.params["mu"]
        t = self.params["t"]
        K = self.params["K"]
        sigma = self.sigma
        drift = self.drift
        x = x*torch.exp(torch.mm(w,sigma.t()))*drift
        return np.exp(-mu*t)*torch.max(torch.Tensor([0]).to(self.device),K-torch.min(x,dim=1).values).unsqueeze(1)

    #x denote the uniform sample 
    def solution(self,x,n_path=2**23,method = "MC"):
        size,dim = x.shape
        y = torch.zeros((size,1)).to(self.device)
        for i in range(size):
            if method == "RQMC":
                rqmc = Generate_sobol(n_path,dim,0,1,scramble=True)
                rqmc = rqmc.to(self.device)
                w = Normdf_inv(rqmc)
            if method == "MC":
                w = torch.randn(n_path, dim, device=self.device)
            sde_batch = self.sde(x[i].reshape(1,-1),w)
            y[i] = torch.mean(sde_batch)
        return y
    

class Heat_Paraboloid(PDE):
    def __init__(self, 
                 dimension,
                 params = PDE_params["HeatEquation_params"],
                 device = torch.device('cuda:0')
                 ):
        super().__init__(params)
        self.dim = dimension
        self.std = np.sqrt(self.params["t"]/self.params["N"])

    #x denote the standard uniform sample 
    def region_uniform(self,x):
        lower,upper = self.params["x"]
        return (upper - lower)*x + lower
    
    def sde(self,x, w):
        w = self.std * w
        return torch.sum((x + np.sqrt(2)*w)** 2, axis=1).unsqueeze(1)

    def solution(self,x):
        t = self.params["t"]
        dim = x.shape[1]
        y = torch.sum(x ** 2, axis=1) + 2 * t * dim
        y = y.unsqueeze(1)
        return y
    
    

PDES = {pde.__name__: pde for pde in PDE.get_subclasses()}

