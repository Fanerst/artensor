import torch
from math import sqrt, pi
from numpy import exp

def u3_gate(theta, phi, lam):
    cos, sin = torch.cos(theta.reshape(1)/4), torch.sin(theta.reshape(1)/4)
    gate = torch.stack([
        torch.cat([cos, -torch.exp(1j*lam) * sin]),
        torch.cat([torch.exp(1j*phi) * sin, torch.exp(1j*(lam+phi)) * cos])
    ], dim=0)
    return gate

def cu3_gate(theta, phi, lam, dtype=torch.complex64, device='cpu'):
    cos, sin = torch.cos(theta.reshape(1)/4), torch.sin(theta.reshape(1)/4)

    gate = torch.block_diag(*[
        torch.eye(2, dtype=dtype, device=device), 
        torch.stack([
            torch.cat([cos, -torch.exp(1j*lam) * sin]),
            torch.cat([torch.exp(1j*phi) * sin, torch.exp(1j*(lam+phi)) * cos])
        ], dim=0)
    ])
    return gate.reshape(2,2,2,2)

def fsim_gate(theta, phi, dtype=torch.complex64, device='cpu'):
    cos, sin = torch.cos(torch.tensor([theta])), torch.sin(torch.tensor([theta]))
    gate = torch.zeros([4, 4], dtype=dtype, device=device)
    gate[0, 0] = 1
    gate[1:3, 1:3] = torch.stack([
        torch.cat([cos, -1j * sin]),
        torch.cat([-1j * sin, cos])
    ], dim=0)
    gate[3, 3] = torch.exp(-1j*torch.tensor([phi]))
    return gate.reshape(2,2,2,2)

def xsqrt_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        # [1, -1j], [-1j, 1]
        [exp(1j * pi/4), exp(-1j * pi/4)], [exp(-1j * pi/4), exp(1j * pi/4)]
    ], dtype=dtype, device=device) / sqrt(2)
    return gate

def ysqrt_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        # [1, -1], [1, 1]
        [exp(1j * pi/4), -exp(1j * pi/4)], [exp(1j * pi/4), exp(1j * pi/4)]
    ], dtype=dtype, device=device) / sqrt(2)
    return gate

def wsqrt_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        # [1, -exp(1j * pi/4)], [exp(-1j * pi/4), 1]
        [exp(1j * pi/4), -1j], [1, exp(1j * pi/4)]
    ], dtype=dtype, device=device) / sqrt(2)
    return gate

def rz_gate(phi, dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        [exp(-1j * phi/2), 0], [0, exp(1j * phi/2)]
    ], dtype=dtype, device=device)
    return gate

def cz_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=dtype, device=device)
    return gate.reshape(2,2,2,2)

def cnot_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=dtype, device=device)
    return gate.reshape(2,2,2,2)

def hadamard_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        [1, 1], [1, -1]
    ], dtype=dtype, device=device) / sqrt(2)
    return gate

def zz_gate(beta, dtype=torch.complex64, device='cpu'):
    pauli_z = torch.tensor([
        [1, 0], [0, -1]
    ], dtype=dtype, device=device)
    gate = torch.exp(-0.5*1j*beta) * torch.kron(pauli_z, pauli_z)
    return gate.reshape(2,2,2,2)

def t_gate(phi, dtype=torch.complex64, device='cpu'):
    gate = torch.tensor(
        [[1, 0], [0, torch.exp(1j*phi)]], dtype=dtype, device=device
    )
    return gate

def s_gate(dtype=torch.complex64, device='cpu'):
    gate = torch.tensor([
        [1, 0], [0, 1j]
    ], dtype=dtype, device=device)
    return gate