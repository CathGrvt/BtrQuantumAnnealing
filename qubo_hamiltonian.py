import numpy as np
import dimod

def generate_hamiltonian(N, theta, r, alpha, beta, lamb):
    # Create the s variables
    s = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            s[i,j] = s[j,i] = 1
    
    # Calculate the terms of the Hamiltonian
    H1 = 0
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                H1 += (np.cos(theta[a,b,c])**lamb / (r[a,b] + r[b,c])) * s[a,b] * s[b,c]

    H2 = - alpha * (np.sum(s) - N)**2
    H3 = - beta * np.sum(s)**2

    # Convert to QUBO or BQM format
    Q = {(i,i): H2 + H3 for i in range(N)}
    for i in range(N):
        for j in range(i+1, N):
            Q[(i,j)] = H1
            Q[(i,i)] += - alpha - 2*beta*N
            Q[(j,j)] += - alpha - 2*beta*N

    # Return the QUBO or BQM problem
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    return bqm

N = 4  # number of particles
theta = np.random.uniform(size=(N,N,N))  # angles between particles
r = np.random.uniform(size=(N,N))  # distances between particles
alpha = 0.5  # parameter
beta = 1.0  # parameter
lamb = 2.0  # parameter

bqm = generate_hamiltonian(N, theta, r, alpha, beta, lamb) #to generate the QUBO or BQM problem

#The resulting bqm object can be used with a D-Wave quantum annealer 
# or a classical solver to find the minimum energy state of the Hamiltonian.




