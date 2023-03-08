import numpy as np
import dimod
import toymodel_3d as toy

def generate_hamiltonian(event, params):
    detector = event.detector
    tracks = event.tracks

    # Create the s variables
    N = len(detector)
    s = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            s[i,j] = s[j,i] = 1
    
    # Calculate the terms of the Hamiltonian
    H1 = 0
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                r_ab = np.linalg.norm(detector[b] - detector[a])
                r_bc = np.linalg.norm(detector[c] - detector[b])
                theta_abc = tracks[(a,b,c)]
                H1 += (np.cos(theta_abc)**params["lamb"] / (r_ab + r_bc)) * s[a,b] * s[b,c]

    H2 = - params["alpha"] * (np.sum(s) - N)**2
    H3 = - params["beta"] * np.sum(s)**2

    # Convert to QUBO or BQM format
    Q = {(i,i): H2 + H3 for i in range(N)}
    for i in range(N):
        for j in range(i+1, N):
            Q[(i,j)] = H1
            Q[(i,i)] += - params["alpha"] - 2*params["beta"]*N
            Q[(j,j)] += - params["alpha"] - 2*params["beta"]*N

    # Return the QUBO or BQM problem
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    return bqm


# Define the event parameters
N_MODULES = 3
N_TRACKS = 3
LX = 2
LY = 2
SPACING = 1

# Generate the event
detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi/50, seed=1)

# Define the Hamiltonian parameters
params = {"alpha": 0.5, "beta": 1.0, "lamb": 2.0}

# Generate the Hamiltonian in BQM format
bqm = generate_hamiltonian(event, params)
