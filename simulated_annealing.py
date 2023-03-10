import dimod
import numpy as np
import toymodel_3d as toy
import dp_hamiltonian as ham
import matplotlib.pyplot as plt


# Set up toy event and generate Hamiltonian
N_MODULES = 3
N_TRACKS = 2
LX = 2
LY = 2
SPACING = 1
detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)
params = {
    'alpha': 0.0,
    'beta': 1.0,
    'lambda': 100.0,
}
A, b, components, segments = ham.generate_hamiltonian(event, params)

# Define the BQM and sampler for simulated annealing
bqm = dimod.BinaryQuadraticModel.from_qubo(A)
sampler = dimod.SimulatedAnnealingSampler()

# Run simulated annealing and retrieve the best sample
response = sampler.sample(bqm, num_reads=1000)
best_sample = response.record.sample[0]

# Visualize the reconstructed tracks
fig = plt.figure()
fig.set_size_inches(12, 6)
ax = plt.axes(projection='3d')
event.display(ax, show_tracks=False)

reconstructed_tracks = []
for i in range(N_TRACKS):
    track = [j for j in range(len(components)) if best_sample[j + i*len(components)] == 1]
    reconstructed_tracks.append(track)
    for component in [components[j] for j in track]:
        component.display(ax, color='g')

ax.view_init(vertical_axis='y')
fig.set_tight_layout(True)
ax.axis('off')
ax.set_title(f"{N_TRACKS} tracks reconstructed")
plt.show()
