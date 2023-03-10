import toymodel_3d as toy
import dp_hamiltonian as ham
import numpy as np
import matplotlib.pyplot as plt
import dimod
from dwave.samplers import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()

N_MODULES = 3
N_TRACKS = 3

LX = 2
LY = 2
SPACING = 1

detector = toy.generate_simple_detector(N_MODULES, LX, LY, SPACING)
event = toy.generate_event(detector, N_TRACKS, theta_max=np.pi / 50, seed=1)

fig = plt.figure()
fig.set_size_inches(12, 6)
ax = plt.axes(projection='3d')
event.display(ax)
ax.view_init(vertical_axis='y')
fig.set_tight_layout(True)
ax.axis('off')
ax.set_title(f"Generated event\n{len(event.modules)} modules\n{len(event.tracks)} tracks - {len(event.hits)} hits")
plt.show()

params = {
    'alpha': 0.0,
    'beta': 1.0,
    'lambda': 100.0,
}
A, b, components, segments = ham.generate_hamiltonian(event, params)

fig = plt.figure()
fig.set_size_inches(12, 6)
ax = plt.axes(projection='3d')
event.display(ax, show_tracks=False)

for segment in segments:
    segment.display(ax)

ax.view_init(vertical_axis='y')
fig.set_tight_layout(True)
ax.axis('off')

ax.set_title(f"{len(segments)} segments generated")

plt.show()

# Define the BQM and sampler for simulated annealing
bqm = dimod.BinaryQuadraticModel.from_qubo(A)
sampler = dimod.SimulatedAnnealingSampler()

# Run simulated annealing and retrieve the best sample
response = sampler.sample(bqm, num_reads=1000)
best_sample = response.record.sample[0]

sampleset = sampler.sample(bqm)
sampleset = sampler.sample(bqm, beta_range=[.1, 4.2], beta_schedule_type='linear')
print(sampleset.first.energy)