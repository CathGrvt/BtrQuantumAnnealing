import toymodel_3d as toy
import dp_hamiltonian as ham
import numpy as np
import matplotlib.pyplot as plt
import dimod


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
    'alpha': 0.1,
    'beta': 0.0,
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

fig, axs = plt.subplots(2,3)
fig.set_size_inches(10,6)
vmin = np.min([A.min()].extend(components[key].min() for key in components))
vmax = np.max([A.max()].extend(components[key].max() for key in components))
im = axs[0,0].matshow(A,vmin=vmin, vmax=vmax)
axs[0,0].set_title("A")

axs_raviter = iter(axs.ravel())
next(axs_raviter)
for key in components:
    ax = next(axs_raviter)
    ax.matshow(components[key],vmin=vmin, vmax=vmax)
    ax.set_title(key)

fig.colorbar(im, ax=axs.ravel().tolist())


# Define the BQM and sampler for simulated annealing
offset = 0.0
vartype = dimod.BINARY
bqm= dimod.BinaryQuadraticModel(b, A, offset, vartype)
sampler = dimod.SimulatedAnnealingSampler()

# Run simulated annealing and retrieve the best sample
response = sampler.sample(bqm, num_reads=1000)
best_sample = response.record.sample[0]
print(best_sample)
print(response.first.energy)


# Use the solution vector to select the corresponding segments from the event
solution_segments = [seg for sol, seg in zip(best_sample, segments) if sol == 1]

# Check if there are any segments in the solution
if len(solution_segments) == 0:
    print("No segments included in the solution.")
else:
    # Display the solution
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = plt.axes(projection='3d')
    event.display(ax, show_tracks=False)

    for segment in solution_segments:
        segment.display(ax)

    ax.view_init(vertical_axis='y')
    fig.set_tight_layout(True)
    ax.axis('off')
    ax.set_title(f"Solution")
    plt.show()

# Calculate Purity and Efficiency
true_tracks = len(event.tracks)
reconstructed_tracks = 0
for track in event.tracks:
    if all(segment in solution_segments for segment in event.tracks):
        reconstructed_tracks += 1

purity = reconstructed_tracks / len(solution_segments) if len(solution_segments) > 0 else 0.0
efficiency = reconstructed_tracks / true_tracks

# Create bar graphs for Purity and Efficiency
labels = ['Purity', 'Efficiency']
values = [purity, efficiency]

fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_ylim([0, 1])
ax.set_ylabel('Value')
ax.set_title('Purity and Efficiency')
plt.show()
