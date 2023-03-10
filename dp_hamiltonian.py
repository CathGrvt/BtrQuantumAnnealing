import numpy as np
import itertools
import q_event_model as em
import copy


def generate_hamiltonian(event: em.event, params: dict):
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = copy.deepcopy(event.modules)
    modules.sort(key=lambda a: a.z)

    segments = []
    for idx in range(len(modules) - 1):
        from_hits = modules[idx].hits
        to_hits = modules[idx + 1].hits

        for from_hit, to_hit in itertools.product(from_hits, to_hits):
            segments.append(em.segment(from_hit, to_hit))

    N = len(segments)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for (i, seg_i), (j, seg_j) in itertools.product(enumerate(segments), repeat=2):
        if i != j:
            r_ab = np.sqrt((seg_j.from_hit.x - seg_i.from_hit.x) ** 2 + (seg_j.from_hit.y - seg_i.from_hit.y) ** 2 + (
                    seg_j.from_hit.z - seg_i.from_hit.z) ** 2)
            r_bc = np.sqrt((seg_j.to_hit.x - seg_i.to_hit.x) ** 2 + (seg_j.to_hit.y - seg_i.to_hit.y) ** 2 + (
                    seg_j.to_hit.z - seg_i.to_hit.z) ** 2)
            r_ac = np.sqrt((seg_j.to_hit.x - seg_i.from_hit.x) ** 2 + (seg_j.to_hit.y - seg_i.from_hit.y) ** 2 + (
                    seg_j.to_hit.z - seg_i.from_hit.z) ** 2)
            r_cb = np.sqrt((seg_j.from_hit.x - seg_i.to_hit.x) ** 2 + (seg_j.from_hit.y - seg_i.to_hit.y) ** 2 + (
                    seg_j.from_hit.z - seg_i.to_hit.z) ** 2)

            s_ab = s_bc = 1
            if seg_i.to_hit == seg_j.from_hit:
                s_bc = -1
            if seg_i.from_hit == seg_j.to_hit:
                s_ab = -1

            # Define the values for theta
            hit_from_i = seg_i.from_hit
            hit_from_j = seg_j.from_hit
            hit_to_i = seg_i.to_hit
            hit_to_j = seg_j.to_hit
            if i != j:
                if (hit_from_i == hit_to_j) or (hit_from_j == hit_to_i):
                    vect_i = seg_i.to_vect()
                    vect_j = seg_j.to_vect()
                    cosine = np.dot(vect_i,vect_j)/(np.linalg.norm(vect_i)*np.linalg.norm(vect_j))

                    # Define the values for cosine
                    eps = 1e-6

                    if np.abs(cosine-1) < eps:
                        cos_val = 1.0
                        theta = 0.0
                    else:
                        theta = np.arccos(cosine)
                        cos_val = np.cos(lambda_val * theta * (r_ab + r_bc))
                        print(theta)
                    

                    # Add terms to the matrices
                    A[i, j] = -0.5 * cos_val * s_ab * s_bc / (r_ab + r_bc)
                    A[j, i] = A[i, j]
                    if r_ac != 0 and r_cb != 0:
                        b[i] += alpha * s_ab * s_bc / (r_ac * r_cb)
                        b[j] += alpha * s_ab * s_bc / (r_ac * r_cb)



    sum_ab = sum([seg.to_hit.module_id == 1 for seg in segments])
    A -= beta * (np.sum(A) - N) ** 2 / N ** 2
    b += beta * (np.sum([seg.to_hit.module_id == 1 for seg in segments]) - sum_ab * N / len(segments))

    components = {'A': A}

    return A, b, components, segments
