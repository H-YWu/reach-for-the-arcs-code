from ncontext import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

input_dir = "data/"
output_dir = "data_open/"
filename = "switzerland"
remove_edges_num = 200

poly_list = gpy.png2poly(input_dir + filename + ".png")

# Downsample polygon
V = None
F = None
for poly in poly_list:
    nv = 0 if V is None else V.shape[0]
    pV = poly[::5, :]
    V = pV if V is None else np.concatenate((V, pV), axis=0)
    F = gpy.edge_indices(pV.shape[0], closed=True) if F is None else \
        np.concatenate((F, nv + gpy.edge_indices(pV.shape[0], closed=True)), axis=0)
V = gpy.normalize_points(V)

# Indices of edges to remove
indices_to_remove = range(remove_edges_num) 
F = np.delete(F, indices_to_remove, axis=0)
# Find vertices that are no longer connected to any edge
vertices_to_keep = np.unique(F)
all_vertices = np.arange(V.shape[0])
vertices_to_remove = np.setdiff1d(all_vertices, vertices_to_keep)
# Remove unused vertices
V = np.delete(V, vertices_to_remove, axis=0)
# Update F_gt to reflect the new indices of V_gt
vertex_mapping = np.zeros(V.shape[0] + len(vertices_to_remove), dtype=int)
vertex_mapping[vertices_to_keep] = np.arange(V.shape[0])
F = vertex_mapping[F]

utility.write_mesh(output_dir + filename, V, F)

V_t, F_t = utility.read_mesh(output_dir + filename + ".npy")

ps.init()
ps.set_navigation_style("planar")
ps.register_curve_network("Ground Truth", V_t, F_t, radius=0.001)
ps.show()
