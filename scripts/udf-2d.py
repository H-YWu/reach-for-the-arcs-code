from ncontext import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps
import polyscope.imgui as psim
import math

# V_gt = np.array([[1., 1.], [5., 1.], [5., 5.], [1., 5.]])    
# F_gt = np.array([[0, 1], [1, 2], [2, 3]])

# Colors
RED = [1.0, 0.0, 0.0]
LIGHTRED = [1.0, 0.745, 0.773]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
LIGHTBLUE = [0.698, 0.698, 1.0]
YELLOW = [1.0, 1.0, 0.0]
CYAN = [0.0, 1.0, 1.0]
MAGENTA = [1.0, 0.0, 1.0]
BLACK = [0.0, 0.0, 0.0]
ORANGE = [1.0, 0.5, 0.0]
PURPLE = [0.5, 0.0, 0.5]

def generate_unit_circle(num_points=30):
    global V_unit_sphere, E_unit_sphere

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    V_unit_sphere = np.array([[np.cos(t), np.sin(t)] for t in theta])
    E_unit_sphere = np.array([(i, (i + 1) % num_points) for i in range(num_points)])

def adjust_sphere_V(center, radius):
    global V_unit_sphere

    return center + radius * V_unit_sphere 

def set_current_box():
    global bbox_min, bbox_max
    global cur_bbox_min, cur_bbox_max
    global bbox_dis
    global cur_per

    cur_bbox_min = bbox_min - bbox_dis * cur_per
    cur_bbox_max = bbox_max + bbox_dis * cur_per

def ground_truth_from_npy(filepath):
    global V_gt, F_gt
    global bbox_min, bbox_max
    global bbox_dis

    V_gt, F_gt = utility.read_mesh(filepath)
    # Randomly rotate the polygon
    rng_seed = 3523
    np.random.seed(rng_seed)
    angle = np.random.rand() * 2 * np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    V_gt = V_gt @ R 

    # Bounding box for SDF
    bbox_min_x = np.min(V_gt[:, 0])
    bbox_max_x = np.max(V_gt[:, 0])
    bbox_min_y = np.min(V_gt[:, 1])
    bbox_max_y = np.max(V_gt[:, 1])
    bbox_min = min(bbox_min_x, bbox_min_y)
    bbox_max = max(bbox_max_x, bbox_max_y)
    bbox_dis = bbox_max - bbox_min 
    set_current_box()

def create_udf():
    global V_gt, F_gt
    global n, U, D
    global cur_bbox_min, cur_bbox_max
    global min_filter_radius, max_filter_radius, min_radius, max_radius
    global filter_per, min_filter_per, max_filter_per

    # Set up a grid
    gx, gy = np.meshgrid(np.linspace(cur_bbox_min, cur_bbox_max, n + 1), np.linspace(cur_bbox_min, cur_bbox_max, n + 1))
    U = np.vstack((gx.flatten(), gy.flatten())).T
    D = nrfta.unsigned_distance_field(U, V_gt, F_gt)

    min_radius = np.min(D)
    max_radius = np.max(D)
    min_filter_radius = min_radius
    max_filter_radius = max_radius
    min_filter_per = 0.0
    max_filter_per = (max_radius - min_radius) / min_radius
    filter_per = max_filter_per

def create_filtered_udf():
    global U_f, D_f, U, D
    global min_filter_radius, max_filter_radius

    valid_indices = np.where(
        (D >= min_filter_radius) & (D <= max_filter_radius)
    )
    U_f = U[valid_indices]
    D_f = D[valid_indices]

def create_power_diagram():
    global V_pd, E_pd
    global U_f, D_f

    V_pd, _, _, E_pd, _, _ = nrfta.power_diagram(U_f, D_f)

def compute_power_vertex_udf():
    global V_pd, V_pd_f, V_gt, F_gt, D_pd, D_pd_f
    global min_filter_pd_radius, max_filter_pd_radius, min_pd_radius, max_pd_radius

    valid_mask = ~np.isnan(V_pd).any(axis=1)
    V_pd_f = V_pd[valid_mask]
    D_pd = nrfta.unsigned_distance_field(V_pd, V_gt, F_gt)
    valid_mask = ~np.isnan(D_pd)
    D_pd_f = D_pd[valid_mask]

    min_pd_radius = np.min(D_pd_f)
    max_pd_radius = np.max(D_pd_f)
    min_filter_pd_radius = min_pd_radius
    max_filter_pd_radius = min(max_pd_radius, min_pd_radius + 0.02 * (max_pd_radius - min_pd_radius))

def visualize():
    global min_filter_radius, max_filter_radius
    global U, D, U_f, D_f, D_pd, D_pd_f, V_gt, F_gt, V_pd, V_pd_f, E_pd
    global E_unit_sphere, show_spheres
    global min_filter_pd_radius, max_filter_pd_radius

    ps.remove_all_structures()
    ps.register_curve_network("Ground Truth (Rotated)", V_gt, F_gt, radius=0.001, color=PURPLE)
    ps.register_point_cloud(f"Grid Points", U, radius=0.0017, color=YELLOW)

    if show_spheres:
        for i, (center, radius) in enumerate(zip(U_f, D_f)):
            ps.register_curve_network(f"UDF Sphere {i}", adjust_sphere_V(center, radius), E_unit_sphere, radius=0.0005, color=RED)

    pd_curve = ps.register_curve_network("Power Diagram", V_pd, E_pd, radius=0.001, color=BLACK)
    Vpdf_colors = np.array([BLACK for _ in range(len(V_pd_f))])
    for i, d in enumerate(D_pd_f):
        if min_filter_pd_radius <= d <= max_filter_pd_radius:
            Vpdf_colors[i] = BLUE 
    pv_cloud = ps.register_point_cloud("Power Vertices", V_pd_f)
    pv_cloud.add_color_quantity("Power Vertices Color", Vpdf_colors, enabled=True)
    Epd_colors = np.array([BLACK for _ in range(np.shape(E_pd)[0])])
    for i, e in enumerate(E_pd):
        if (
            (min_filter_pd_radius <= D_pd[e[0]] <= max_filter_pd_radius) and
            (min_filter_pd_radius <= D_pd[e[1]] <= max_filter_pd_radius)
        ): 
            Epd_colors[i] = ORANGE 
    pd_curve.add_color_quantity("Power Faces Color", Epd_colors, defined_on='edges', enabled=True)

def callback():
    global npy_files_selected, npy_selected_index, n, cur_per, show_spheres
    global V_gt, F_gt
    global min_filter_radius, max_filter_radius, min_radius, max_radius
    global filter_per, min_filter_per, max_filter_per
    global min_filter_pd_radius, max_filter_pd_radius, min_pd_radius, max_pd_radius
 
    psim.PushItemWidth(300)

    num_widges = 8
    changed = [False for _ in range(num_widges)]

    changed[0] = psim.BeginCombo("Input Shape", npy_files_selected)
    if changed[0]:
        for i, val in enumerate(npy_files):
            _, selected = psim.Selectable(val, npy_files_selected==val)
            if selected:
                npy_files_selected = val
                npy_selected_index = i
        psim.EndCombo()
        ground_truth_from_npy(npy_paths[npy_selected_index])

    changed[1], n = psim.SliderInt("Grid Resolution", n, v_min=5, v_max=100)

    psim.TextUnformatted("Exclude Spheres with Radius: ")
    psim.PushItemWidth(100)
    changed[2], min_filter_radius = psim.SliderFloat("below (sphere)", min_filter_radius, v_min=min_radius, v_max=max_radius) 
    psim.SameLine()
    changed[3], max_filter_radius = psim.SliderFloat("above (sphere)", max_filter_radius, v_min=min_filter_radius, v_max=max_radius)
    psim.PopItemWidth()
 
    changed[4], cur_per = psim.SliderFloat("SDF Enlarged Percentage", cur_per, v_min=min_per, v_max=max_per)
    if changed[4]:
        set_current_box()

    psim.TextUnformatted("Exclude Spheres with Radius Greater Then the Minimum by Percentage: ")
    changed[5], filter_per = psim.SliderFloat("%", filter_per , v_min=min_filter_per, v_max=max_filter_per)
    if changed[5]:
        min_filter_radius = min_radius
        max_filter_radius = min_radius * (1.0 + filter_per)
    
    psim.TextUnformatted("Exclude Power Vertices with Radius: ")
    psim.PushItemWidth(100)
    changed[6], min_filter_pd_radius = psim.SliderFloat("below (vertex)", min_filter_pd_radius, v_min=min_pd_radius, v_max=max_pd_radius) 
    psim.SameLine()
    changed[7], max_filter_pd_radius = psim.SliderFloat("above (vertex)", max_filter_pd_radius, v_min=min_filter_pd_radius, v_max=max_pd_radius)
    psim.PopItemWidth()
 
    if changed[2] or changed[3] or changed[5]:
        create_filtered_udf()
        create_power_diagram()
        compute_power_vertex_udf()

    if changed[0] or changed[1] or changed[4]:
        create_udf()
        create_filtered_udf()
        create_power_diagram()
        compute_power_vertex_udf()
 
    last_show_sphere = show_spheres
    if psim.Button("Show Spheres"):
        show_spheres = True
    psim.SameLine()
    if psim.Button("Hide Spheres"):
        show_spheres = False

    if any(changed) or last_show_sphere != show_spheres:
        visualize()

    psim.PopItemWidth()

# Command line arguments
parser = argparse.ArgumentParser(description='UDF 2D Test Framework.')
parser.add_argument('file_name', type=str, nargs='?', default=None, help='The file name to process')
args = parser.parse_args()

# Parameters
## Selectable input shapes
data_dir = 'data_open/'
all_files = os.listdir(data_dir)
### Used for options
npy_files = [file for file in all_files if file.lower().endswith('.npy')]
if args.file_name and args.file_name in npy_files:
    npy_selected_index = npy_files.index(args.file_name)
else:
    npy_selected_index = 0
npy_files_selected = npy_files[npy_selected_index]
### Used for locate
npy_paths = [os.path.join(data_dir, npy_file) for npy_file in npy_files]
## Grid resolution: nxn
n = 10
## Bounding box size for SDF
### Fit
bbox_min = -1.0 
bbox_max = 1.0
### Current
cur_bbox_min = -1.0 
cur_bbox_max = 1.0 
### Fit box size
box_dis = 2.0 
### Percentage for controlling
min_per = 0.005
max_per = 1.0
cur_per = 0.3

show_spheres = False

# Setup 2D polyscope
ps.init()
ps.set_navigation_style("planar")

generate_unit_circle()
ground_truth_from_npy(npy_paths[npy_selected_index])
create_udf()
create_filtered_udf()
create_power_diagram()
compute_power_vertex_udf()
visualize()

ps.set_user_callback(callback)
ps.show()
