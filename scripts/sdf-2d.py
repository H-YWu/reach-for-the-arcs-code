from ncontext import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps
import polyscope.imgui as psim
import math

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

def spheres_tangent(center1, radius1, center2, radius2, epsilon=1e-9):
    distance = math.sqrt(math.pow(center1[0]-center2[0], 2) + math.pow(center1[1]-center2[1], 2))
    if abs(distance - (radius1 + radius2)) < epsilon:
        return True, "externally tangent"
    elif abs(distance - abs(radius1 - radius2)) < epsilon:
        return True, "internally tangent"
    else:
        return False, None

def spheres_statistics():
    global num_spheres, num_tangent_pairs, num_overlap_pairs, num_in_out_tangent_pairs, num_contained_spheres

    num_spheres = U_ori.shape[0] 
    is_contained = [False for _ in range(num_spheres)]
    num_tangent_pairs = 0
    num_overlap_pairs = 0
    num_in_out_tangent_pairs = 0

    for i, (centeri, si) in enumerate(zip(U_ori, S_ori)):
        for j, (centerj, sj) in enumerate(zip(U_ori, S_ori)):
            if j >= i:
                break
            tangent, msg = spheres_tangent(centeri, abs(si), centerj, abs(sj))
            if tangent == True:
                num_tangent_pairs += 1
                if msg == "internally tangent":
                    if abs(si) < abs(sj):
                        is_contained[i] = True
                    else:
                        is_contained[j] = True
                    num_overlap_pairs += 1
                elif si * sj < 0:
                    num_in_out_tangent_pairs = 1
 
    num_contained_spheres = sum(is_contained)

def closest_distance(center1, radius1, center2, radius2):
    center_dist = math.sqrt(math.pow(center1[0]-center2[0], 2) + math.pow(center1[1]-center2[1], 2))
    closest_dist = center_dist - (radius1 + radius2)
    return closest_dist

def spheres_distance():
    global min_closest_distance_spheres
    global U_f, S_f

    num_fs = U_f.shape[0] 
    min_closest_distance_spheres = [float('inf') for _ in range(num_fs)]

    for i, (centeri, si) in enumerate(zip(U_f, S_f)):
        for j, (centerj, sj) in enumerate(zip(U_f, S_f)):
            if j >= i:
                break
            if si * sj < 0:
                continue
            closest_dist = closest_distance(centeri, abs(si), centerj, abs(sj))
            if closest_dist < min_closest_distance_spheres[i]:
                min_closest_distance_spheres[i] = closest_dist
            if closest_dist < min_closest_distance_spheres[j]:
                min_closest_distance_spheres[j] = closest_dist

def set_current_box():
    global bbox_min, bbox_max
    global cur_bbox_min, cur_bbox_max
    global bbox_dis
    global cur_per

    cur_bbox_min = bbox_min - bbox_dis * cur_per
    cur_bbox_max = bbox_max + bbox_dis * cur_per

def ground_truth_polygon_from_png(filepath):
    global V_gt, F_gt
    global bbox_min, bbox_max
    global bbox_dis

    poly_list = gpy.png2poly(filepath)
    # Downsample polygon 
    V_gt = None
    F_gt = None
    for poly in poly_list:
        nv = 0 if V_gt is None else V_gt.shape[0]
        pV = poly[::5,:]
        V_gt = pV if V_gt is None else np.concatenate((V_gt, pV), axis=0)
        F_gt = gpy.edge_indices(pV.shape[0],closed=True) if F_gt is None else \
            np.concatenate((F_gt, nv+gpy.edge_indices(pV.shape[0],closed=True)),
                axis=0)
    V_gt = gpy.normalize_points(V_gt)
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

def create_sdf():
    global n, sdf, V_gt, F_gt, U_ori, S_ori
    global min_out_filter_radius, max_out_filter_radius, min_out_radius, max_out_radius
    global min_in_filter_radius, max_in_filter_radius, min_in_radius, max_in_radius
    global cur_bbox_min, cur_bbox_max
    global filter_out_per, min_filter_out_per, max_filter_out_per
    global filter_in_per, min_filter_in_per, max_filter_in_per

    sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]
    # Set up a grid
    gx, gy = np.meshgrid(np.linspace(cur_bbox_min, cur_bbox_max, n + 1), np.linspace(cur_bbox_min, cur_bbox_max, n + 1))
    U_ori = np.vstack((gx.flatten(), gy.flatten())).T
    S_ori = sdf(U_ori)

    out_S = S_ori[S_ori>0] 
    min_out_radius = np.min(out_S)
    max_out_radius = np.max(out_S)
    min_out_filter_radius = min_out_radius
    #max_out_filter_radius = max_out_radius
    min_filter_out_per = 0.0
    max_filter_out_per = 1.0 
    filter_out_per = 0.7 / math.sqrt(n) 
    max_out_filter_radius = min_out_radius + filter_out_per * (max_out_radius - min_out_radius)
    in_S = S_ori[S_ori<0] 
    min_in_radius = np.min(np.abs(in_S))
    max_in_radius = np.max(np.abs(in_S))
    min_in_filter_radius = min_in_radius
    #max_in_filter_radius = max_in_radius
    min_filter_in_per = 0.0
    max_filter_in_per = 1.0 
    filter_in_per = 1.0 
    max_in_filter_radius = min_in_radius + filter_in_per * (max_in_radius - min_in_radius)

def create_filtered_sdf():
    global U_f, S_f, U, S
    global min_out_filter_radius, max_out_filter_radius, min_in_filter_radius, max_in_filter_radius

    valid_indices = np.where(
        ((S_ori > 0) & (S_ori >= min_out_filter_radius) & (S_ori <= max_out_filter_radius)) | 
        ((S_ori < 0) & (S_ori <= -min_in_filter_radius) & (S_ori >= -max_in_filter_radius))
    )
    U_f = U_ori[valid_indices]
    S_f = S_ori[valid_indices]

def create_voronoi_diagram():
    global V_vdo, V_vdi, V_vc, E_vdo, E_vdi, E_vc
    global U_f, S_f

    Sv_f = np.array([1.0 if s > 0.0 else -1.0 for s in S_f])

    V_vdo, V_vdi, V_vc, E_vdo, E_vdi, E_vc = nrfta.power_diagram(U_f, Sv_f)

def create_power_diagram():
    global V_pdo, V_pdi, V_pc, E_pdo, E_pdi, E_pc
    global U_f, S_f

    V_pdo, V_pdi, V_pc, E_pdo, E_pdi, E_pc = nrfta.power_diagram(U_f, S_f)

def create_marching_cubes():
    global V_mc, F_mc, n, U_ori, S_ori

    V_mc, F_mc = gpy.marching_squares(S_ori, U_ori, n+1, n+1)

def create_filtered_rfta():
    global V_rfta, F_rfta
    global U_f, S_f

    V_rfta, F_rfta = rfta.reach_for_the_arcs(U_f, S_f, verbose=False, parallel=True,
        fine_tune_iters=10)

def rfts_from_power_crust():
    global U_ori, sdf
    global V_pc, E_pc, V_rfts, F_rfts

    V_rfts, F_rfts = gpy.reach_for_the_spheres(U_ori, sdf, V_pc, E_pc)

def create_filtered_rfts(imesh=0):
    global U_f, sdf
    global V_rfts, F_rfts
    global V_mc, F_mc, V_pc, E_pc

    # imesh == 0
    V0, F0 = gpy.regular_circle_polyline(12)
    if imesh == 1:
        V0 = V_mc
        F0 = F_mc
    elif imesh == 2:
        V0 = V_pc
        F0 = E_pc

    V_rfts, F_rfts = gpy.reach_for_the_spheres(U_f, sdf, V0, F0)

def visualize():
    global min_out_filter_radius, max_out_filter_radius
    global U_ori, S_ori, U_f, S_f, V_gt, F_gt, P_pos, Pf, Pf_pos, Pf_neg, N, N_pos, N_neg
    global current_step, fine_tune_current_iter
    global V_pdo, V_pdi, V_pc, E_pdo, E_pdi, E_pc
    global E_unit_sphere, show_spheres
    global min_closest_distance_spheres, show_thin
    global V_mc, F_mc
    global show_voro, V_vdo, V_vdi, V_vc, E_vdo, E_vdi, E_vc
    global show_rfta, V_rfta, F_rfta
    global show_rfts, V_rfts, F_rfts

    ps.remove_all_structures()
    ps.register_curve_network("Ground Truth (Rotated)", V_gt, F_gt, radius=0.001, color=PURPLE)
    ps.register_point_cloud(f"Grid Points", U_ori, radius=0.0017, color=YELLOW)

    if current_step == 0:
        if show_spheres:
            for i, (center, s) in enumerate(zip(U_f, S_f)):
                radius = np.abs(s)
                scolor = RED if s > 0 else BLUE
                show_radius = 0.0005
                if show_thin and min_closest_distance_spheres[i] > 0.0:
                    show_radius = 0.002
                ps.register_curve_network(f"SDF Sphere {i}", adjust_sphere_V(center, radius), E_unit_sphere, radius=show_radius, color=scolor)

        ps.register_curve_network("Power Diagram Out", V_pdo, E_pdo, radius=0.001, color=GREEN)
        ps.register_curve_network("Power Diagram In", V_pdi, E_pdi, radius=0.001, color=YELLOW)
        ps.register_curve_network("Power Crust", V_pc, E_pc, radius=0.002, color=BLACK)
        ps.register_curve_network("Marching Cubes", V_mc, F_mc, radius=0.002, color=RED, enabled=False)
        if show_voro:
            ps.register_curve_network("Voronoi Crust", V_vc, E_vc, radius=0.002, color=BLUE)
        if show_rfta:
            ps.register_curve_network("Reach For the Arcs", V_rfta, F_rfta, radius=0.002, color=RED)
        if show_rfts:
            ps.register_curve_network("Reach For the Spheres", V_rfts, F_rfts, radius=0.002, color=GREEN)

    if current_step == 1:
        if not (P_pos is None or P_pos.size==0):
            ps.register_point_cloud("(Positive) Sampled Points", scale*P_pos + trans[None,:], radius=0.003, color=LIGHTRED)
            ps_pf_pos = ps.register_point_cloud("(Positive) Feasible Points", scale*Pf_pos + trans[None,:], radius=0.003, color=RED)
            ps_pf_pos.add_vector_quantity("(Positive) Normals of Feasible Points", N_pos/math.pow(scale, 3), enabled=True, vectortype='ambient', color=ORANGE)
        if not (P_neg is None or P_neg.size==0):
            ps.register_point_cloud("(Inner) Sampled Points", scale*P_neg + trans[None,:], radius=0.003, color=LIGHTBLUE)
            ps_pf_neg = ps.register_point_cloud("(Inner) Feasible Points", scale*Pf_neg + trans[None,:], radius=0.003, color=BLUE)
            ps_pf_neg.add_vector_quantity("(Inner) Normals of Feasible Points", N_neg /math.pow(scale, 3), enabled=True, vectortype='ambient', color=PURPLE)
    if current_step == 2:
        ps.register_curve_network("Current Reconstruction", scale*V_rfta[fine_tune_current_iter-1] + trans[None,:], F_rfta[fine_tune_current_iter-1], radius=0.002, color=BLACK)
        ps_pf = ps.register_point_cloud("Feasible Points", scale*Pf[fine_tune_current_iter-1] + trans[None,:], radius=0.003, color=PURPLE)
        ps_pf.add_vector_quantity("Normals of Feasible Points", N[fine_tune_current_iter-1]/math.pow(scale, 3), enabled=True, vectortype='ambient', color=MAGENTA)
    if current_step == 3:
        ps.register_curve_network("Reconstruction", scale*V_rfta[fine_tune_current_iter] + trans[None,:], F_rfta[fine_tune_current_iter], radius=0.002, color=BLACK)

def step_forward(tol=1e-4):
    global current_step, total_step, fine_tune_current_iter
    global U_ori, S_ori, U, S, P, V_gt, F_gt, V_rfta, F_rfta, P, P_pos, P_neg, Pf, Pf_pos, Pf_neg, N, N_pos, N_neg, f, f_pos, f_neg
    global fine_tune_iters, batch_size, num_rasterization_spheres, screening_weight, rasterization_resolution, max_points_per_sphere
    global n_local_searches, local_search_iters, local_search_t
    global parallel, rng, seed, n_sdf, trans, scale 

    if current_step == 0:
        rng_seed = 3452
        d = U_ori.shape[1]
        assert d==2 or d==3, "Only dimensions 2 and 3 supported."
        assert max_points_per_sphere>=1, "There has to be at least one point per sphere."

        n_sdf = U_ori.shape[0]

        # Pick default values if not supplied.
        if n_local_searches is None:
            n_local_searches = math.ceil(2. * n_sdf**(1./d))

        # RNG used to compute random numbers and new seeds during the method.
        rng = np.random.default_rng(seed=rng_seed)
        seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

        # Resize the SDF points in U and the SDF samples in S so it's in [0,1]^d
        trans = np.min(U_ori, axis=0)
        U = U_ori - trans[None,:]
        scale = np.max(U)
        U /= scale
        S = S_ori/scale

    if current_step == total_step-1:
        return
    if current_step != 2:
        if current_step == 1:
            fine_tune_current_iter = 0
        current_step = (current_step + 1) % total_step
    if current_step == total_step:
        return
    elif fine_tune_current_iter == fine_tune_iters:
        current_step = (current_step + 1) % total_step
 
    if current_step == 1:
        # Pick default values if not supplied.
        if rasterization_resolution is None:
            rasterization_resolution = 64 * math.ceil(n_sdf**(1./d)/16.)

        # Split the SDF into positive and negative spheres
        neg = S<0
        pos = np.logical_not(neg)
        pos,neg = np.nonzero(pos)[0], np.nonzero(neg)[0]
        U_pos, U_neg = U[pos,:], U[neg,:]
        S_pos, S_neg = S[pos], S[neg]
        if pos.size > 0:
            P_pos,N_pos,f_pos,Pf_pos = nrfta.sdf_to_point_cloud(U_pos, S_pos,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                tol=tol,
                parallel=parallel)
        else:
            P_pos,N_pos,f_pos,Pf_pos = None,None,None,None
        if neg.size>0:
            P_neg,N_neg,f_neg,Pf_neg = nrfta.sdf_to_point_cloud(U_neg, S_neg,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                tol=tol,
                parallel=parallel)
        else:
            P_neg,N_neg,f_neg,Pf_neg = None,None,None,None

        P0 = None
        N0 = None
        f0 = None
        Pf0 = None
        P = [None for _ in range(fine_tune_iters+1)]
        N = [None for _ in range(fine_tune_iters+1)]
        f = [None for _ in range(fine_tune_iters+1)]
        Pf = [None for _ in range(fine_tune_iters+1)]
        V_rfta = [None for _ in range(fine_tune_iters+1)]
        F_rfta = [None for _ in range(fine_tune_iters+1)]

        if P_pos is None or P_pos.size==0:
            P0,N0,f0,Pf0 = P_neg,N_neg,neg[f_neg],Pf_neg
        elif P_neg is None or P_neg.size==0:
            P0,N0,f0,Pf0 = P_pos,N_pos,pos[f_pos],Pf_pos
        else:
            P0 = np.concatenate((P_pos, P_neg), axis=0)
            N0 = np.concatenate((N_pos, N_neg), axis=0)
            f0 = np.concatenate((pos[f_pos], neg[f_neg]), axis=0)
            Pf0 = np.concatenate((Pf_pos, Pf_neg), axis=0)

        if Pf0 is None or Pf0.size==0:
            ps.warning("Warning: No feasible points found!") 
            reset()
            return
        
        P[0] = P0
        N[0] = N0
        f[0] = f0
        Pf[0] = Pf0

    if current_step == 2:
        # One Iter
        ## Generate a random batch of size batch size.
        if batch_size > 0 and batch_size < n_sdf:
            batch = rng.choice(n_sdf, batch_size)
        else:
            batch = np.arange(n_sdf)
            rng.shuffle(batch)

        V_rfta[fine_tune_current_iter], F_rfta[fine_tune_current_iter] = nrfta.point_cloud_to_mesh(Pf[fine_tune_current_iter], N[fine_tune_current_iter],
            screening_weight=screening_weight,
            outer_boundary_type="Neumann",
            parallel=False)

        if V_rfta[fine_tune_current_iter].size == 0:
            ps.warning("Warning: No mesh generated!") 
            reset()
            return

        Pf[fine_tune_current_iter+1], N[fine_tune_current_iter+1], f[fine_tune_current_iter+1] = nrfta.fine_tune_point_cloud_iteration(U,
            S,
            V_rfta[fine_tune_current_iter],
            F_rfta[fine_tune_current_iter],
            Pf[fine_tune_current_iter],
            N[fine_tune_current_iter],
            f[fine_tune_current_iter],
            batch,
            max_points_per_sphere,
            seed(),
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, parallel)
 
        fine_tune_current_iter += 1
 
    if current_step == 3:
        V_rfta[fine_tune_current_iter],F_rfta[fine_tune_current_iter] = nrfta.point_cloud_to_mesh(Pf[fine_tune_current_iter], N[fine_tune_current_iter],
            screening_weight=screening_weight,
            outer_boundary_type="Neumann",
            parallel=False) # disabled parallelization here because it ocassionally crashes (Misha's version)

        if V_rfta[fine_tune_current_iter] is None or V_rfta[fine_tune_current_iter].size==0:
            ps.warning("Warning: No mesh generated!") 
            reset()

def reset():
    global current_step, fine_tune_current_iter
    current_step = 0 
    fine_tune_current_iter = -1

def change_list_size(lst, new_size):
    if lst == None:
        return
    current_size = len(lst)
    if new_size > current_size:
        lst.extend([None] * (new_size - current_size))
    elif new_size < current_size:
        del lst[new_size:]

def callback():
    global png_files_selected, png_selected_index, n, cur_per, show_spheres, show_thin
    global V_gt, F_gt, sdf
    global min_out_filter_radius, max_out_filter_radius, min_out_radius, max_out_radius
    global filter_out_per, min_filter_out_per, max_filter_out_per
    global filter_in_per, min_filter_in_per, max_filter_in_per
    global min_in_filter_radius, max_in_filter_radius, min_in_radius, max_in_radius
    global num_spheres, num_tangent_pairs, num_overlap_pairs, num_in_out_tangent_pairs, num_contained_spheres
    global current_step, step_names, fine_tune_current_iter, fine_tune_iters
    global show_voro, show_rfta, show_rfts
 
    psim.PushItemWidth(300)

    if psim.Button("Reset"):
        reset()
        visualize()
    if psim.Button("Last Step"):
        if current_step > 0:
            if current_step == 3:
                fine_tune_current_iter = fine_tune_iters
            if current_step == 2:
                if fine_tune_current_iter == 1:
                    current_step -= 1
                else:
                    fine_tune_current_iter -= 1
            else:
                current_step -= 1
        if current_step == 0:
            reset()
        visualize()
    psim.SameLine()
    if psim.Button("Next Step"):
        step_forward()
        visualize()
    psim.SameLine()
    psim.TextUnformatted(f"{step_names[current_step]}")
    if current_step == 2:
        psim.SameLine()
        psim.TextUnformatted(f": Iter {fine_tune_current_iter}")
    
    psim.Separator()
    num_widges = 10 
    changed = [False for _ in range(num_widges)]

    changed[0] = psim.BeginCombo("Input Shape", png_files_selected)
    if changed[0]:
        for i, val in enumerate(png_files):
            _, selected = psim.Selectable(val, png_files_selected==val)
            if selected:
                png_files_selected = val
                png_selected_index = i
        psim.EndCombo()
        ground_truth_polygon_from_png(png_paths[png_selected_index])
        # Create and abstract SDF function that is the only connection to the shape

    changed[1], n = psim.SliderInt("Grid Resolution", n, v_min=5, v_max=100)

    psim.TextUnformatted("Exclude Outer Spheres with Radius: ")
    psim.PushItemWidth(150)
    changed[2], min_out_filter_radius = psim.SliderFloat("below (out)", min_out_filter_radius, v_min=min_out_radius, v_max=max_out_radius) 
    psim.SameLine()
    changed[3], max_out_filter_radius = psim.SliderFloat("above (out)", max_out_filter_radius, v_min=min_out_filter_radius, v_max=max_out_radius)
    psim.PopItemWidth()

    psim.TextUnformatted("Exclude Inner Spheres with Radius: ")
    psim.PushItemWidth(150)
    changed[4], min_in_filter_radius = psim.SliderFloat("below (in)", min_in_filter_radius, v_min=min_in_radius, v_max=max_in_radius) 
    psim.SameLine()
    changed[5], max_in_filter_radius = psim.SliderFloat("above (in)", max_in_filter_radius, v_min=min_in_filter_radius, v_max=max_in_radius)
    psim.PopItemWidth()

    changed[6], fine_tune_iters = psim.SliderInt("Fine-Tune Iterations", fine_tune_iters, v_min=max(1, fine_tune_current_iter), v_max=50)
    if changed[6]:
        change_list_size(P, fine_tune_iters+1)
        change_list_size(N, fine_tune_iters+1)
        change_list_size(f, fine_tune_iters+1)
        change_list_size(Pf, fine_tune_iters+1)
        change_list_size(V_rfta, fine_tune_iters+1)
        change_list_size(F_rfta, fine_tune_iters+1)
    
    changed[7], cur_per = psim.SliderFloat("SDF Enlarged Percentage", cur_per, v_min=min_per, v_max=max_per)
    if changed[7]:
        set_current_box()

    psim.TextUnformatted("Exclude Spheres with Radius Greater Then the Minimum by Percentage: ")
    psim.PushItemWidth(150)
    changed[8], filter_out_per = psim.SliderFloat("Out", filter_out_per , v_min=min_filter_out_per, v_max=max_filter_out_per)
    if changed[8]:
        min_out_filter_radius = min_out_radius
        max_out_filter_radius = min_out_radius + filter_out_per * (max_out_radius - min_out_radius)
    psim.SameLine()
    changed[9], filter_in_per = psim.SliderFloat("In", filter_in_per , v_min=min_filter_in_per, v_max=max_filter_in_per)
    if changed[9]:
        min_in_filter_radius = min_in_radius
        max_in_filter_radius = min_in_radius + filter_in_per * (max_in_radius - min_in_radius)
    psim.PopItemWidth()
 
    show_voro = False
    if psim.Button("Voronoi crust"):
        create_voronoi_diagram()
        show_voro = True
    show_rfta = False
    if psim.Button("RFTA"):
        create_filtered_rfta()
        show_rfta = True
    show_rfts = False 
    if psim.Button("RFTS (Sphere)"):
        create_filtered_rfts(0)
        show_rfts = True
    psim.SameLine()
    if psim.Button("RFTS (Marching Cube)"):
        create_filtered_rfts(1)
        show_rfts = True
    psim.SameLine()
    if psim.Button("RFTS (Power Crust)"):
        create_filtered_rfts(2)
        show_rfts = True
 
    if changed[2] or changed[3] or changed[4] or changed[5] or changed[8] or changed[9]:
        create_filtered_sdf()
        create_power_diagram()

    if changed[0] or changed[1] or changed[7]:
        create_sdf()
        create_filtered_sdf()
        create_power_diagram()
        create_marching_cubes()
        reset()
    
    last_show_sphere = show_spheres
    if psim.Button("Show Spheres"):
        show_spheres = True
    psim.SameLine()
    if psim.Button("Hide Spheres"):
        show_spheres = False
 
    show_thin = False
    if psim.Button("Guess Thin Parts"):
        spheres_distance()
        show_thin = True

    if any(changed) or last_show_sphere != show_spheres or show_thin or show_voro or show_rfta or show_rfts:
        visualize()

    psim.Separator()
    psim.TextUnformatted(f"#Spheres: {num_spheres}")
    psim.TextUnformatted(f"#Tangent Pairs: {num_tangent_pairs}")
    psim.TextUnformatted(f"#Overlapped Pairs: {num_overlap_pairs}")
    psim.TextUnformatted(f"#Inside-Outside Tangent Pairs: {num_in_out_tangent_pairs}")
    psim.TextUnformatted(f"#Spheres Contained in Other Spheres: {num_contained_spheres}")

    psim.PopItemWidth()


# Command line arguments
parser = argparse.ArgumentParser(description='SDF 2D Test Framework.')
parser.add_argument('file_name', type=str, nargs='?', default=None, help='The file name to process')
args = parser.parse_args()

# Parameters
## Selectable input shapes
data_dir = 'data/'
all_files = os.listdir(data_dir)
### Used for options
png_files = [file for file in all_files if file.lower().endswith('.png')]
if args.file_name and args.file_name in png_files:
    png_selected_index = png_files.index(args.file_name)
else:
    png_selected_index = 0
png_files_selected = png_files[png_selected_index]
### Used for locate
png_paths = [os.path.join(data_dir, png_file) for png_file in png_files]
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
## Customized Statistics
num_spheres = 0
num_tangent_pairs = 0
num_overlap_pairs = 0
num_in_out_tangent_pairs = 0
num_contained_spheres = 0
## Algorithm (rfta)
current_step = 0
total_step = 4
step_names = ["Visualize SDF", "Intialize Feasible Points", "Fine-Tune Point Cloud", "Point Cloud to Mesh"]
fine_tune_current_iter = -1 
fine_tune_iters = 10
batch_size = 10000
num_rasterization_spheres = 0
screening_weight = 10.
rasterization_resolution = None
max_points_per_sphere = 3
n_local_searches = None
local_search_iters = 20
local_search_t = 0.01
parallel = False

show_spheres = False
show_thin = False
show_voro = False
show_rfta = False
show_rfts = False

# Setup 2D polyscope
ps.init()
ps.set_navigation_style("planar")

generate_unit_circle()
ground_truth_polygon_from_png(png_paths[png_selected_index])
create_sdf()
create_filtered_sdf()
create_power_diagram()
create_marching_cubes()
visualize()

ps.set_user_callback(callback)
ps.show()
