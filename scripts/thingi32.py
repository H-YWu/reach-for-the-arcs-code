from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps
import time
import csv
from multiprocessing import Process, Queue, TimeoutError
from collections import defaultdict
import pandas as pd
import seaborn as sns
import os

global path_to_all_meshes, resolutions, methods, metrics, path_to_results_csv, error_sums, shape_counts, timing_sums, selected_mesh_files, num_shapes, timeout_duration

parser = argparse.ArgumentParser(description='Thingi32 experiment.')
parser.add_argument('--num_shapes', type=int, default=100, help='number of shapes to use')
parser.add_argument('--run', action=argparse.BooleanOptionalAction)
parser.set_defaults(run=False)
parser.add_argument('--reload', action=argparse.BooleanOptionalAction)
parser.set_defaults(reload=False)
parser.add_argument('--metrics', action=argparse.BooleanOptionalAction)
parser.set_defaults(metrics=False)
parser.add_argument('--table', action=argparse.BooleanOptionalAction)
parser.set_defaults(table=False)
args = parser.parse_args()

# number of shapes
num_shapes = 100

# rng
rng_seed = 1
np.random.seed(rng_seed)
random.seed(rng_seed)

# methods
methods = [ "rfts", "rfta" ]

# metrics
# metrics = [ "chf", "hd", "sdf" ]
metrics = [ "chf" ]

# path to meshes
# Compared to VoroMesh
path_to_all_meshes = "data/Thingi32/obj/"
# Get all mesh files
all_mesh_files = os.listdir(path_to_all_meshes)
# Shuffle the list
random.shuffle(all_mesh_files)

if args.reload:
    selected_mesh_files = all_mesh_files[:num_shapes]
else:
    # otherwise, just get a list of all the folders of the type "results/Thingi32/{mesh}"
    selected_mesh_files = os.listdir("results/Thingi32")
    # only the ones that are subdirectories
    selected_mesh_files = [mesh for mesh in selected_mesh_files if os.path.isdir(f"results/Thingi32/{mesh}")]

path_to_results = "results/Thingi32/"

# path to results csv
path_to_results_csv = "results/Thingi32/results.csv"
path_to_avg_results_csv = "results/Thingi32/avg_results.csv"

# resolutions
resolutions = [ 32 ]

# timeout duration
timeout_duration = 1000

# now loop over every mesh, every resolution, every method, and add row with error metrics to a CSV file
# CSV file will have columns: mesh, resolution, method, metric, error

def is_mesh_broken(V, F):
    return np.any(np.isnan(V)) or np.any(np.isnan(F)) or np.any(np.isinf(V)) or np.any(np.isinf(F)) or np.any(np.isneginf(V)) or np.any(np.isneginf(F)) or F.shape[0] == 0 or V.shape[0] == 0 or F.shape[1] != 3 or V.shape[1] != 3

# auxiliary function to run method
def run_method(U, S, method, savepath, rot_matrix, V_gt, F_gt, queue):
    try:
        if method == "rfta":
            V, F = rfta.reach_for_the_arcs(U, S, parallel=True, max_points_per_sphere=30, fine_tune_iters=20, rng_seed=rng_seed)
        elif method == "rfts":
            # Choose an initial surface for reach_for_the_spheres
            V0, F0 = gpy.icosphere(2)
            # Reconstruct triangle mesh
            V, F = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)
        else:
            queue.put(f"Method {method} not found")
            raise RuntimeError(f"Unknown method {method}")      
        if is_mesh_broken(V, F):
            queue.put(f"Broken or invalid mesh detected")
            raise RuntimeError(f"Broken or invalid mesh detected")
        if savepath is not None:
            gpy.write_mesh(savepath, V @ np.linalg.inv(rot_matrix), F)
    except:
        queue.put(f"Method {method} failed")
        raise RuntimeError(f"Method {method} failed")
    
# def our_energy(U,S,v,f):
#     d2, I, b = gpy.squared_distance(U, v, f, use_cpp=True, use_aabb=True)
#     d = np.sqrt(d2)
#     g = np.abs(S)-d
#     return 1000*np.sum(g**2.0)/U.shape[0]

# auxiliary function to compute metrics
def compute_error(V_gt, F_gt, V, F, U, S, R, metric):
    if metric == "chf":
        return utility.chamfer(V_gt, F_gt, V, F)
    # elif metric == "hd":
    #     return gpy.approximate_hausdorff_distance(V_gt, F_gt, V, F)
    # elif metric == "sdf":
    #     return our_energy(U,S,V @ R,F)
    else:
        assert False, "unknown metric"

# auxiliary function to compute SDF data
def get_sdf_data(V_gt, F_gt, resolution):
    # Create and abstract SDF function that is the only connection to the shape
    sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

    # Set up a grid
    n = resolution
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)
    return U, S

def rotation_matrix(axis, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.

    R = dd^T + cos(theta)*(I - dd^T) + sin(theta)*skew(d)

    Parameters:
    axis : array
        Axis around which to rotate.
    angle : float
        Angle, in radians, by which to rotate.

    Returns:
    numpy.ndarray
        A rotation matrix.
    """

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis vector
    x, y, z = axis

    # Construct the skew-symmetric matrix
    skew_sym = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Outer product of the axis vector with itself
    outer = np.outer(axis, axis)

    # Rotation matrix
    R = outer + np.cos(angle) * (I - outer) + np.sin(angle) * skew_sym

    return R

def generate_latex_table(csv_file, methods, resolutions, metrics):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter the DataFrame based on user specifications
    df_filtered = df[df['method'].isin(methods) & df['resolution'].isin(resolutions) & df['metric'].isin(metrics)]

    # Pivot the DataFrame to get the desired format
    df_pivot = df_filtered.pivot_table(index='resolution', columns=['metric', 'method'], values='error')

    # Reorder columns based on methods_order
    new_columns = [(metric, method) for metric in metrics for method in methods if (metric, method) in df_pivot.columns]
    df_pivot = df_pivot.reindex(columns=new_columns)

    # Generate the LaTeX table
    latex_table = df_pivot.to_latex()

    return latex_table

def generate_combined_latex_table(csv_files, methods_order, resolutions, metrics, captions, method_names, metric_names):
    # Table format with vertical lines
    # table_format = "l" + "|c" * (len(methods_order) * len(metrics))
    table_format = "c" + "".join("|" + "c" * len(methods_order) for _ in metrics)

    # Initialize the header of the LaTeX table
    header_row = "Grid & " + " & ".join(f"{metric} {method}" for metric in metric_names for method in method_names) + " \\\\\n"
    
    # Initialize the LaTeX table with the header and the table format
    latex_table = "\\begin{tabular}{" + table_format + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\rowcolor{tableheader}\n"
    latex_table += header_row
    latex_table += "\\midrule\n"
    
    # Process each CSV file
    for csv_file, caption in zip(csv_files, captions):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Filter by specified resolutions and metrics
        df_filtered = df[df['resolution'].isin(resolutions) & df['metric'].isin(metrics)]
        
        # Pivot the DataFrame to have methods as columns
        df_pivot = df_filtered.pivot(index='resolution', columns=['method', 'metric'], values='error')
        
        # Reorder the columns based on the specified metric order and methods
        method_metric_tuples = [(method, metric) for metric in metrics for method in methods_order]
        df_pivot = df_pivot.reindex(columns=method_metric_tuples)

        # Find the smallest value for each metric and resolution
        min_values = {}
        for metric in metrics:
            for resolution in resolutions:
                min_values[(resolution, metric)] = df_pivot.xs(metric, level='metric', axis=1).loc[resolution].min()

        # Add the section caption
        latex_table += "\\rowcolor{tablesubheader}\n"
        latex_table += "\\multicolumn{" + str(len(methods_order) * len(metrics) + 1) + "}{c}{" + caption + "} \\\\\n"
        latex_table += "\\midrule\n"
        
        # Add the table content with alternating row colors
        for row_index, (index, row) in enumerate(df_pivot.iterrows()):
            row_color = "\\rowcolor[HTML]{EFEFEF}" if row_index % 2 == 1 else ""
            row_values = [f"${index}^3$"]
            for metric in metrics:
                for method in methods_order:
                    value = row[(method, metric)]
                    if pd.notna(value) and value == min_values[(index, metric)]:
                        row_values.append(f"\\textbf{{{value:.4f}}}")
                    else:
                        row_values.append(f"{value:.4f}" if pd.notna(value) else "-")
            latex_table += row_color + " & ".join(map(str, row_values)) + " \\\\\n"
        
        # Add a midrule after each section
        latex_table += "\\midrule\n"

    # Finalize the tabular environment
    # latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    return latex_table


def main():

    if args.run:
        for mesh in sorted(selected_mesh_files):
            mesh_name = mesh.split(".")[0]
            print(f"mesh = {mesh_name}")
            mesh_results_path = f"results/Thingi32/{mesh_name}"
            os.makedirs(f"{mesh_results_path}", exist_ok=True)
 
            if args.reload:
                filename = path_to_all_meshes + mesh
                try:
                    V_gt, F_gt = gpy.read_mesh(filename)
                    assert not is_mesh_broken(V_gt, F_gt)
                except:
                    continue
                # normalize
                V_gt = gpy.normalize_points(V_gt)
                # rotate randomly
                axis = np.random.rand(3)
                axis = axis / np.linalg.norm(axis)
                angle = np.random.rand() * 2 * np.pi
                R = rotation_matrix(axis, angle)
                V_gt = V_gt @ R
                # make it so that the mesh results directory exists
                os.makedirs(mesh_results_path, exist_ok=True)
                # write ground truth mesh
                gpy.write_mesh(f"results/Thingi32/{mesh_name}/ground_truth.obj", V_gt @ np.linalg.inv(R), F_gt)
                # write rotation matrix
                np.save(f"results/Thingi32/{mesh_name}/rotation_matrix.npy", R)
            else:
                # read rotation matrix
                R = np.load(f"results/Thingi32/{mesh_name}/rotation_matrix.npy")
                # read ground truth mesh
                V_gt, F_gt = gpy.read_mesh(f"results/Thingi32/{mesh_name}/ground_truth.obj")
                # rotate
                V_gt = V_gt @ R
 
            for resolution in resolutions:
                os.makedirs(f"{mesh_results_path}/{resolution}", exist_ok=True)
                U, S = get_sdf_data(V_gt, F_gt, resolution)
                # save U and S as npy files
                np.save(f"{mesh_results_path}/{resolution}/U.npy", U)
                np.save(f"{mesh_results_path}/{resolution}/S.npy", S)
                for method in methods:
                    # run all methods in queue to avoid memory issues
                    queue = Queue()
                    p = Process(target=run_method, args=(U, S, method, f"{mesh_results_path}/{resolution}/{method}.obj", R, V_gt, F_gt, queue))
                    p.start()
                    p.join(timeout=timeout_duration)
                    if p.is_alive():
                        print(f"Method {method} on mesh {mesh_name} with resolution {resolution} timed out.")
                        p.terminate()
                        p.join()
                    else:
                        print(f"Method {method} on mesh {mesh_name} with resolution {resolution} finished successfully.")
    
    if args.metrics:
        # now loop over every mesh, every resolution, every method, and add row with error metrics to a CSV file
        # CSV file will have columns: mesh, resolution, method, metric, error
        with open(path_to_results_csv, mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(["mesh", "resolution", "method", "metric", "error"])
            # loop oever every subdirectiry of results/Thingi32
            for mesh_path in sorted(os.listdir("results/Thingi32")):
            # for mesh in sorted(selected_mesh_files):
                mesh_name = mesh_path.split("/")[-1]
                mesh = f"{mesh_name}.obj"
                # mesh_name = mesh.split(".")[0]
                print(f"mesh = {mesh_name}")
                ground_truth_path = f"results/Thingi32/{mesh_name}/ground_truth.obj"
                rotation_path = f"results/Thingi32/{mesh_name}/rotation_matrix.npy"
                try:
                    V_gt, F_gt = gpy.read_mesh(ground_truth_path)
                    R = np.load(rotation_path)
                except:
                    continue
                for resolution in resolutions:
                    U = np.load(f"results/Thingi32/{mesh_name}/{resolution}/U.npy")
                    S = np.load(f"results/Thingi32/{mesh_name}/{resolution}/S.npy")
                    all_errors = np.zeros((len(methods), len(metrics)))
                    some_method_failed = False
                    for (i, method) in enumerate(methods):
                        try:
                            reconstruction_path = f"results/Thingi32/{mesh_name}/{resolution}/{method}.obj"
                            V, F = gpy.read_mesh(reconstruction_path)
                            for (j,metric) in enumerate(metrics):
                                error = compute_error(V_gt, F_gt, V, F, U, S, R, metric)
                                all_errors[i,j] = error
                                print(f"mesh = {mesh_name}, resolution = {resolution}, method = {method}, metric = {metric}, error = {error}")
                        except:
                            some_method_failed = True
                            print(f"mesh = {mesh_name}, resolution = {resolution}, method = {method}, metric = {metric}, error = failed")
                            continue
                    if not some_method_failed:
                        for (i, method) in enumerate(methods):
                            for (j,metric) in enumerate(metrics):
                                results_writer.writerow([mesh_name, resolution, method, metric, all_errors[i,j]])
            results_file.close()

        # now to calculate averages per method, resolution and metric (i.e., across all meshes)
        # CSV file will have columns: resolution, method, metric, error
        # not super effcient (we are reading the whole thing many times, but whatever, it's simpler and this is not a bottleneck)
        with open(path_to_avg_results_csv, mode='w') as avg_results_file:
            avg_results_writer = csv.writer(avg_results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            avg_results_writer.writerow(["resolution", "method", "metric", "error"])
            for resolution in resolutions:
                for (i, method) in enumerate(methods):
                    for (j,metric) in enumerate(metrics):
                        # read all errors for this method, resolution and metric
                        errors = []
                        with open(path_to_results_csv, mode='r') as results_file:
                            results_reader = csv.reader(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            for row in results_reader:
                                if len(row) < 5:
                                    continue
                                if row[1] == str(resolution) and row[2] == method and row[3] == metric:
                                    errors.append(float(row[4]))
                        avg_error = np.mean(errors)
                        avg_results_writer.writerow([resolution, method, metric, avg_error])
            avg_results_file.close()

    if args.table:
        captions = [                                    # Captions for each CSV 
                    "Average results over all test shapes"
                ]
        method_names = [ "RFTS", "RFTA"]
        metric_names = [ "Chf", "$\mathcal{E}_{SDF}$"]
        csv_files = [path_to_avg_results_csv]
        latex_table = generate_combined_latex_table(csv_files, methods, resolutions, metrics, captions, method_names, metric_names)
        print(latex_table)
        latex_table_path = "./paper/sections/Thingi32.tex"
        with open(latex_table_path, 'w') as latex_table_file:
            latex_table_file.write(latex_table)
            latex_table_file.close()

if __name__ == "__main__":
    main()