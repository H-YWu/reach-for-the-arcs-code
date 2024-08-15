import platform, os, sys
os_name = platform.system()
if os_name == "Darwin":
    # Get the macOS version
    os_version = platform.mac_ver()[0]
    # print("macOS version:", os_version)

    # Check if the macOS version is less than 14
    if os_version and os_version < "14":
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build-studio')))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
elif os_name == "Windows":
    # For Windows systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Debug')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build/Release')))
else:
    # For other systems
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../build')))
    
from .fine_tune_point_cloud import fine_tune_point_cloud, fine_tune_point_cloud_iteration
from .locally_make_feasible import locally_make_feasible
from .outside_points_from_rasterization import outside_points_from_rasterization
from .point_cloud_to_mesh import point_cloud_to_mesh
from .power_diagram import power_diagram
from .sdf_to_point_cloud import sdf_to_point_cloud