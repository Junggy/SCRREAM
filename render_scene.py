import pyrender, trimesh, os, glob, cv2, argparse
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap("gist_rainbow")

if os.name == 'nt':
    separator = "\\"
else:
    separator = '/'

scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1.0, 1.0, 1.0])

parser = argparse.ArgumentParser(description="Undistort images")
parser.add_argument("dataset_dir")
parser.add_argument("scene_name")
parser.add_argument("traj_name")
parser.add_argument("idx")

args = parser.parse_args()

dataset_dir = args.dataset_dir
scene_name  = args.scene_name
traj_name = args.traj_name
frame_idx = int(args.idx)

base = os.path.join(dataset_dir,scene_name,scene_name+"_"+traj_name)
mesh_base = os.path.join(dataset_dir,scene_name,"meshes")
mesh_names = glob.glob(os.path.join(mesh_base,"*.obj"))

with open(os.path.join(base,"meta.txt"),"r") as f:
    meshes_in_the_scene = [each_line.strip().split(" ")[1] for each_line in f.readlines()]

meshes = {}

for each_mesh_name in mesh_names:

    each_mesh_name_clean = each_mesh_name.split(separator)[-1].split(".")[0]

    if not (each_mesh_name_clean in meshes_in_the_scene): 
        print("skipping",each_mesh_name_clean)
        continue
    else:
        print("loading",each_mesh_name_clean)

    trimesh_obj = trimesh.load(each_mesh_name)
    trimesh_obj.visual = trimesh.visual.ColorVisuals()
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    meshes[each_mesh_name] = mesh
    scene.add(mesh)

intrinsic = np.loadtxt(os.path.join(base, "intrinsics.txt"))
fx,fy,px,py = intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2]
camera = pyrender.IntrinsicsCamera(0,0,0,0)
camera.fx = fx
camera.fy = fy
camera.cx = px
camera.cy = py
camera_node = scene.add(camera)


images = glob.glob(os.path.join(base, "rgb", "*.png"))
poses = glob.glob(os.path.join(base, "camera_pose", "*.txt"))
instances = glob.glob(os.path.join(base, "instance", "*.png"))
depth_d435 = glob.glob(os.path.join(base, "depth_d435", "*.png"))
depth_tof  = glob.glob(os.path.join(base, "depth_tof", "*.png"))

assert len(set([len(images),len(poses),len(instances),len(depth_d435),len(depth_tof)])) == 1

h,w,_ = cv2.imread(images[0]).shape

r = pyrender.OffscreenRenderer(w, h)

error_cmap = plt.get_cmap("seismic")
depth_cmap = plt.get_cmap("inferno")
instance_cmap = plt.get_cmap("gist_rainbow")

images.sort()
poses.sort()
instances.sort()
depth_d435.sort()
depth_tof.sort()

cv2pyrender = np.array([[1,1,1,1],
                       [-1,-1,-1,-1],
                       [-1,-1,-1,-1],
                       [1,1,1,1]])

if frame_idx == -1:
    start = 0
    end   = len(images)
else:
    start = frame_idx
    end   = frame_idx+1

for idx in range(start,end):

    each_rgb = cv2.imread(images[idx],-1)
    each_instance = cv2.imread(instances[idx],-1)
    each_depth_d435 = cv2.imread(depth_d435[idx],-1) / 1000
    each_depth_tof = cv2.imread(depth_tof[idx],-1) / 1000
    each_pose = np.loadtxt(poses[idx])

    scene.set_pose(camera_node, cv2pyrender.T * each_pose)

    color, depth = r.render(scene)

    d435_error = (depth - each_depth_d435) * (each_depth_d435 != 0)
    tof_error = (depth - each_depth_tof) * (each_depth_tof != 0)

    dmax = 4
    dmin = 0
    error_max = 1
    error_min = -error_max

    each_instance_cmap = (instance_cmap(each_instance)[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    rgb_augmented = cv2.addWeighted(each_instance_cmap, 0.3, each_rgb, 0.5, 1).astype(np.float32) / 255

    if frame_idx == -1:

        depth_clipped = ((depth.clip(dmin,dmax) - dmin) / (dmax-dmin) * 255).astype(np.uint8)
        d435_clipped = ((each_depth_d435.clip(dmin,dmax) - dmin) / (dmax-dmin) * 255).astype(np.uint8)
        tof_clipped = ((each_depth_tof.clip(dmin, dmax) - dmin) / (dmax - dmin) * 255).astype(np.uint8)

        depth_8bit = depth_cmap(depth_clipped)[:,:,[2,1,0]]
        d435_8bit = depth_cmap(d435_clipped)[:,:,[2,1,0]]
        tof_8bit = depth_cmap(tof_clipped)[:, :, [2,1,0]]

        error_d435_clipped = error_cmap((d435_error.clip(error_min,error_max) - error_min) / 2*error_max)[:,:,[2,1,0]]
        error_tof_clipped = error_cmap((tof_error.clip(error_min,error_max) - error_min) / 2*error_max)[:,:,[2,1,0]]

        plot_row1 = np.hstack([depth_8bit,d435_8bit,tof_8bit])
        plot_row2 = np.hstack([rgb_augmented,error_d435_clipped,error_tof_clipped])

        plot = np.vstack([plot_row1,plot_row2])
        h_,w_,_ = plot.shape
        plot_reshape = cv2.resize(plot,(w_//2,h_//2))

        cv2.imshow("error",plot_reshape)
        cv2.waitKey(5)

    else:
        plt.figure()
        plt.subplot(2,3,1)
        plt.title("Depth GT")
        plt.imshow(depth,cmap="inferno",vmax=dmax,vmin=dmin)
        plt.subplot(2,3,2)
        plt.title("Depth D435")
        plt.imshow(each_depth_d435, cmap="inferno", vmax=dmax, vmin=dmin)
        plt.subplot(2,3,3)
        plt.title("Depth ToF")
        plt.imshow(each_depth_tof, cmap="inferno", vmax=dmax, vmin=dmin)
        plt.subplot(2,3,4)
        plt.title("RGB with Semantic Mask")
        plt.imshow(rgb_augmented[:,:,[2,1,0]])
        plt.subplot(2,3,5)
        plt.title("D435 Depth Error")
        plt.imshow(d435_error, cmap="seismic", vmax=error_max, vmin=error_min)
        plt.subplot(2,3,6)
        plt.title("ToF Depth Error")
        plt.imshow(tof_error, cmap="seismic", vmax=error_max, vmin=error_min)
        plt.show()
