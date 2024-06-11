import pyrender, trimesh, os, glob, cv2, argparse, smplx, pickle
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap("gist_rainbow")

if os.name == 'nt':
    separator = "\\"
else:
    separator = '/'

scene = pyrender.Scene(bg_color=[0, 0, 0],ambient_light=[0.5,0.5,0.5])

parser = argparse.ArgumentParser(description="Undistort images")
parser.add_argument("dataset_dir")
parser.add_argument("scene_name")
parser.add_argument("idx")
parser.add_argument("view_idx")

args = parser.parse_args()

dataset_dir = args.dataset_dir
scene_name  = args.scene_name
frame_idx = int(args.idx)
view_idx = int(args.view_idx)

assert frame_idx >= 0, "frame_idx has to be greater than 0"
assert view_idx >= -1 and view_idx < 4, "view_idx has to be a value between -1 and 3"

base = os.path.join(dataset_dir,scene_name)
mesh_base = os.path.join(dataset_dir,scene_name,"meshes")
mesh_names = glob.glob(os.path.join(mesh_base,"*.obj"))

meshes = {}

# load the human as a scanned mesh
human_mesh_name_clean = "human-{0:02d}.obj".format(frame_idx)
human_mesh_name = os.path.join(dataset_dir,scene_name,"meshes",human_mesh_name_clean)
print("loading scanned human mesh")

trimesh_obj = trimesh.load(human_mesh_name)
human_mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
meshes["scanned_mesh"] = human_mesh

# load the human as a SMPL mesh
import smplx, torch

print("loading smpl mesh")
annotation_folder = os.path.join(dataset_dir,scene_name,"human_annotation")
annotation_file = os.path.join(annotation_folder,"human-{0:02d}_smpl.pkl".format(frame_idx))

with open(annotation_file, 'rb') as f:
    x = pickle.load(f)

smpl_model = smplx.create("models",model_type="smpl", gender="male",num_beta=300,ext="npz")
faces = smpl_model.faces

output = smpl_model(return_verts=True,
          body_pose = torch.tensor(x["pose"][3:]).float().unsqueeze(0),
          betas = torch.tensor(x["betas"])[:10].float().unsqueeze(0),
          global_orient = torch.tensor(x["pose"][:3]).float().unsqueeze(0),
          transl = torch.tensor(x["trans"]).float().unsqueeze(0),
          )
vertices = output.vertices
smpl_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vertices.detach().cpu().numpy().squeeze(),
                       faces=faces))
meshes["smpl_mesh"] = smpl_mesh

intrinsic = np.loadtxt(os.path.join(base, "intrinsics.txt"))
fx,fy,px,py = intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2]
camera = pyrender.IntrinsicsCamera(0,0,0,0)
camera.fx = fx
camera.fy = fy
camera.cx = px
camera.cy = py
camera_node = scene.add(camera)

light = pyrender.SpotLight(color=np.ones(3), intensity=10.0,
                            innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/3.0)
light_node = scene.add(light)


images = glob.glob(os.path.join(base, "rgb", "*.png"))
poses = glob.glob(os.path.join(base, "camera_pose", "*.txt"))
instances = glob.glob(os.path.join(base, "instance", "*.png"))

assert len(set([len(images),len(poses),len(instances)])) == 1

h,w,_ = cv2.imread(images[0]).shape

r = pyrender.OffscreenRenderer(w, h)

error_cmap = plt.get_cmap("seismic")
depth_cmap = plt.get_cmap("inferno")
instance_cmap = plt.get_cmap("gist_rainbow")

images.sort()
poses.sort()
instances.sort()

cv2pyrender = np.array([[1,1,1,1],
                       [-1,-1,-1,-1],
                       [-1,-1,-1,-1],
                       [1,1,1,1]])

if view_idx == -1:
    start = 0
    end   = 4
else:
    start = view_idx
    end   = view_idx+1

for idx_pre in range(start,end):

    idx = frame_idx * 4 + idx_pre

    each_rgb = cv2.imread(images[idx],-1)
    each_instance = cv2.imread(instances[idx],-1)
    each_pose = np.loadtxt(poses[idx])

    scene.set_pose(camera_node, cv2pyrender.T * each_pose)
    scene.set_pose(light_node, cv2pyrender.T * each_pose)

    added_node = scene.add(meshes["scanned_mesh"])
    color_scanned, _ = r.render(scene)
    scene.remove_node(added_node)

    added_node = scene.add(meshes["smpl_mesh"])
    color_smpl, _ = r.render(scene)
    scene.remove_node(added_node)

    each_instance_cmap = (instance_cmap(each_instance)[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    rgb_augmented = cv2.addWeighted(each_instance_cmap, 0.3, each_rgb, 0.5, 1).astype(np.float32) / 255
    rgb_augmented_scanned = cv2.addWeighted(color_scanned[:,:,[2,1,0]], 0.7, each_rgb, 0.3, 1).astype(np.float32) / 255
    rgb_augmented_smpl    = cv2.addWeighted(color_smpl[:,:,[2,1,0]], 0.7, each_rgb, 0.3, 1).astype(np.float32) / 255

    plot = np.hstack([rgb_augmented,rgb_augmented_scanned,rgb_augmented_smpl])

    if view_idx == -1:

        h_,w_,_ = plot.shape
        plot_reshape = cv2.resize(plot,(w_//2,h_//2))

        cv2.imshow("human_visualization",plot_reshape)
        cv2.waitKey(500)

    else:
        plt.figure()
        plt.imshow(plot[:,:,[2,1,0]])
        plt.show()