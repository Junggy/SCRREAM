import pyrender, trimesh, argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, glob
import _pickle as cPickle

cls_id_to_name = {1: "box",
                  2: "bottle",
                  3: "can",
                  4: "cup",
                  5: "remote",
                  6: "teapot",
                  7: "cutlery",
                  8: "glass",
                  9: "shoe",
                  10: "tube"}

name_to_cls_id = {v:k for k,v in cls_id_to_name.items()}

housecat2pyrender_conversion = np.array([[1, 1, 1, 1],
                                         [-1, -1, -1, -1],
                                         [-1, -1, -1, -1],
                                         [1, 1, 1, 1]])
    
def main():

    parser = argparse.ArgumentParser(description="render_obj_with_pose")

    parser.add_argument("dataset_dir")
    parser.add_argument("scene_name")
    parser.add_argument("idx")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    folder_name = os.path.join(dataset_dir,args.scene_name)

    render_idx = int(args.idx)

    object_list = glob.glob(os.path.join(folder_name,"obj_pose")+"/*.txt")
    

    with open(os.path.join(folder_name,"meta.txt")) as f:
        object_list= [each_line.split(" ")[1] for each_line in f.readlines() if  each_line.split(" ")[0] in cls_id_to_name.values()]

    objects = {}
    for each_object in object_list:

        if os.name == 'nt':
            object_name_full = each_object.split("\\")[-1]
        else:
            object_name_full = each_object.split("/")[-1]
        each_class, each_name = object_name_full.split(".")[0].split("-")
        objects[(each_class,each_name)] = {}

    # setup pyrender scene with camera
    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.7,0.7,0.7])
    k = np.loadtxt(os.path.join(folder_name, "intrinsics.txt"))



    camera = pyrender.IntrinsicsCamera(0, 0, 0, 0)
    camera.fx = k[0, 0]
    camera.fy = k[1, 1]
    camera.cx = k[0, 2]
    camera.cy = k[1, 2]
    scene.add(camera)

    # mesh
    obj_folder = os.path.join(dataset_dir,"pose_meshes_canonical")
    bbox_scales = []
    for each_key in objects.keys():
        print("loading mesh (category, instance) :",each_key)

        obj_class, obj_name = each_key

        if obj_class in name_to_cls_id.keys():

            
            obj_fname = os.path.join(obj_folder,obj_class + "-" + obj_name + '.obj')

            trimesh_obj = trimesh.load(obj_fname)
            bbox_scales.append(2*trimesh_obj.vertices.max(0))
            mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
            objects[each_key]["mesh"] = scene.add(mesh)

    bbox_scales = np.stack(bbox_scales,0)

    n_images = len(glob.glob("{0}/{1}/*.png".format(folder_name,"rgb")))
    print("n_images :", n_images)

    with open(os.path.join(folder_name, "meta.txt"), 'r') as f:
        instance_labels = [each_line.strip().split(" ") for each_line in f.readlines()]

    if render_idx==-1:
        _start,_end = 0,n_images
    else:
        _start,_end = render_idx, render_idx+1

    for idx in range(_start,_end):

        img = plt.imread("{0}/rgb/{1:06d}.png".format(folder_name, idx))[:,:,[2,1,0]]

        if idx == _start:
            h,w,_ = img.shape
            r = pyrender.OffscreenRenderer(w, h)

        pkl_name = "{0}/labels/{1:06d}_label.pkl".format(folder_name, idx)
        with open(pkl_name, 'rb') as f:
            label = cPickle.load(f)

        RTs = []
        
        instance_count = 0
        for label_idx,each_line in enumerate(instance_labels):
            
            cls,name,instance = each_line

            if cls in name_to_cls_id.keys():
                cls = name_to_cls_id[cls]

                _,name = name.split("-")

                obj_to_cam = np.identity(4)

                rotation = label['rotations'][instance_count]
                translation = label['translations'][instance_count]
                obj_to_cam[:3,:3] = rotation
                obj_to_cam[:3,3] = translation
                RTs.append(np.array(obj_to_cam))
                obj_to_cam *= housecat2pyrender_conversion
                scene.set_pose(objects[(cls_id_to_name[int(cls)],name)]["mesh"], obj_to_cam)

                instance_count += 1

        RTs = np.stack(RTs,0)
        color_i, depth_i = r.render(scene)

        img = (img).astype(np.float32)
        mask = np.stack([np.zeros_like(depth_i),(depth_i != 0),np.zeros_like(depth_i)],-1).astype(np.float32)

        _,_,c = img.shape

        overlay =  cv2.addWeighted(mask, 0.2, img, 0.8, 0)
        overlay = (overlay * 255).astype(np.uint8)
        overlay = draw_detections(overlay, k, RTs, bbox_scales)


        if render_idx == -1:
            cv2.imshow("vis",overlay)
            cv2.waitKey(1)
        else:
            plt.figure()
            plt.imshow(overlay[:,:,[2,1,0]])
            plt.show()



"""
Functions for drawing 3d BBox (draw_detections, transform_coordinates_3d, get_3d_bbox, calculate_2d_projections, draw)
are adapted from Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation Detection and 
evaluation code (https://github.com/hughw19/NOCS_CVPR2019)
"""
def draw_detections(image, intrinsics, gt_RTs, gt_scales):

    draw_image_bbox = image.copy()

    if gt_RTs is not None:
        for ind, RT in enumerate(gt_RTs):

            xyz_axis = 0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, RT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

            bbox_3d = get_3d_bbox(gt_scales[ind], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)

            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (0, 255, 0))

    return draw_image_bbox

def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]
    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)

    # draw pillars in blue color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)

    return img



if __name__ == "__main__":
    main()
#              ^
# cv pose :   /  z
#             ---> x
#            |
#            v y

#                      ^ y
# pyrender pose :      |
#                      ---> x
#                    /
#                   v z
