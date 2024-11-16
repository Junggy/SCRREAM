# SCRREAM

This is the official repository for SCRREAM (**SC**an, **R**egister, **RE**nder **A**nd **M**ap) dataset. We provide code example and intruction for visualizing our dataset as well as dataset download link.

## Link to Download Dataset
**Indoor Reconstruction and SLAM dataset & Object Removal and Scene Editing dataset :**

https://drive.google.com/file/d/1Lu5VdlLn5NNoPSh2Tp90M6NF2Y1us677/view?usp=sharing (scene01-05, 59 Gb) 
https://drive.google.com/file/d/1V8HRX4-12jv-ZNW6AM_fyHoNiDstBV8q/view?usp=sharing (scene06-11, 101 Gb) 

In case of slow internet, please download with smaller zip files

https://drive.google.com/file/d/11QIO2ZT2DnFo8V9OMiBzMjRAwjcUJ5qC/view?usp=sharing (scene01)\
https://drive.google.com/file/d/1jvIucH0PI9hkuYy3cRwYR7_eFpt7Nhpi/view?usp=sharing (scene02)\
https://drive.google.com/file/d/1mL2NajhaUxXvhvjGd7WPe7KKe037TshJ/view?usp=sharing (scene03)\
https://drive.google.com/file/d/1Tj6LckAubUd-OI8SwCEwULgwjRatdZbD/view?usp=sharing (scene04)\
https://drive.google.com/file/d/1E7Ahm4ERde9gXsUb3pXVJntOFWUqWSil/view?usp=sharing (scene05)\
https://drive.google.com/file/d/16kB_pjzb5V-VS8Ma-EVwtJYH0jxt1ptb/view?usp=sharing (scene06)\
https://drive.google.com/file/d/1-f-RkDdjVKwJdZeGLHGPwpz--cJNd11M/view?usp=sharing (scene07)\
https://drive.google.com/file/d/1VPlD4zvALDeDfPtXBigubgAJxJlueZBA/view?usp=sharing (scene08)\
https://drive.google.com/file/d/1DUBOMxurmjWSU2R5MUs6iXJMFyKTBhIA/view?usp=sharing (scene09)\
https://drive.google.com/file/d/1h2v4hrY3IxsR49MV65ZOxXI8xMMHGL1Y/view?usp=sharing (scene10)\
https://drive.google.com/file/d/1fwjpmXQ29sO_wk1SOmfdhlGq7Kyv8crN/view?usp=sharing (scene11)


**Human Reconstruction Dataset :**

https://drive.google.com/file/d/1BHHb5ibNsYsm00FXUrwLRkvC630sYq3U/view?usp=sharing (scene01-02, 7.7 Gb)

**Pose Estimation Dataset :**

https://drive.google.com/file/d/1kcy8DCu6L2GtU2vK22w9FhhlLP6ljq9b/view?usp=sharing (scene01-02, 7.4 Gb)

Once the dataset is downloaded, unzip in the ```dataset``` folder.
It should look like this

```
dataset\
   human_scene01\..
   human_scene02\..
   pose_meshes_canonical\..
   pose_scene01\..
   pose_scene02\..
   scene01\..
      ...
   scene11\..
   README.txt
```


## Instruction for Visualization
### Requirements
Install requirements with pip with this command
```
pip install -r requirements.txt
```
We tested with python version 3.9 on windows 10 machine.

### Visualizing Indoor Reconstruction and SLAM Dataset & Object Removal and Scene Editing Dataset

To visualize the Indoor Reconstruction and SLAM Dataset & Object Removal and Scene Editing Dataset,
run ```render_scene.py``` script with argument ```{dataset_dir} {scene} {traj} {frame}```, such as
```
python render_scene.py {dataset_dir} {scene} {traj} {frame}
```
If ```{frame}``` is set to -1, the script with go thorugh the entire frame as a video sequence.
If ```{frame}``` is set to positive integer, the script will display the image with the given frame number 

```
python render_scene.py dataset scene01 full_00  -1 \\ for visualizing entire sequnce in scene01_full_00
python render_scene.py dataset scene01 full_00 100 \\ for visualizing 100th frame in scene01_full_00
```

To visualize the reduced scene for object removal or scene editing experiments, use ```{traj}``` that contains reduced. scene01, scene02, scene04, scene05, scene06, scene07, scene08, scene09 contain reduced scene. 
For example, 
```
render_scene.py dataset scene01 reduced_00 -1 \\ for visualizing enture sequnce in scene01_full_00
```
will play the sequence from scene01 with reduced objects.

The visualization is formatted as 2x3 image layout with the given format:
```
ㅣ   (Ground Truth Depth)   |    (D435 Depth)    |     (ToF Depth)    |
ㅏㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅓ
ㅣ (RGB with Semantic Mask) |    (D435 Error)    |     (ToF Error)    |
```

### Visualizing Human Reconstruction Dataset
(Pre-requisite) Download SMPL model into . As our dataset uses smpl model format, only downloading smpl format will be enough (download .pkl file if possible. Our dataset assumes .pkl in the script).
The folder should look like this :
```
models\
   smpl\..
   README.txt
```

To visualize the human reconstruction dataset, run ```render_human.py``` script with argument ```{dataset_dir} {scene} {frame} {view}```, such as 
```
python render_human.py {dataset_dir} {scene} {frame} {view}
```
Note that our human dataset contains 4 multiview images per each frame (or human posture). If ```view``` is set to -1, the script will plot 4 consecutive views on given frame.
```
python human_scene.py dataset human_scene01 0 0 \\ for visualizing first view of frame 0 in the human_scene01
python human_scene.py dataset human_scene01 0 -1 \\ for visualizing all 4 views of frame 0 in the human_scene01
```
The visualization is formatted as 1x3 image layout with the given format:
```
| (RGB with Semantic Mask) | (RGB with Scanned Human Mesh) | (RGB with SMPL Human Mesh) |
```




### Visualizing 6D Pose Estimation Dataset
To visualize the human reconstruction dataset, run ```render_pose.py``` script with argument ```{dataset_dir} {scene} {frame}```, such as 
```
python render_pose.py {dataset_dir} {scene} {frame}
```
If ```{frame}``` is set to -1, the script with go thorugh the entire frame as a video sequence.
If ```{frame}``` is set to positive integer, the script will display the image with the given frame number
```
python pose_scene.py dataset pose_scene01  -1 \\ for visualizing entire sequnce in pose_scene01
python pose_scene.py dataset pose_scene01 100 \\ for visualizing 100th frame in pose_scene01
```
The visualization is formatted as 1x1 image layout with the given format:
```
| (RGB with Pose 3D Bounding Box and Mask) |
```



