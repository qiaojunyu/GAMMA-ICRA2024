# GAMMA: Generalizable Articulation Modeling and Manipulation for Articulated Objects


## 1. down URDF
Please follow the instruction in https://github.com/daerduoCarey/where2act/tree/main/code#before-start, download where2act_original_sapien_dataset.zip and unzip here.

## 2. URDF annotation
In this paper, our joint types include revolute and parallel-axis; therefore, the selected parts in the code are "door" and "drawer."

```bash
python anno_urdf.py --data_type train  --file_path "down URDF file path"
or
python anno_urdf.py --data_type test --file_path "down URDF file path"
```
if you want to add new joint types, you can add the joint type in the code (anno_urdf.py) from line 100 to 101:
```bash
if joint_info["joint_type"] == JointType.revolute.value or joint_info["joint_type"] == JointType.prismatic.value:
    select_joints.append(joint_info)
```
## 3. Data generation

This render code is based on sapien 2.2.2
```bash
pip install sapien==2.2.2
```
more detail, please refer to https://sapien.ucsd.edu/

```bash
 python data_render.py  --urdf_path ./where2act_original_sapien_dataset/ --object_id 10143 --render_view_id 0 --save_path ./train/
```
if you want to change the camera parameters (data_render.py) from line 119 to 121:
```bash
    phi = np.random.uniform(150, 210)
    theta = np.random.uniform(45, 80)
    distance = np.random.uniform(4.5, 5.0)
```

## process data for trainig and testing
```bash
 python data_process.py  --urdf_path ./where2act_original_sapien_dataset/ --render_file_path ./train/10143/0 --save_path ./train_processed/
```





