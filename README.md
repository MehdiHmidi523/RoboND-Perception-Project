# Project: Perception Pick & Place

### Submission for Udacity Robotics Software Engineer Nanodegree
### Sebastian Castro - 2018

[//]: # (Image References)

[intro_screenshot]: ./misc_images/GazeboPickPlace.PNG
[confusion_matrices]: ./misc_images/confusion_matrices.PNG
[world1]: ./misc_images/World1.PNG
[world2]: ./misc_images/World2.PNG
[world3]: ./misc_images/World3.PNG

---

## Introduction
In this project, we implemented a perception pipeline for object recognition in a pick and place application using a simulated PR2 robot. 

This robot uses an RGB-D camera to detect a noisy point cloud of the scene in front of it. Then, several perception algorithms are used to identify the location and type of objects on the table. The robot must then grab each object with the correct arm and place it in the correct bin, as specified by an externally provided pick list.

![Project overview image][intro_screenshot]


## Perception Pipeline Methodology
Here, we will discuss the steps taken in the perception pipeline and the parameters selected at each step.

### STEP 1: Point Cloud Processing
The first set of steps is to extract the object information from the noisy point cloud provided. This consists of:

* Removing noise using Statistical Outlier Filtering. We selected a neighborhood of **20** points and a standard deviation threshold of **0.5** sigma.
* Downsampling the point cloud for faster processing. In accordance with the exercises, we selected a voxel size of **0.01** m (1 cm).
* Using knowledge of static scene to only consider a subset of XYZ point boundaries. To do this, we used a Pass-Through filter. Z limits between **0.25** and **2.0** meters ensured we only kept the objects and table. Additionally, X limits between **-0.5** and **0.5** meters removed data points from the bins on either side of the robot's field of vision.
* Finally, we used RANSAC to fit a plane model to the table. Using a threshold of **0.03** m (3 cm) adequately separate the table (inliers) from the objects (outliers).

The relevant point cloud processing code in `project_template.py` is below:

```
# Statistical Outlier Filtering
out_filt = pt_cloud.make_statistical_outlier_filter();
out_filt.set_mean_k(20); # Number of neighboring points to analyze
out_filt.set_std_dev_mul_thresh(0.5); # Standard deviation threshold
cloud_filtered = out_filt.filter();

# Voxel Grid Downsampling
LEAF_SIZE = 0.01; 	# Voxel size [m]
vox = cloud_filtered.make_voxel_grid_filter();
vox.set_leaf_size(LEAF_SIZE,LEAF_SIZE,LEAF_SIZE);
cloud_downsampled = vox.filter(); 

# PassThrough Filter
# First, do this on the Z to remove the ground
Z_MIN = 0.25;	# Minimum Z value [m]
Z_MAX = 2;		# Maximum Z value [m]
passthru = cloud_downsampled.make_passthrough_filter();
passthru.set_filter_field_name('z');
passthru.set_filter_limits(Z_MIN,Z_MAX);
cloud_passthru = passthru.filter();
# Then, do this on the Y to remove boxes on the side
X_MIN = -0.5;		# Minimum X value [m]
X_MAX =  0.5;		# Maximum X value [m]
passthru = cloud_passthru.make_passthrough_filter();
passthru.set_filter_field_name('y');
passthru.set_filter_limits(X_MIN,X_MAX);
cloud_passthru = passthru.filter();

# RANSAC Plane Segmentation
DIST_THRESH = 0.03;	# Distance threshold for plane fitting [m]
seg = cloud_passthru.make_segmenter();
seg.set_model_type(pcl.SACMODEL_PLANE);
seg.set_method_type(pcl.SAC_RANSAC);
seg.set_distance_threshold(DIST_THRESH);
inliers, coefs = seg.segment();

# Extract inliers and outliers
cloud_inliers = cloud_passthru.extract(inliers,negative=False);
cloud_outliers = cloud_passthru.extract(inliers,negative=True);
```

### STEP 2: Point Cloud Clustering
The next set of steps is to take the unstructured point cloud data for the objects on the table and divide them into different clusters. Ideally, 
each cluster will represent a separate object on the table. This consists of:

* Using Euclidean Clustering to separate the objects into separate clusters. We selected a distance tolerance of **0.05** m (5 cm), minimum cluster size of **100** points, and maximum cluster size of **10000** points.
* Assigning colors to each of the selected clusters
* Publishing the cluster information as a ROS message to the `/pcl_cluster` topic. This data is needed to create the overlays on the image and point cloud data in RViz.

The relevant point cloud processing code in `project_template.py` is below:

```
# Euclidean Clustering
cloud_xyz = XYZRGB_to_XYZ(cloud_outliers);
tree = cloud_xyz.make_kdtree();
ec = cloud_xyz.make_EuclideanClusterExtraction();
ec.set_ClusterTolerance(0.05);	# Distance tolerance [m]
ec.set_MinClusterSize(100);	# Minimum cluster size
ec.set_MaxClusterSize(10000);	# Maximum cluster size
ec.set_SearchMethod(tree);
cluster_indices = ec.Extract();

# Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_indices));
color_cluster_point_list = [];
for j, indices in enumerate(cluster_indices):
for i, index in enumerate(indices):
    color_cluster_point_list.append([
                                     cloud_xyz[index][0],
                                     cloud_xyz[index][1],
                                     cloud_xyz[index][2],
                                     rgb_to_float(cluster_color[j])
                                    ])
cluster_cloud = pcl.PointCloud_PointXYZRGB();
cluster_cloud.from_list(color_cluster_point_list);

# Convert PCL data to ROS messages
cluster_msg = pcl_to_ros(cluster_cloud);

# Publish ROS messages
pcl_cluster_pub.publish(cluster_msg);
```

### STEP 3: Object Detection
In this final step, we extract features from each point cloud cluster to try classify the type of object these points correspond to.

The features used were:
* **HSV** color space, with **25** bins for each channel between 0 and 255.
* Normals only in the **Z** direction, with **5** bins between -1 and 1.

HSV performed better than RGB for separating objects that were more similar in color. This is because the saturation and value channels could be used to better separate these objects. 25 color bins were selected by changing the numbers of color bins and checking the test accuracy. We also found that the normal data contributed to overfitting, especially since the training data used many more orientations than the actual project environment. Due to this, we only created features out of the Z-direction normals, and picked a smaller number of bins.

The SVM kernel selected was **linear**. A polynomial kernel with order 2 or higher yielded worse test results. The Radial Basis Function (RBF) kernel gave slightly better test results, but empirically performed worse on the actual project. This is likely because radial basis functions are more prone to overfitting since they can create more complex decision boundaries than a linear SVM.

**NOTE:** The modified `features.py` and `capture_features.py` scripts are in the `/pr2_robot/scripts` folder of this repository. You can run `capture_features.py` using rosrun from the `pr2_robot` package, but `features.py` must be copied over to your existing `sensor_stick` source package.

The relevant classification code in `project_template.py` is below:

```
# Grab the points for the cluster
pcl_cluster = cloud_outliers.extract(pts_list)

# Compute the associated feature vector
feat_cloud = pcl_cluster.to_array()
chists = compute_color_histograms(ros_cluster, using_hsv=True)
normals = get_normals(ros_cluster)
nhists = compute_normal_histograms(normals)
feature = np.concatenate((chists, nhists))

# Make the prediction
prediction = classifier.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
```


## Software Implementation
A few modifications were made to the project infrastructure to help automate the process of object detector training and results creation.

### Data Collection Automation
Arguments were added to the `capture_features.py` script to make training data collection easier and more configurable. The 2 optional arguments are
* World Number (default = 1)
* Number of point clouds per object (default = 10)

For example, to capture training features for all objects in World 3, with 50 point clouds for each object, you can enter the following command at the Terminal.

```
rosrun pr2_robot capture_features.py 3 50
```

### Launch File and Results File Automation
We added a `world` argument to the `pick_place.launch` file, which accordingly sets the Gazebo world number, pick list file name, and creates a ROS parameter called `/world_num` on the parameter server. If no argument is specified, World 1 will be picked.

For example, this is the command in the Terminal that starts the pick and place environment for World 2:

```
roslaunch pr2_robot pick_place_project.launch world:=2
```

Looking at the contents of the launch file itself, these were the modifications:

```
<!--Create a ROS parameter for the world number based on an argument-->
  <arg name="world" default="1"/>
  <rosparam param="world_num" subst_value="true">$(arg world)</rosparam>

<!--Launch a gazebo world-->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
<!--Set the world name based on the world specified-->
<arg name="world_name" value="$(find pr2_robot)/worlds/test$(arg world).world"/>
...
</include>

<!--Set the list name based on the world number specified-->
<rosparam command="load" file="$(find pr2_robot)/config/pick_list_$(arg world).yaml"/>

```

The `/world_num` ROS parameter can be used in the main project node to automatically set the file name of, and write the world number to, to the YAML file. For example, see the following lines of code in `project_template.py`:

```
test_scene_num.data = rospy.get_param('/world_num');

filename = 'output' + str(test_scene_num.data) + '.yaml';
send_to_yaml(filename,dict_list);
```

## Results and future work
We trained a single SVM model using the world 3 data (which contains all objects) and 50 data points per object. Training accuracy was **84.58%**. See below for the confusion matrices.

![Confusion matrices for final trained model][confusion_matrices]

Generally, more training data means better results. An interesting observation from this project: When collecting more data (100 data points per object instead of 50, increased the training accuracy from 84.58% to 91.25%. However, classification results were not as accurate on the actual pick and place project environment.

**This trained model was able to correctly classify all objects in Worlds 1 through 3, as shown below.**

### World 1
![World 1 screenshot][world1]

### World 2
![World 2 screenshot][world2]

### World 3
![World 3 screenshot][world3]

Refer to the the YAML files in the `/pr2_robot/scripts` folder to see the resulting pick and place request information.

The classification integration with the pick and place server was implemented, but not fully -- therefore it is commented out in `project_template.py`. However, uncommenting the pick and place lines of code below works well enough to pick up and put down an object, keeping in mind that this project submission does not generate the collision map by turning the robot torso.

```
# Wait for 'pick_place_routine' service to come up
rospy.wait_for_service('pick_place_routine')

try:
    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

    # Insert your message variables to be sent as a service request
    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

    print ("Response: ",resp.success)

except rospy.ServiceException, e:
    print "Service call failed: %s"%e
```



