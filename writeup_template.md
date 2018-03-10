# Project: Perception Pick & Place

### Submission for Udacity Robotics Software Engineer Nanodegree
### Sebastian Castro - 2018

[//]: # (Image References)

[confusion_matrices]: ./misc_images/confusion_matrices.PNG
[world1]: ./misc_images/World1.PNG
[world2]: ./misc_images/World2.PNG
[world3]: ./misc_images/World3.PNG

---

## Introduction
Describe the problem


## Perception Pipeline Methodology
Here, we will discuss the steps taken in the perception pipeline and the parameters selected at each step.

### STEP 1: Point Cloud Processing

### STEP 2: Point Cloud Clustering

### STEP 3: Object Detection
The features used were:
* HSV color space with 25 bins for each channel
* Normals only in the Z direction, with 5 bins

HSV performed better than RGB for separating objects that were more similar in color. This is because the saturation and value channels could be used to better separate these objects. 25 color bins were selected by changing the numbers of color bins and checking the test accuracy. We also found that the normal data contributed to overfitting, especially since the training data used many more orientations than the actual project environment. Due to this, we only created features out of the Z-direction normals, and picked a smaller number of bins.

The SVM kernel used was linear. Polynomial order 2+ yielded worse results. RBF gave slightly better test results, but empirically
performed worse on the actual project. This is likely because radial basis functions are more prone to overfitting since they can create more complex 
decision boundaries than a linear SVM.

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
Arguments were added to the capture_features.py from Exercise 3 to make training data collection easier. The 2 option arguments are
* World Number (default = 1)
* Number of point clouds per object (default = 10)

For example, to capture training features for all objects in World 3, with 50 point clouds for each object:
```
rosrun pr2_robot capture_features.py 3 50
```

### Launch File and Results File Automation
Added argument to the `pick_place.launch` file, which sets the world number, pick list file, and creates a ROS parameter on the parameter server with the world number. If no argument is specified, world 1 will be picked.


```
roslaunch pr2_robot pick_place_project.launch world:=2
```

This ROS parameter can be used in the main project file to automatically set the file name of, and write the world number to, to the YAML file.


## Results and future work
We trained a single SVM using the world 3 data (which contains all objects) and 50 data points per object. Training accuracy was 84.58%. See below for the confusion matrices.

![Confusion matrices for final trained model][confusion_matrices]

An interesting observation: We also tried collecting more data (100 data points per object). While the training accuract went up to 91%, results were not as accurate on the actual project environment.

World 1
![World 1 screenshot][world1]

World 2
![World 2 screenshot][world2]

World 3
![World 3 screenshot][world3]

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

---

# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



