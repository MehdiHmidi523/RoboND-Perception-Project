#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    pt_cloud = ros_to_pcl(pcl_msg)
    
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

    # Euclidean Clustering
    cloud_xyz = XYZRGB_to_XYZ(cloud_outliers);
    tree = cloud_xyz.make_kdtree();   
    ec = cloud_xyz.make_EuclideanClusterExtraction();
    ec.set_ClusterTolerance(0.05);	# Distance tolerance [m]
    ec.set_MinClusterSize(100);		# Minimum cluster size
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

# Exercise-3 TODOs:

    # Classify the clusters! 
    # Loop through each detected cluster one at a time
    detected_objects_labels = []
    detected_objects = []
    for idx, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_outliers.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        feat_cloud = pcl_cluster.to_array()
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = classifier.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_xyz[pts_list[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label,label_pos,idx))

        # Add the detected object to the list of detected objects.
        det_obj = DetectedObject()
        det_obj.label = label
        det_obj.cloud = ros_cluster
        detected_objects.append(det_obj)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    dict_list = [];

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list');

    # Parse parameters into individual variables
    test_scene_num = Int32();
    test_scene_num.data = rospy.get_param('/world_num');
    dropbox_data = rospy.get_param('/dropbox');
    object_name = String();
    pick_pose = Pose();
    place_pose = Pose();
    arm_name = String();

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for pick_object in object_list_param:

        # Get the PointCloud for a given object and obtain its label and centroid
        centroid = [];
        for object in object_list:
            if object.label == pick_object['name']:
                object_name.data = str(object.label);
                points_arr = ros_to_pcl(object.cloud).to_array();
                centroid = np.mean(points_arr,axis=0)[:3];
        if len(centroid) == 0:
            continue;

        # Create 'pick_pose' for the object
        pick_pose.position.x = np.asscalar(centroid[0]);
        pick_pose.position.y = np.asscalar(centroid[1]);
        pick_pose.position.z = np.asscalar(centroid[2]);

        # Assign the 'arm_name' and 'place_pose'
        for db in dropbox_data:
            if db['group'] == pick_object['group']:
                dropbox_arm = db['name'];
                dropbox_loc = db['position'];
        arm_name.data = str(dropbox_arm);
        place_pose.position.x = dropbox_loc[0];
        place_pose.position.y = dropbox_loc[1];
        place_pose.position.z = dropbox_loc[2];

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict);

        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        #try:
        #    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
        #    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            #print ("Response: ",resp.success)

        #except rospy.ServiceException, e:
            #print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    filename = 'output' + str(test_scene_num.data) + '.yaml';
    send_to_yaml(filename,dict_list);

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition',anonymous=True);

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2,pcl_callback,queue_size=1)

    # Create Publishers
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2,queue_size=1);
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1);
    detected_objects_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1);

    # Load Model From disk
    model = pickle.load(open('model.sav','rb'))
    classifier = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin();
