#!/usr/bin/env python
import numpy as np
import pickle
import rospy
import sys

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    # Select the models based on the world number and number of samples 
    # per model specified in arguments.
    # If no argument specified, default to world 1 and 10 samples
    num_args = len(sys.argv)
    if num_args == 1:
        world_num = 1;
        samples_per_model = 10;
    elif num_args == 2:
        world_num = int(sys.argv[1]);
        samples_per_model = 10;
    else:
        world_num = int(sys.argv[1]);
        samples_per_model = int(sys.argv[2]);    

    # From the corresponding pick_list_*.yaml files
    if world_num == 1:
        models = ['biscuits','soap','soap2'];
    elif world_num == 2:
        models = ['biscuits','soap','soap2','book','glue'];
    elif world_num == 3:
        models = ['biscuits','soap','soap2','book','glue',\
                  'sticky_notes','snacks','eraser'];

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(samples_per_model):

            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

