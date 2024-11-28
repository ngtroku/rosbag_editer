
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spoofing_sim
import json

def binary_to_xyz(binary):
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()

def load_reference(csv_file_name):
    df = pd.read_csv(csv_file_name)
    return df

def compare_reference(rosbag_time, dataframe_reference):
    reference_time = dataframe_reference['timestamp']
    x, y = np.array(dataframe_reference['x']), np.array(dataframe_reference['y'])

    # find corresponding timestamp
    time_diff = np.abs(reference_time - rosbag_time)
    corresponding_index = np.argmin(time_diff)

    return x[corresponding_index], y[corresponding_index]

def check_spoofing_condition(odom_x, odom_y, spoofer_x, spoofer_y, distance_threshold):
    dist_spoofer_to_robot = ((odom_x - spoofer_x) ** 2 + (odom_y - spoofer_y) ** 2) ** 0.5

    if dist_spoofer_to_robot <= distance_threshold:
        return True
    else:
        return False
    
def decide_spoofing_param(odom_x, odom_y, spoofer_x, spoofer_y):
    spoofing_angle = np.degrees(np.arctan2(odom_y - spoofer_y, odom_x - spoofer_x)) + 180
    return spoofing_angle

# Create a typestore and get the string class.
typestore = get_typestore(Stores.ROS1_NOETIC) # ROSbagをrecordしたときのROSversionに指定する

with open('config.json', 'r') as f:
    config = json.load(f)

# settings
topic_length = 22
start_time = None
reference_file = "/home/rokuto/rosbag_editer/11_15_benign.csv"
spoofing_mode = config['main']['spoofing_mode'] # removal or static or dynamic
spoofer_x = config['main']['spoofer_x']
spoofer_y = config['main']['spoofer_y']
distance_threshold = config['main']['distance_threshold'] # unit:m

dataframe_reference = load_reference(reference_file)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Create reader instance and open for reading.
with Reader('/home/rokuto/11_15.bag') as reader:

    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():

        if connection.topic == config['main']['lidar_topic']:
            ax.cla()
            ax2.cla()
            now_time = timestamp/1e9

            if start_time == None:
                start_time = now_time

            rosbag_time = now_time - start_time
            odom_x, odom_y = compare_reference(rosbag_time, dataframe_reference)
            is_spoofing = check_spoofing_condition(odom_x, odom_y, spoofer_x, spoofer_y, distance_threshold)
            
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            iteration = int(len(msg.data)/topic_length) # velodyne:22, livox:18
            bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length) # velodyne:22, livox:18 
            x, y, z = binary_to_xyz(bin_points)
            coordinate_array = np.vstack((x, y, z)).T

            if is_spoofing and spoofing_mode == "removal":
                spoofing_angle = decide_spoofing_param(odom_x, odom_y, spoofer_x, spoofer_y)
                x_spoofed, y_spoofed, z_spoofed = spoofing_sim.spoof_main(coordinate_array, spoofing_angle, config['main']['spoofing_range'])

            elif is_spoofing and spoofing_mode == "static":
                spoofing_angle = decide_spoofing_param(odom_x, odom_y, spoofer_x, spoofer_y)
                x_spoofed, y_spoofed, z_spoofed = spoofing_sim.injection_main(coordinate_array, spoofing_angle, config['main']['spoofing_range'], config['main']['wall_dist'])

            elif is_spoofing and spoofing_mode == "dynamic":
                spoofing_angle = decide_spoofing_param(odom_x, odom_y, spoofer_x, spoofer_y)
                x_spoofed, y_spoofed, z_spoofed = spoofing_sim.dynamic_injection_main(coordinate_array, now_time, spoofing_angle, config['main']['spoofing_range'])
            
            else:
                x_spoofed, y_spoofed, z_spoofed = x, y, z
            
            ax.scatter(x_spoofed, y_spoofed, s=1)
            ax2.plot(np.array(dataframe_reference['x']), np.array(dataframe_reference['y']))
            ax2.scatter(odom_x, odom_y, marker='*', s=50, color='red')
            ax2.scatter(spoofer_x, spoofer_y, marker='*', s=50, color='darkblue')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            plt.pause(0.05)

plt.show()
