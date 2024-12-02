
from rosbags.rosbag1 import Reader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as Header
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as Pcl2
from rosbags.typesys.types import sensor_msgs__msg__PointField as Field
from rosbags.typesys.types import builtin_interfaces__msg__Time as Ts

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

def set_timestamp(raw_timestamp):
    # 後ろ9文字を取り出す
    raw_timestamp = str(raw_timestamp)
    last_9_digits = raw_timestamp[-9:]

    # 残りの部分を取り出す
    first_part = raw_timestamp[:-9]

    return int(first_part), int(last_9_digits)

# Create a typestore and get the string class.
typestore = get_typestore(Stores.ROS1_NOETIC) # ROSbagをrecordしたときのROSversionに指定する
Pointcloud = typestore.types['sensor_msgs/msg/PointCloud2']

with open('config.json', 'r') as f:
    config = json.load(f)

# settings
topic_length = 22
start_time = None
reference_file = "./11_15_benign.csv"
spoofing_mode = config['main']['spoofing_mode'] # removal or static or dynamic
spoofer_x = config['main']['spoofer_x']
spoofer_y = config['main']['spoofer_y']
distance_threshold = config['main']['distance_threshold'] # unit:m

dataframe_reference = load_reference(reference_file)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Create reader instance and open for reading.
with Reader('./11_15.bag') as reader:
    with Writer('./11_15_2.bag') as writer:
        topic = str(config['main']['spoofed_topic'])
        type_pcl2 = Pcl2.__msgtype__
        conn = writer.add_connection(topic, Pcl2.__msgtype__, typestore=typestore)

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

                spoofed_cloud = np.vstack((x_spoofed, y_spoofed, z_spoofed)).T
                points = spoofed_cloud.reshape(-1).view(np.uint8)

                ts_first, ts_last = set_timestamp(timestamp)

                MSG = Pcl2(
                Header(seq=msg.header.seq, stamp=Ts(sec=ts_first, nanosec=ts_last), frame_id='velodyne'),
                #header=Hdr(seq=msg.header.seq, stamp=Ts(sec=ts_first, nanosec=ts_last), frame_id='velodyne'),
                height=1, width=int(spoofed_cloud.shape[0]),
                fields=[
                    Field(name='x', offset=0, datatype=7, count=1),
                    Field(name='y', offset=4, datatype=7, count=1),
                    Field(name='z', offset=8, datatype=7, count=1)],
                is_bigendian=False, point_step=12,
                row_step=int(spoofed_cloud.shape[0]) * 12, data=points, is_dense=True)

                #writer.write(connection, timestamp, typestore.serialize_ros1(MSG, type_pcl2))
                writer.write(conn, timestamp, cdr_to_ros1(serialize_cdr(MSG, type_pcl2), type_pcl2))
                
                ax.scatter(x_spoofed, y_spoofed, s=1)
                ax2.plot(np.array(dataframe_reference['x']), np.array(dataframe_reference['y']))
                ax2.scatter(odom_x, odom_y, marker='*', s=50, color='red')
                ax2.scatter(spoofer_x, spoofer_y, marker='*', s=50, color='darkblue')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                plt.pause(0.05)

plt.show()
