
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def binary_to_xyz(binary):
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)

    return x.flatten(), y.flatten(), z.flatten()

# Create a typestore and get the string class.
typestore = get_typestore(Stores.ROS1_NOETIC) # ROSbagをrecordしたときのROSversionに指定する

# settings
topic_length = 22
time_array = []
x_vel, y_vel, z_vel = [], [], []
IMU_start_time = None
visualize_mode = 'acc' # mode 'vel' or 'acc'

# graph settings
fig = plt.figure(figsize=(12,9))

gs_master = GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0, :])
gs_right = GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_master[0, 1])
gs_under = GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs_master[1, :])

fig.suptitle('IMU topic viewer')
ax1 = fig.add_subplot(gs_1[:, 0])
ax2 = fig.add_subplot(gs_right[0, :])
ax3 = fig.add_subplot(gs_right[1, :])
ax4 = fig.add_subplot(gs_right[2, :])
ax5 = fig.add_subplot(gs_under[0, :])
ax6 = fig.add_subplot(gs_under[1, :])
ax7 = fig.add_subplot(gs_under[2, :])

# Create reader instance and open for reading.
with Reader('/home/rokuto/11_15.bag') as reader:

    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():
        
        if connection.topic == '/velodyne_points':
            ax1.cla()
            
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            iteration = int(len(msg.data)/topic_length) # velodyne:22, livox:18
            bin_points = np.frombuffer(msg.data, dtype=np.uint8).reshape(iteration, topic_length) # velodyne:22, livox:18 
            x, y, z = binary_to_xyz(bin_points)

            ax1.scatter(x, y, s=1)

        elif connection.topic == '/imu':
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax5.cla()
            ax6.cla()
            ax7.cla()

            if visualize_mode == 'vel':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                angular_vel_x, angular_vel_y, angular_vel_z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

                x_vel.append(angular_vel_x)
                y_vel.append(angular_vel_y)
                z_vel.append(angular_vel_z)

                if len(time_array) == 0:
                    IMU_start_time = time
                    time_array.append(time - IMU_start_time)
                else:
                    time_array.append(time - IMU_start_time)
                
                if len(x_vel) % 10 == 0:

                    ax2.plot(time_array, x_vel)
                    ax2.set_ylabel('vel_x (m/s)')
                    ax3.plot(time_array, y_vel)
                    ax3.set_ylabel('vel_y (m/s)')
                    ax4.plot(time_array, z_vel)
                    ax4.set_ylabel('vel_z (m/s)')
                    ax4.set_xlabel('time (s)')

                    plt.tight_layout()
                    plt.pause(0.001)

            elif visualize_mode == 'acc':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                angular_acc_x, angular_acc_y, angular_acc_z = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z

                x_vel.append(angular_acc_x)
                y_vel.append(angular_acc_y)
                z_vel.append(angular_acc_z)

                if len(time_array) == 0:
                    IMU_start_time = time
                    time_array.append(time - IMU_start_time)
                else:
                    time_array.append(time - IMU_start_time)
                
                if len(x_vel) % 10 == 0:

                    ax2.plot(time_array[-100:], x_vel[-100:])
                    ax2.set_ylabel('acc_x (m/s^2)')
                    ax3.plot(time_array[-100:], y_vel[-100:])
                    ax3.set_ylabel('acc_y (m/s^2)')
                    ax4.plot(time_array[-100:], z_vel[-100:])
                    ax4.set_ylabel('acc_z (m/s^2)')
                    ax4.set_xlabel('time (s)')

                    ax5.plot(time_array, x_vel)
                    ax5.set_ylabel('acc_x (m/s^2)')

                    ax6.plot(time_array, y_vel)
                    ax6.set_ylabel('acc_y (m/s^2)')
                    
                    ax7.plot(time_array, z_vel)
                    ax7.set_ylabel('acc_z (m/s^2)')

                    plt.tight_layout()
                    plt.pause(0.001)

plt.tight_layout()
plt.show()