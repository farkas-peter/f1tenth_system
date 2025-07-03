import json
import matplotlib.pyplot as plt
import math
import numpy as np

def load_log_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_data(log_dict):
    timestamps = []
    x_vals = []
    y_vals = []
    x_ekf = []
    y_ekf = []

    input_speed = []
    output_speed = []
    input_steering = []
    output_steering = []

    target_x = []
    target_y = []

    for t_str in sorted(log_dict.keys(), key=lambda x: float(x)):
        t = float(t_str)
        entry = log_dict[t_str]

        if (
            entry["odom"] is None or
            entry["ekf_odom"] is None or
            entry["input_speed"] is None or
            entry["output_speed"] is None or
            entry["input_steering_angle"] is None or
            entry["output_steering_angle"] is None
        ):
            continue

        odom = entry['odom']
        pos = odom['position']

        ekf_odom = entry['ekf_odom']
        ekf_pos = ekf_odom['position']

        timestamps.append(t)
        x_vals.append(pos['x'])
        y_vals.append(pos['y'])
        x_ekf.append(ekf_pos['x'])
        y_ekf.append(ekf_pos['y'])

        input_speed.append(entry['input_speed'])
        output_speed.append(entry['output_speed'])

        if entry['input_steering_angle'] > 0.85:
            input_steering.append(0.85)
        elif entry['input_steering_angle'] < 0.21:
            input_steering.append(0.21)
        else:
            input_steering.append(entry['input_steering_angle'])

        output_steering.append(entry['output_steering_angle'])


        if "target_point" in entry and "position" in entry["target_point"]:
            tp = entry["target_point"]["position"]
            tx = tp['x']
            ty = tp['y']

            # Az aktuális járműpozícióhoz mért távolság
            d = distance(tx, ty, pos['x'], pos['y'])
            if d <= 1.0 and d != 0.0:
                target_x.append(tx)
                target_y.append(ty)
        """
        if "target_point" in entry and "position" in entry["target_point"]:
            tp = entry["target_point"]["position"]
            target_x.append(tp['x'])
            target_y.append(tp['y'])
        """

    return {
        "timestamps": timestamps,
        "x": x_vals,
        "y": y_vals,
        "x_ekf": x_ekf,
        "y_ekf": y_ekf,
        "input_speed": input_speed,
        "output_speed": output_speed,
        "input_steering": input_steering,
        "output_steering": output_steering,
        "target_x": target_x,
        "target_y": target_y
    }

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def plot_all(data):
    label_size = 18
    tick_size = 14
    legend_size = 14

    cones_x = [1.2,1.25,2.93,3.24,4.3,4.81,5.59,6.33,6.86,7.72,7.83,8.56]
    cones_y = [0.40,-0.25,0.67,0.05,1.25,0.784,2.57,1.97,4.07,3.54,5.97,5.68]

    # 1. X-Y pozíció (globális)
    plt.figure(figsize=(18, 6))
    plt.plot(data["x"], data["y"], label='Ego trajectory')
    #plt.scatter(cones_x, cones_y, label='Cones', s=100, marker = 'o', color = 'orange')
    #plt.plot(data["x_ekf"], data["y_ekf"], label='trajectory with imu')
    if data["target_x"] and data["target_y"]:
        plt.scatter(data["target_x"], data["target_y"], label='Target points', color='green', s=30, marker='x')
    plt.xlabel("X [m]",fontsize =label_size)
    plt.ylabel("Y [m]",fontsize =label_size)
    plt.title("Vehicle Global Position (Odometry)",fontsize =label_size)
    plt.grid()
    plt.axis("equal")
    plt.xticks(fontsize =tick_size)
    plt.yticks(fontsize =tick_size)
    #plt.xlim(-1,10)
    plt.legend(fontsize =legend_size)

    
    timestamps = np.array(data["timestamps"])
    op_speed_raw = np.array(data["output_speed"])
    ip_speed_raw = np.array(data["input_speed"])
    mask = timestamps <= 21.0
    filtered_timestamps = timestamps[mask]
    filtered_op_speed = op_speed_raw[mask]
    filtered_ip_speed = ip_speed_raw[mask]
    # 2. Sebesség
    plt.figure(figsize=(18, 6))
    plt.plot(filtered_timestamps, filtered_ip_speed, label='Speed demand')
    plt.plot(filtered_timestamps, filtered_op_speed, label='Actual speed')
    plt.xlabel("Time [s]",fontsize =label_size)
    plt.ylabel("Speed [RPM]",fontsize =label_size)
    plt.title("Speed over Time",fontsize =label_size)
    plt.grid()
    plt.xticks(fontsize =tick_size)
    plt.yticks(fontsize =tick_size)
    plt.legend(fontsize =legend_size)
    
    
    # 3. Kormányzási szög
    """
    timestamps = np.array(data["timestamps"])
    steering_raw = np.array(data["output_steering"])

    mask = timestamps <= 19.6
    filtered_timestamps = timestamps[mask]
    filtered_steering = ((steering_raw[mask] - 0.5235) / -1.2135) * (180.0 / math.pi)
    
    plt.figure(figsize=(18, 6))
    #plt.plot(data["timestamps"], data["input_steering"], label='input_steering')
    plt.plot(filtered_timestamps, filtered_steering, label='Steering angle')
    plt.xlabel("Time [s]",fontsize =label_size)
    plt.ylabel("Steering Angle [°]",fontsize =label_size)
    plt.title("Steering Angle over Time",fontsize =label_size)
    plt.grid()
    plt.xticks(fontsize =tick_size)
    plt.yticks(fontsize =tick_size)
    plt.legend(fontsize =legend_size)
    """
    plt.show()

if __name__ == "__main__":
    log_data = load_log_file("log_20250703_130807.json")  # ← fájlnevet cseréld ki szükség szerint
    processed_data = extract_data(log_data)
    plot_all(processed_data)
