import json
import matplotlib.pyplot as plt
import math

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
            entry["output_steering_angle"] is None or
            entry["target_point"] is None
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
            if d <= 1.0:
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
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def plot_all(data):
    label_size = 18
    tick_size = 14
    legend_size = 14

    cones_x = [1.6,1.6,4.1,4.1,6.6,6.6,9.1,9.1,11.6,11.6]
    cones_y = [-0.35,0.35,-0.25,0.45,-0.45,0.25,-0.55,0.15,-0.25,0.45]

    # 1. X-Y pozíció (globális)
    plt.figure(figsize=(18, 6))
    plt.plot(data["x"], data["y"], label='Ego trajectory')
    plt.scatter(cones_x, cones_y, label='Cones', s=100, marker = 'o', color = 'orange')
    #plt.plot(data["x_ekf"], data["y_ekf"], label='trajectory with imu')
    if data["target_x"] and data["target_y"]:
        plt.scatter(data["target_x"][:-8], data["target_y"][:-8], label='Target points', color='green', s=30, marker='x')
    plt.xlabel("X [m]",fontsize =label_size)
    plt.ylabel("Y [m]",fontsize =label_size)
    plt.title("Vehicle Global Position (Odometry)",fontsize =label_size)
    plt.grid()
    plt.xticks(fontsize =tick_size)
    plt.yticks(fontsize =tick_size)
    plt.legend(fontsize =legend_size)

    """
    # 2. Sebesség
    plt.figure(figsize=(10, 4))
    plt.plot(data["timestamps"], data["input_speed"], label='input_speed')
    plt.plot(data["timestamps"], data["output_speed"], label='output_speed')
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed over Time")
    plt.grid()
    plt.legend()
    """
    
    # 3. Kormányzási szög
    plt.figure(figsize=(18, 6))
    converted_output_steering_deg = [
        ((val - 0.5235) / -1.2135) * (180.0 / math.pi)
        for val in data["output_steering"]
    ]
    #plt.plot(data["timestamps"], data["input_steering"], label='input_steering')
    plt.plot(data["timestamps"], converted_output_steering_deg, label='Steering angle')
    plt.xlabel("Time [s]",fontsize =label_size)
    plt.ylabel("Steering Angle [°]",fontsize =label_size)
    plt.title("Steering Angle over Time",fontsize =label_size)
    plt.grid()
    plt.xticks(fontsize =tick_size)
    plt.yticks(fontsize =tick_size)
    plt.legend(fontsize =legend_size)

    plt.show()

if __name__ == "__main__":
    log_data = load_log_file("log_20250625_143923.json")  # ← fájlnevet cseréld ki szükség szerint
    processed_data = extract_data(log_data)
    plot_all(processed_data)
