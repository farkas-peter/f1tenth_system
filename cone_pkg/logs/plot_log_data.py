import json
import matplotlib.pyplot as plt

def load_log_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_data(log_dict):
    timestamps = []
    x_vals = []
    y_vals = []

    input_speed = []
    output_speed = []
    input_steering = []
    output_steering = []

    for t_str in sorted(log_dict.keys(), key=lambda x: float(x)):
        t = float(t_str)
        entry = log_dict[t_str]

        odom = entry['odom']
        pos = odom['position']

        timestamps.append(t)
        x_vals.append(pos['x'])
        y_vals.append(pos['y'])

        input_speed.append(entry['input_speed'])
        output_speed.append(entry['output_speed'])

        if entry['input_steering_angle'] > 0.85:
            input_steering.append(0.85)
        elif entry['input_steering_angle'] < 0.21:
            input_steering.append(0.21)
        else:
            input_steering.append(entry['input_steering_angle'])

        output_steering.append(entry['output_steering_angle'])

    return {
        "timestamps": timestamps,
        "x": x_vals,
        "y": y_vals,
        "input_speed": input_speed,
        "output_speed": output_speed,
        "input_steering": input_steering,
        "output_steering": output_steering
    }

def plot_all(data):
    # 1. X-Y pozíció (globális)
    plt.figure()
    plt.plot(data["x"], data["y"], label='trajectory')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Vehicle Global Position (Odometry)")
    plt.grid()
    plt.axis("equal")
    plt.legend()

    # 2. Sebesség
    plt.figure()
    plt.plot(data["timestamps"], data["input_speed"], label='input_speed')
    plt.plot(data["timestamps"], data["output_speed"], label='output_speed')
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed over Time")
    plt.grid()
    plt.legend()

    # 3. Kormányzási szög
    plt.figure()
    plt.plot(data["timestamps"], data["input_steering"], label='input_steering')
    plt.plot(data["timestamps"], data["output_steering"], label='output_steering')
    plt.xlabel("Time [s]")
    plt.ylabel("Steering Angle [rad]")
    plt.title("Steering Angle over Time")
    plt.grid()
    plt.legend()

    plt.show()

if __name__ == "__main__":
    log_data = load_log_file("log_20250624_104317.json")  # ← fájlnevet cseréld ki szükség szerint
    processed_data = extract_data(log_data)
    plot_all(processed_data)
