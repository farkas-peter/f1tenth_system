import sys
import h5py
import numpy as np


def analyze_timestamps(name, timestamps):
    timestamps = np.asarray(timestamps, dtype=np.int64)

    if len(timestamps) < 2:
        print(f"{name}: Not enough samples.")
        return

    print(f"\n--- {name} ---")
    print(f"Samples:       {len(timestamps)}")

    # ---------------------------------------------------------
    # Check invalid timestamps
    # ---------------------------------------------------------

    zero_indices = np.where(timestamps == 0)[0]

    if len(zero_indices) > 0:
        print(f"WARNING: {len(zero_indices)} zero timestamps found!")
        print(f"Indices: {zero_indices[:20]}")

    # Differences directly in nanoseconds to avoid float issues
    dt_ns = np.diff(timestamps)

    negative_indices = np.where(dt_ns < 0)[0]
    zero_dt_indices = np.where(dt_ns == 0)[0]

    #if len(negative_indices) > 0:
        #print(f"WARNING: {len(negative_indices)} backwards timestamp jumps found!")

        #for i in negative_indices[:10]:
            #print(f"  [{i}] -> [{i+1}]: {timestamps[i]} -> {timestamps[i+1]}")

    if len(zero_dt_indices) > 0:
        print(f"WARNING: {len(zero_dt_indices)} duplicate timestamps found!")

    # ---------------------------------------------------------
    # Only use valid consecutive intervals for FPS statistics
    # ---------------------------------------------------------

    valid_dt_ns = dt_ns[dt_ns > 0]

    if len(valid_dt_ns) == 0:
        print("No valid timestamp intervals.")
        return

    dt_ms = valid_dt_ns / 1e6

    mean_dt_s = np.mean(valid_dt_ns) / 1e9
    fps = 1.0 / mean_dt_s

    valid_duration_s = np.sum(valid_dt_ns) / 1e9

    print(f"Valid intervals: {len(valid_dt_ns)}")
    print(f"Valid duration:  {valid_duration_s:.3f} s")
    print(f"Average FPS:     {fps:.3f} Hz")
    print(f"Average dt:      {np.mean(dt_ms):.3f} ms")
    print(f"Median dt:       {np.median(dt_ms):.3f} ms")
    print(f"Minimum dt:      {np.min(dt_ms):.3f} ms")
    print(f"Maximum dt:      {np.max(dt_ms):.3f} ms")
    print(f"Std dt:          {np.std(dt_ms):.3f} ms")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 check_fps.py <recording.h5>")
        sys.exit(1)

    filename = sys.argv[1]

    with h5py.File(filename, "r") as f:

        if "color/timestamps" in f:
            analyze_timestamps(
                "Color",
                f["color/timestamps"][:]
            )

        if "depth/timestamps" in f:
            analyze_timestamps(
                "Depth",
                f["depth/timestamps"][:]
            )

        if "imu/timestamps" in f:
            analyze_timestamps(
                "IMU",
                f["imu/timestamps"][:]
            )


if __name__ == "__main__":
    main()