import os
import numpy as np


def process_gt_file(input_file, output_file):
    """
    Process gt.txt file and save the first half of frames

    Args:
        input_file (str): Path to input gt.txt file
        output_file (str): Path to output gt.txt file
    """
    # Read all lines from input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Convert lines to numpy array for easier processing
    data = np.array([line.strip().split(',') for line in lines]).astype(float)

    # Get unique frame IDs
    frame_ids = np.unique(data[:, 0])

    # Calculate middle frame
    mid_frame = frame_ids[len(frame_ids) // 2]

    # Filter data for second half frames
    last_half_data = data[data[:, 0] > mid_frame].copy()

    # Get minimum frame number for each track
    tracks = np.unique(last_half_data[:, 1])
    new_frame_nums = {}

    for track_id in tracks:
        track_mask = last_half_data[:, 1] == track_id
        track_frames = last_half_data[track_mask, 0]
        start_frame = np.min(track_frames)
        frame_nums = np.arange(1, len(track_frames) + 1)
        for old_frame, new_frame in zip(sorted(track_frames), frame_nums):
            new_frame_nums[(old_frame, track_id)] = new_frame

    # Update frame numbers
    for i in range(len(last_half_data)):
        old_frame = last_half_data[i, 0]
        track_id = last_half_data[i, 1]
        last_half_data[i, 0] = new_frame_nums[(old_frame, track_id)]

    # Sort by track_id and frame number
    sort_idx = np.lexsort((last_half_data[:, 0], last_half_data[:, 1]))
    last_half_data = last_half_data[sort_idx]

    # Save filtered data with original float format
    with open(output_file, 'w') as f:
        for row in last_half_data:
            # Only confidence is float, everything else is integer
            formatted_row = []
            for i, val in enumerate(row):
                if i == 8:  # confidence score
                    formatted_row.append(f"{val:.5f}")  # confidence with 5 decimal places
                elif i in [0, 1, 2, 3, 4, 5, 6, 7]:  # frame, track_id, bbox coordinates, and flags
                    formatted_row.append(f"{int(val)}")
            line = ','.join(formatted_row)
            f.write(f"{line}\n")
            f.write(f"{line}\n")


def process_mot_sequence(sequence_path):
    """
    Process a single MOT sequence

    Args:
        sequence_path (str): Path to sequence folder (e.g., MOT20-01)
    """
    gt_path = os.path.join(sequence_path, 'gt', 'gt.txt')
    if not os.path.exists(gt_path):
        print(f"Warning: gt.txt not found in {gt_path}")
        return

    # Create val_half directory if it doesn't exist
    val_dir = os.path.join(sequence_path, 'gt')
    os.makedirs(val_dir, exist_ok=True)

    # Process and save
    output_path = os.path.join(val_dir, 'gt_val_half.txt')
    process_gt_file(gt_path, output_path)
    print(f"Processed {sequence_path}")


def main():
    # Base directory containing MOT20-train
    base_dir = "data/gt/mot_challenge/MOT20-train"

    # Process each sequence
    for seq in os.listdir(base_dir):
        if seq.startswith('MOT20-'):
            sequence_path = os.path.join(base_dir, seq)
            process_mot_sequence(sequence_path)


if __name__ == "__main__":
    main()