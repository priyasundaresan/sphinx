import subprocess
import argparse
import os

def merge_rollout_mp4s(rollout_dir):
    # Ensure the directory exists
    if not os.path.isdir(rollout_dir):
        raise ValueError(f"Directory {rollout_dir} does not exist.")
    
    # List all MP4 files in the directory
    mp4_files = [f for f in os.listdir(rollout_dir) if f.endswith('.mp4')]
    
    if not mp4_files:
        raise ValueError("No MP4 files found in the directory.")
    
    # Create the file list for FFmpeg
    filelist_path = os.path.join(rollout_dir, 'filelist.txt')
    with open(filelist_path, 'w') as filelist:
        for mp4_file in sorted(mp4_files):
            #filelist.write(f"file '{os.path.join(rollout_dir, mp4_file)}'\n")
            filelist.write(f"file '{mp4_file}'\n")

    # Define the output file path
    output_file = os.path.join(rollout_dir, 'merged.mp4')
    
    # Run FFmpeg to merge the files
    command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', filelist_path,
        '-c', 'copy', output_file
    ]
    print(command)
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully merged MP4 files into {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed with error: {e}")
    finally:
        # Clean up the temporary file list
        if os.path.exists(filelist_path):
            os.remove(filelist_path)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Merge all MP4 files in a directory into a single MP4 file.")
    parser.add_argument('-d', '--dir', type=str, required=True, help="Directory containing MP4 files to merge")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the merge function with the provided directory
    merge_rollout_mp4s(args.dir)
