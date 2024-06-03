import os
import argparse

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Process the video ID for the AI model.')
parser.add_argument('video_id', type=str, help='Video ID to process')
args = parser.parse_args()

# Assuming the video files are stored in /app/videos/
video_directory = "/app/videos/"
video_filename = f"{args.video_id}.mp4"  # Assuming videos are in mp4 format
video_path = os.path.join(video_directory, video_filename)

# Now video_path contains the path to the specific video file
print(f"Video path: {video_path}")