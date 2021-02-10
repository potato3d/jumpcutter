# jumpcutter
Trim videos by removing silent parts.
This is my own customization and improvements upon https://github.com/lamaun/jumpcutter/

# Notes
* Requires Python 3.
* Requires ffmpeg in path. Will start several subprocesses that call ffmpeg.
* Tested with hundreds of files ranging from a few megs (minutes) to gigabytes (hours).
* As the program runs, it saves every frame of the video as an image file in a temporary folder. If your video is long, this could take a LOT of space.

# Changes
* Improved performance and memory use of original script (jumpcut_file.py).
* Removed ability to speed-up video to improve performance (can always speed up during playback).
* Changed temporary folder to be where the script is (instead of OS-specific temp).
* Added another script to run in batch over a list of folders (jumpcutter.py that automatically calls jumpcut_file.py).

# Usage
To run on a single video file:

python3 jumpcut_file.py -i input_file -o output_file [args]

To run on all video files within several directories:

python3 jumpcutter.py

The input paths as well as the command-line arguments to jumpcut_file.py are hard-coded in jumpcutter.py. Feel free to change that to forward command-line arguments to jumpcut_file.py.
