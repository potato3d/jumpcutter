import os
from pathlib import Path
import subprocess
from time import perf_counter
import sys

SRC_DIRS = [
    r"c:/input_dir/"
    ]

OUTFILE_SUFFIX = '_COMPACT'

total_start = perf_counter()
for dir in SRC_DIRS:
    dirpath = Path(dir)
    for filepath in dirpath.rglob('*.mp4'):
        print(filepath)
        if filepath.stem.endswith(OUTFILE_SUFFIX):
            continue
        outfilepath = filepath.with_name(filepath.stem + OUTFILE_SUFFIX + '.mp4')
        if outfilepath.exists():
            continue
        tstart = perf_counter()
        subprocess.run(args=[sys.executable, 'jumpcut_file.py', '-i', str(filepath), '-o', str(outfilepath)]) # sys.executable makes sure we use the virtualenv interpreter
        tstop = perf_counter()
        print("dir time:", tstop - tstart)
total_stop = perf_counter()
print("total time:", total_stop - total_start)