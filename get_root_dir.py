'''
Module for determining the root directory so that code can be run on pancreas or remotely from MB Pro
'''


import socket
import os
import subprocess


# Return the MATLAB working folder depending on host system (My MacBook or Pancreas/sub-hosts)
def get_mat_root():
    hname = socket.gethostname()
    if "Trevors" in hname:
        # mat_root1 = "/Users/hiltontj/Documents/MATLAB/pancreas/" # ** need a work-around for this to include spook
        mat_root = "/Users/hiltontj/Documents/MATLAB/spook/"
        
        # Check if the mounted pancreas folder is empty, and if so, remount it
        if 'pancreas' in mat_root and os.listdir(mat_root) == []:
            print("Re-mounting Pancreas...")
            subprocess.call("/Users/hiltontj/Documents/scripts/unmountpancreas.sh")
            subprocess.call("/Users/hiltontj/Documents/scripts/mountpancreas.sh")
            print("Pancreas re-mounted.")

    elif hname in ['pancreas', 'sheldon', 'leonard', 'bernadette', 'howard', 'raj', 'penny']:
        mat_root = "/home/hiltontj/Documents/MATLAB/Thesis/work/"

    elif hname in 'pepi':
        mat_root = "/home/trevor/Documents/MATLAB/Thesis/work/"

    return mat_root
