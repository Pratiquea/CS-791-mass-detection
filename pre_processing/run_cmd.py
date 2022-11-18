import os
import subprocess

cmd = ["df",  "/", "-h"]#, "| awk 'FNR == 2 {print $4}'"]
a = subprocess.Popen(cmd,stdout=subprocess.PIPE)
space_left = a.communicate()[0].decode('utf-8').split()[10]
space_left = space_left.replace("G", "")
print("space left = {}".format(space_left))