
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test - Parallel Cubes 1 proc"
name = "Parallel_Cubes_1"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

#specify tolerance to determine test failure / passing
tolerance = 1.0e-8; 
meanvalue = 0.000594484007237;

# run AlbanyT 
command = ["./AlbanyT", "cubes.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

for line in open(log_file_name):
  if "Main_Solve: MeanValue of final solution" in line:
    s = line
    s = line[40:]
    d = float(s)
    print d
    if (d > meanvalue + tolerance or d < meanvalue - tolerance):
      result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
