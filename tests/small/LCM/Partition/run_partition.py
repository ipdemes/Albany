##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0
    
log_file_name = "partition.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run the partition test
command = ["./PartitionTest"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-stat", "-f", \
               "partition.exodiff", \
               "partition.gold.e", \
               "output.e"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code
    
sys.exit(result)
