#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 - Uniform Traction
######################
print "test 1 - Uniform Traction"
name = "Cohesive"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT 
command = ["./AlbanyT", "input" + name + ".yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run algebra
command = './algebra out'+name+'.e out'+name+'.alg.e < alg.i > '+log_file_name
p = Popen(command, shell=True)
return_code = p.wait()
if return_code != 0:
    result = return_code


# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           "out" + name + ".alg.gold.e", \
           "out" + name + ".alg.e"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
