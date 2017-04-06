
#!/bin/bash

# CISM-ALBANY

# run cism-albany after modifying (if needed) the paths of the input nc "name" file and the "dycore_input_file" in the file inputFiles/cism-albanyT.config.
cd inputFiles
../cism_driver cism-albanyT.config
cd ..

# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#epu --auto greenland_cism-albanyT.exo.4.

#note that if you diff the original greenland.nc file and the one stored by cism greenland_cism-albanyT.nc, beta changed a bit.

#STANDALONE-ALBANY

#-- Generation of ascii files using matlab.

#move to mFiles directory
cd mFiles

# modify (if needed) maltab script "build_cism_msh_from_nc" to fix input/output paths and filenames.
# run matlab script "build_cism_msh_from_nc"
matlab -nojvm < build_cism_msh_from_nc.m

#move back to top directory
cd ..

#create 2d exodus file for Greenland.
#Warning!! this part is very hacky, you'll get a runtime error, but the correct *.exo will be saved in the albanyMesh folder. Also, this can be extremely slow with large files, unless trilinos is compiled with the nodebug option -D CMAKE_CXX_FLAGS:STRING="-O3 -fPIC -fno-var-tracking -DNDEBUG".
./AlbanyT inputFiles/create2dExo.xml

#run standalone Albany simulation
./AlbanyT inputFiles/input_standalone-albanyT.xml 

# [optional] if you run the above on multiple processors, you need to merge the exodus files into one:
#$ path-to-trilinos-install/bin/epu --auto greenland_cism-albanyT.exo.4.

#COMPARE CISM-ALBANY with STANDALONE ALBANY
#move to mFiles directory
cd mFiles

#run the script compare_exos.m
matlab -nojvm < compare_exos.m

# you'll see the max difference (in absolut value) between fields. Note that the raher significant difference in beta comes from the fact that beta is changed in cism according to the floating condition.

#STORE STANDALONE ALBANY FIELDS INTO nc.
#create a copy of greenland.nc
cp ../ncGridSamples/greenland.nc ../greenland_standalone-albanyT.nc
matlab -nojvm < print_exo_fields_into_nc.m


#Note: When the thickness and the bedrock topography are interpolated back to the grid, some accuracy is lost (try comparing the original "greenland.nc" with the newly created "geenland_standalone-albanyT.nc"). In fact, if you now re-run cism-alabny #using the new nc grid you'll see a significant difference with the standalone albany solution:

#move back to top folder
cd ..

#after modifying the inputFiles/cism-albanyT.config to use the new gid greenland_standalone-albanyT.nc, run cism-albanyT, and compare again
cd inputFiles
../cism_driver cism-albanyT.config
cd ..

cd mFiles
matlab -nojvm < compare_exos.m

#quite a difference.. this is an interpolation error.. so it should diminish as the grid is refined.


