#!/usr/bin/env python3
import os
import apt

#######################
## Basic Variables ####
#######################
root = os.getcwd()

#######################
## Check Packages #####
#######################

cache = apt.Cache()

def check_package(package):
    if cache[package]:
        print(f"Found package: {package}")
    else:
        print(f"Could not find package: {package}")
        print("Exiting...")
        exit(-1)

#######################################
## Setup Project Structure ############
#######################################
print("Setting up Project Structure")
if not os.path.exists("build"):
    os.makedirs("build")

if not os.path.exists("packages/build"):
    os.makedirs("packages/build")

##################################
## Libraries #####################
##################################
print("Checking Libraries")

##################################
## Packages ######################
##################################

print("Checking Packages")

# Dealii
if os.path.exists("packages/dealii"):
    print("Found Package Dealii 9.4")
else:
    print("Downloading Package Dealii 9.4")
    os.system("git clone -b dealii-9.4 https://github.com/dealii/dealii.git packages/dealii")

if not os.path.exists("packages/build/dealii"):
    os.makedirs("packages/build/dealii")

dealii_source_dir = "packages/dealii"
dealii_build_dir = "packages/build/dealii"
dealii_variables = [
    "CMAKE_BUILD_TYPE=DebugRelease",
    "DEAL_II_WITH_CXX17=ON",
]

dealii_arguments = f"-S {dealii_source_dir} -B {dealii_build_dir}"
for var in dealii_variables:
    dealii_arguments += f" -D{var}"

os.system(f"cmake {dealii_arguments}")
os.system(f"cmake --build {dealii_build_dir} --target library -j4")

