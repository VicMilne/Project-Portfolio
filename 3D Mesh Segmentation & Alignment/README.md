# 3D Mesh Segmentation Optimizing For Alignment

## Overview

This project is the resulting final product of my undergraduate thesis working with the Bernini's Bronze technical study. The goal of my research was to explore the detection and analysis of joints in bronze multiplies (bronzes originating from the same original model). Since the process of creating a multiple usually involves piece-moulding, or casting parts individually and then joining them together in a multi-step process, a set of bronze multiples often contains subtle differences in the orientation of parts around particular seams.

The approach taken in this project involves randomly generating potential planar seams to divide a mesh and then selecting seams that minimize the alignment error between the segmented multiple and another reference multiple. Effectively, a selected seam would split the mesh in two, providing an additional degree of freedom by allowing the two parts to rotate and translate independently. Theoretically, an optimal segmentation would occur exactly on the existing seams, as this would allow the otherwise identical parts of each multiple to align perfectly with each other.

## Data Cleansing and Preparation

- This process process accepts .obj files as input, specifically containing the minimum amount of information
    - should omit normals and texture mapping
- Given the time complexity is O(n^2 * log(n)), downsampling complex meshes is advisable
    - Given the goal of this process is macro-level alignment, sacrificing detail is generally acceptable
- non-euclidean geometry, holes, and unconnected faces should be removed using a mesh editor (ex: MeshLab)

## Main Files

### 3D_Model_Pairwise_Segmentation
- main file for performing a segmentation on two input meshes
- first mesh is the segmented mesh, second in the base model
- outputs a series of files containing pickled lists of sets
    - each set is a collection of point ids representing a segment (point ids in the segmented model)
    - a list of sets is the list of all segments in the model
    - each pickled list is the state of the segmentation at a particular snapshot
        - ex: if configured to segment into 10 parts, outputs lists for 2 segs, 3 segs, 4 segs, ...
- currently, need to manually adjust the parameters for the segmentation
    - how many segments until stopping point
    - how often to run a segmentation refinment

### Construct_Part_Files
- receives as input a file containing a pickled segmentation list (output from 3D_Model_Pairwise_Segmentation)
- creates a series of .obj files, with each file containing a segment of the overall mesh

### Evaluate_Error
- receives as input multiple segmentation schema (pickled list files) and trusted "correct" schema (set of .obj files)
- evaluates the number of successfully classified points according to the "correct" schema for each candidate schema
- implements the Hungarian algorithm for assignment
    - to evaluate success rate, need to determine which segments "correspond" with each other
    - Hungarian algorithm finds the set of correspondences that maximizes overlap (best success rate)

### Graph_Rmses
- receives as input multiple segmentation schema (pickled list files) and the source and base case models
- evaluates the RMSE after segmentation and alignment for each schema, plots them in a bar chart

## Helper Modules

### mesh_3d_classes
- contains data structure classes for points, lines, faces, and overall mesh (combination of them)
- implements function for initializing model objects by reading from .obj files
- also contains function for computing concavity at each edge, and visualizing edges with a data map

### seam_gen_funcs
- implements a semi-random planar segmentation method for generating potential seams
    - randomly selects a point, defines a plane that contains the normal to that point

### segmenter_class
- class that keeps track of all potential seams and maintains segmentation schema
- implements a method for updating the error of each potential seam given the current schema (parallelized)
- implements a method for selecting the current best seam and adding it to the current schema (adding a segment)
- implements a method for refining the current schema by merging and recombining all seams
    - greedy successive selection of seams often yields inoptimal local mins, so merging previously selected seams and resegmenting can fix past mistakes

### split_align_funcs
- collection of functions for determining points on either side of a seam, and aligning two point clouds
- alignment function contains an optional variable to visualize the two point clouds after alignment

### util_funcs
- rmse_resize: for resizing a model in relation to another, based on what minimizes alignment error
- show_segs: for visualizing all segments in a segmentation
- eval_rmse: for computing the current rmse over all segments when aligning to the base model
- create_adj_dict: for constructing a dictionary of adjacencies between points in a model