# 3D Model Playing
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
from mesh_3d_classes import Model
from util_funcs import rmse_resize, create_adj_dict
from seam_gen_funcs import planar_seam_generation
from segmenter_class import Model_Segmenter

if(__name__ == "__main__"):

    f_name = ""
    f_name_2 = ""


    if(".obj" in f_name):
        f_name = f_name[:-4]
    if(".obj" in f_name_2):
        f_name_2 = f_name_2[:-4]
    
    # instantiate m1 and m2 using the two object files
    m1 = Model(f_name)
    m2 = Model(f_name_2)
    
    
    # initialize o3d point clouds from the models' points
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(m1.point_array)

    # target = rmse_resize(source, np.ndarray.copy(m2.point_array))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(m2.point_array)

    # compute normal vectors for the target, allows for faster PointToPlane ICP mode
    target.estimate_normals()

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    threshold = 50

    # perform ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())


    source.transform(reg_p2p.transformation)

    # update all points in m1 with new locations after transformation
    transformed_array = list(np.asarray(source.points))
    for i, point in enumerate(m1.points):
        point.loc = np.array(transformed_array[i])

    # recompute bounds, stored array of point locations
    m1.compute_bounds_and_array() 

    # compute concavity measure for edges in m1
    m1.compute_intra_weights()

    adj_dict = create_adj_dict(m1)

    # visualize_edges(m1) # uncomment to visualize the edge weights of m1

    # generate a dictionary of seams
    # selecting number of points in model as targeted number of seams
    
    pot_seams = planar_seam_generation(m1, adj_dict, len(m1.points))

    # initialize segment list as one set containing every point id
    cur_segs = [set()]
    for point in m1.points:
        cur_segs[0].add(point.id)
    
    #visualize_seams(m1, seams)

    # initialize, update dictionary describing each candidate seam (rmse reduction, segment number affected)
    segmenter = Model_Segmenter(m1.point_array, np.asarray(target.points), adj_dict, pot_seams, cur_segs)

    segmenter.update_candidates_parallel()

    # how many segments until stopping point (i-1 segs)
    num_segs_stop = 11
    for i in range(2, num_segs_stop):
        # greedily select segmentation seams that reduce mse the most
        # split until there are i segments
        segmenter.perform_split(i)

        # save the resulting segs (formatted as a list of sets of point ids)
        f = open("{f1}_{f2}_split_{num}".format(f1=f_name, f2=f_name_2, num=i), 'ab')
        pickle.dump(segmenter.cur_segs, f)
        f.close()
        
        # every 4 segments, perform a refine
        if(i % 4 == 0):
            # merge and resegment each current seam, one by one, to refine results
            best = segmenter.refine_segs()
            # save the resulting segs (formatted as a list of sets of point ids)
            f = open("{f1}_{f2}_refine_{num}".format(f1=f_name, f2=f_name_2, num=i), 'ab')
            pickle.dump(segmenter.cur_segs, f)
            f.close()
