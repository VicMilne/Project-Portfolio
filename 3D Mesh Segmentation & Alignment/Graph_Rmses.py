# Evaluate error per iter

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
import pickle
from mesh_3d_classes import Model
from split_align_funcs import align_multi

if(__name__ == "__main__"):

    seg_scheme_names = ["1 Seg", "2 Seg", "3 Seg", "4 Seg", "4 Seg Refined", "5 Seg", "6 Seg", "7 Seg", "8 Seg", "8 Seg Refined"]
    segmentations_list = ["splitnc_2", "splitnc_3", "splitnc_4", "refinenc_4", "splitnc_5", "splitnc_6", "splitnc_7", "splitnc_8", "refinenc_8"]
    f_name = ""
    f_name_2 = ""
    out_file = "image"

    xxx = perf_counter()
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
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    source.transform(reg_p2p.transformation)

    # update all points in m1 with new locations after transformation
    transformed_array = np.asarray(source.points)
    for i, point in enumerate(m1.points):
        point.l = list(transformed_array[i])

    # recompute bounds, stored array of point locations
    m1.compute_bounds_and_array()

    # initalize MSE list with the default option (no segmentation)
    list_mses = [pow(reg_p2p.inlier_rmse, 2)]

    # for all potential segmentation schema in the list
    for k in range(len(segmentations_list)):

        # open the file, following the default naming convention
        f = open("{f_name}_{f_name_2}_{word}".format(f_name=f_name, f_name_2=f_name_2, word=segmentations_list[k]), "rb")
        cur_segs = pickle.load(f)

        # check the alignment error to the target of each segment, save the result
        cur_rmses = []
        for i, seg in enumerate(cur_segs):
            reg_p2p = align_multi(m1.point_array, seg, target, False)
            weight = len(seg)
            cur_rmses.append([reg_p2p.inlier_rmse, weight])

        # determine weighted average of all mse values, weighted by number of points for each seg
        overall_rmse = 0
        overall_weight = 0
        for val in cur_rmses:
            overall_mse += pow(val[0], 2) * val[1]
            overall_weight += val[1]

        overall_mse /= overall_weight

        # append result to overall list
        list_mses.append(overall_mse)


    check = 0

    # construct graph and save
    plt.figure(figsize=(20, 10), dpi=80)
    plt.rcParams.update({'font.size': 14})
    plt.bar(seg_scheme_names, list_mses, width=0.4)
    plt.title("{f_name} to {f_name_2} Alignment Error After Each Stage of Algorithm".format(f_name=f_name, f_name_2=f_name_2))
    plt.xlabel("Stages")
    plt.ylabel("MSE of All Points After Alignment (mm)")
    plt.savefig("{fname}.png".format(fname=out_file))