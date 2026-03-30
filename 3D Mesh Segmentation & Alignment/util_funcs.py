# utils

import numpy as np
import open3d as o3d
from split_align_funcs import align_multi

# resizes the adjusted point cloud to the baseline point cloud, such that rmse is minimized 
def rmse_resize(baseline, adjusted_pts):
    # baseline: o3d.geometry.PointCloud object, size target
    # adjusted: o3d.geometry.PointCloud object, to be resized to align with baseline

    # initalize transformation array
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    threshold = 50

    # determine current centre of mass
    avg_pt = adjusted_pts.sum(axis=0)/adjusted_pts.shape[0]

    # move adjusted_pts to origin
    adjusted_pts -= avg_pt

    # initialize search parameters (search around 100%, search from 90% to 110%)
    value = 1
    search_rad = 0.1

    adjusted = o3d.geometry.PointCloud()

    for i in range(2):
        # create list of resize factors to check (21 equally space values over the define search area)
        candidates = np.linspace(value-search_rad, value+search_rad, 21)

        # check the resulting rmse after alignment for each of the candidates
        rmse_results = []
        for mult in candidates:
            # create new set of points using the resize factor
            new_pts = adjusted_pts * mult
            # move back to the original location (easier to align)
            new_pts += avg_pt
            # set adjusted's points to the new point set
            adjusted.points = o3d.utility.Vector3dVector(new_pts)

            # align baseline to adjusted (should go in this direction)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                baseline, adjusted, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            rmse_results.append(reg_p2p.inlier_rmse)
        
        # select the candidate that returns the best rmse_result, update search location
        value = candidates[np.argmin(rmse_results)]
        # reduce search radius by a factor of 10
        search_rad /= 10
    
    # update the adjusted pts using the scaling factor determined above
    adjusted_pts *= value
    adjusted_pts += avg_pt

    # update adjusted with these new points, then return
    adjusted.points = o3d.utility.Vector3dVector(new_pts)
    return adjusted

def show_segs(source, cur_segs):
    # list of 12 potential colours + grey
    colour_list = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [0.5,0,0],
                    [0,0.5,0], [0,0,0.5], [1,0.5,0.5], [0.5,1,0.5], [0.5,0.5,1], [0.5,0.5,0.5]]
    colours = []

    for i in range(len(source.points)):
        col_set = False
        # check if point is in each segment
        for j in range(len(cur_segs)):
            if(i in cur_segs[j]): 
                # select one of the 12 available colours from colour list
                colours.append(colour_list[j%12])
                col_set = True

        # if point is somehow not present in any segment, default to grey
        if(not col_set):
            colours.append(colour_list[-1])
        
    # visualize segments by colour
    source.colors = o3d.utility.Vector3dVector(colours)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([source, coord_frame],
                                 zoom=0.5,
                                 front=[0.9288, -0.2951, -0.2242],
                                 lookat=[0, 1, 1],
                                 up=[0, 0, 1])
    return

# compute the rmse over all points after individually aligning each segment to the target
def eval_rmse(m1, target, cur_segs):
    # m1: model object
    # target: o3d.geometry.PointCloud object, target to be aligned to
    # cur_segs: list of segments, sets of point indices

    overall_rmse = 0
    for seg in cur_segs:
        # use align_multi to obtain RMSE
        reg_p2p = align_multi(m1.point_array, seg, target, False)
        # to find overall rmse, must square rmse for that part (mse) and multiply by # of points in segment (se)
        overall_rmse += pow(reg_p2p.inlier_rmse, 2) * len(seg)
    
    # determine mean over all points, then compute square root
    overall_rmse /= len(m1.points)
    overall_rmse = pow(overall_rmse, 1/2)

    return overall_rmse

def create_adj_dict(m):
    adj_dict = {}
    for point in m.points:
        adj_dict[point.id] = []
        for edge in point.edges:
            if(edge.p1.id == point.id):
                adj_dict[point.id].append(edge.p2.id)
            else:
                adj_dict[point.id].append(edge.p1.id)
    
    return adj_dict