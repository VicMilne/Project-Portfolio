# Split Alignment Functions

import numpy as np
import open3d as o3d

def dfs(adj_dict, visited_pts, init_queue):

    while(init_queue):
        id = init_queue.pop()
        visited_pts.add(id)
        for nid in adj_dict[id]:
            if(nid not in visited_pts):
                init_queue.append(nid)
    
    return visited_pts
    

# given pos and neg sets of boundary points (points with edges that span the segmentation boundary)
# create list 
def split(adj_dict, pos_ordered, neg_ordered, seg=None):
    # model: model object
    # pos_edge: set of positive boundary points (points with edges that span seg boundary)
    # neg_edge: set of negative boundary points (points with edges that span seg boundary)
    # seg:

    # initialize point sets with the boundary points
    pos_edge = set()
    neg_edge = set()
    for p in pos_ordered:
        pos_edge.add(p)
    for p in neg_ordered:
        neg_edge.add(p)

    # create initial set of non-boundary positive segment points
    queue = []
    for id in pos_edge:
        for new_id in adj_dict[id]:
            if(new_id not in pos_edge and new_id not in neg_edge):
                queue.append(new_id)

    # perform dfs to find all points connected to those in the queue
    # cur_edge points are already visited, dfs can't cross the segmentation boundary
    pos_set = dfs(adj_dict, pos_edge, queue)
    
    del_list = []
    # if only segmenting a certain subset of points
    if(seg is not None):
        for val in pos_set:
            # add to delete list if point is not in subset
            if(val not in seg):
                del_list.append(val)
    
    # remove all deletion points
    for val in del_list:
        pos_set.remove(val)

    if(seg is not None):
        neg_set = seg - pos_set
    else:
        full_set = set(range(0, len(adj_dict)))
        neg_set = full_set - pos_set

    return pos_set, neg_set

# given 2 sets of point clouds, determine the ICP alignment transformation and alignment error
def align_multi(m_array, points, target, downsample, show=False):
        # m: model to be aligned / transformed
        # points: list of point indices, corresponding to points in m
        # target: o3d.geometry.PointCloud object, target to be aligned to

        p_array = m_array[np.array(list(points))]

        if(downsample):
            down_list = np.random.choice(list(range(len(p_array))), max(int(len(p_array)/4), 1), replace=False)
            p_array = p_array[np.array(list(down_list))]


        # use list of locations to create a source point cloud
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(p_array)

        # initialize transformation array
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

        # use o3d method to perform icp
        # reg_p2p result contains updated transformation array, resulting alignment error
        threshold = 50
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
        
        # if ICP result returns 0 alignment error, assumed something went wrong, return very high value
        if(reg_p2p.inlier_rmse == 0):
            reg_p2p.inlier_rmse = np.inf
        
        # if show is provided, use o3d visualize to show the aligned point clouds
        if(show):
            # colour point clouds
            source.paint_uniform_color([1, 0.206, 0])
            target.paint_uniform_color([0, 0.651, 0.929])
            # apply computed transformation to source
            source.transform(reg_p2p.transformation)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # visualize
            o3d.visualization.draw_geometries([source, target, coord_frame],
                                            zoom=0.5,
                                            front=[0.9288, -0.2951, -0.2242],
                                            lookat=[0, 1, 1],
                                            up=[0, 0, 1])
        
        return reg_p2p