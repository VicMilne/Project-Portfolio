# Seam Generation

from time import perf_counter
import numpy as np
import open3d as o3d
from split_align_funcs import split, align_multi

def planar_seam_generation(m, adj_dict, targ_seams):
    # m: model object
    # num_seams: int determining the number of planar seams to generate

    # use o3d PointCloud to estimate normals for each point

    rng = np.random.default_rng(seed=2)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(m.point_array)
    source.estimate_normals()
    m_norms = np.asarray(source.normals)

    seams = {}
    num_points = len(m.points)

    # calculate centroid of the point cloud
    avg_pt = m.point_array.sum(axis=0)/m.point_array.shape[0]

    # define a targeted number of planar seams, selecting a point in the model each time

    num_seams = 0

    while(num_seams < targ_seams):

        index = rng.integers(0, num_points)

        point = m.points[index]

        # if point is unconnected to rest of the model, skip
        if(len(point.edges) == 0):
            continue
        
        # generate gaussian random vector for orientation
        rand_vec = rng.normal(0, 1, 3)
        rand_vec /= np.linalg.norm(rand_vec)

        # define segmentation plane to include: random vector + normal to the surface of the mesh
        # cross product of both provides the normal vector to the segmentation plane
        plane_norm = np.cross(m_norms[index], rand_vec)
        plane_norm = plane_norm / np.linalg.norm(plane_norm)

        # determine the plane offset such that the plane intersects the current point
        d = np.dot(point.loc, plane_norm)
        # slightly offset plane, doesn't perfectly intersect
        d += point.edges[0].size / 100

        # for later applications, ensure the centroid of the point cloud is on the negative side of the plane
        if(np.dot(avg_pt, plane_norm) - d > 0):
            plane_norm *= -1
            d *= -1

        # use edge function to determine the points on either side of the plane
        pos_edge, neg_edge, length, conc = plane_edge_function(m, point, plane_norm, d)

        # if edge function fails, move on to next seam
        if(pos_edge is None):
            continue


        # determine the resulting segments from the seam
        pos_points, neg_points = split(adj_dict, pos_edge, neg_edge)

        # if resulting segmentation is invalid, move on to next seam
        if(pos_points is None or len(neg_points) == len(neg_edge) or len(neg_points) == 0):
            continue

        # initialize plane entry
        # concavity value raised to the 1/2, transformation reduces impact of the penalty
        # 0.25^0.5 = 0.5, 0.64^0.5 = 0.8, etc
        seams[num_seams] = [pos_edge, neg_edge, pow(conc, 1/2)]
        num_seams += 1

    
    return seams

### traces over the surface of the mesh in a loop, determining points on either side of the seam
def plane_edge_function(model, point, vec, d):
    # model: model object
    # point: point in model near segmentation plane
    # vec: normal vector of segmentation plane
    # d: offset of segmentation plane


    p_id = point.id
    p1 = point

    # find what side of the plane the point lies on
    val1 = np.dot(point.loc, vec) - d

    f_id = -1
    e_id = -1
    # search edges from the point, find an edge that crosses the segmentation plane
    for edge in point.edges:
        if(edge.p1.id == p_id):
            p2 = edge.p2
        else:
            p2 = edge.p1
        val2 = np.dot(p2.loc, vec) - d
        # if points are on opposite sides of the plane, crosses, set e_id and f_id
        if(val1 * val2 < 0):
            e_id = edge.id
            f_id = edge.f[0].id
            break

    # if no crossing edges found, something went wrong, return NULL
    if(f_id == -1):
        return None, None, None, None

    # face ids traversed
    face_walk = []
    # points on positive side of seam
    pos_edge = []
    # points on negative side of seam
    neg_edge = []

    face_walk.append(f_id)

    edge = model.edges[e_id]

    # compute intersect between plane and edge

    tot_d = abs(val1) + abs(val2)
    prev_point = (p1.loc * abs(val2) + p2.loc * abs(val1)) / tot_d

    # add point ids to positive and negative lists
    if(val1 < 0):
        pos_edge.append(p2.id)
        neg_edge.append(p1.id)
    else:
        pos_edge.append(p1.id)
        neg_edge.append(p2.id)

    # get next face in chain
    new_f = edge.f[0].id
    if(new_f == f_id):
        # abort if edge contains only one face (hole)
        if(len(edge.f) == 1):
            return None, None, None, None
        new_f = edge.f[1].id
    face_walk.append(new_f)

    # full computed values for length, concavity
    tot_len = 0
    tot_conc = 0
    tot_sim = 0

    # begin traversing along the mesh
    while(new_f != f_id):
        # abort if faces visited > total faces (means it's looping without returning to original face)
        if(len(face_walk) > len(model.faces)):
            return None, None, None, None
        
        face = model.faces[new_f]

        # search through edges for the other edge that intersects the plane
        for e in face.edges:
            if(e.id == e_id):
                continue
            p1 = e.p1
            p2 = e.p2
            val1 = np.dot(p1.loc, vec) - d
            val2 = np.dot(p2.loc, vec) - d
            # if one point is pos and other is neg, break
            if(val1*val2 <= 0):
                break
        
        # compute intersect between plane and edge
        tot_d = abs(val1) + abs(val2)
        cur_point = (p1.loc * abs(val2) + p2.loc * abs(val1)) / tot_d
        # compute length of segment, current point - prev point
        length = np.linalg.norm(cur_point - prev_point)
        prev_point = cur_point
        tot_len += length
        # optional threshold to remove long seams
        if(tot_len > model.thres):
            return None, None, None, None

        # concavity measure, compare orientation of edge with orientation of the segmentation plane
        # weighted sum of each edge's concavity
        # edges that are similar to vectors in the plane are weighted higher
        for ee in face.edges:
            e_vec = ee.p1.loc - ee.p2.loc
            e_vec /= np.linalg.norm(e_vec)
            sim = abs(np.dot(e_vec, vec))
            tot_conc += (1 - sim) * (1 - sim) * e.conc * length
            tot_sim += (1 - sim) * (1 - sim) * length


        # if last value added was positive, next will be negative
        if(pos_edge[-1] in [p1.id, p2.id]):
            # check which of the two points needs to be added
            if(val1 > 0):
                neg_edge.append(p2.id)
            else:
                neg_edge.append(p1.id)
        else:
            if(val1 > 0):
                pos_edge.append(p1.id)
            else:
                pos_edge.append(p2.id)

        # get next face
        new_f = e.f[0].id
        if(new_f == face_walk[-1]):
            # abort if edge contains only one face (hole)
            if(len(e.f) == 1):
                return None, None, None, None
            new_f = e.f[1].id

        face_walk.append(new_f)
        e_id = e.id
    
    # limit allowed concavity to prevent outliers
    conc = tot_conc / tot_sim
    if(conc < 0.85):
        conc = 0.85
    
    return pos_edge, neg_edge, tot_len, conc

# visualize all seams on the point cloud
# each point is classified by the best seam that passes through it
# red seams reduce RMSE the most, green seams the least
def visualize_seams(m, target, adj_dict, seams, segment=None):
    # m: model object
    # seams: list of seam data structures [pos_edge, neg_edge, penalty]
    # segment: optional dictionary of point ids, defines a segment of points

    point_scores_rmse = {}
    for seam_id in seams:
        pos_edge = seams[seam_id][0]
        neg_edge = seams[seam_id][1]
        penalty = seams[seam_id][2]
        if(segment is not None):
            invalid_seam = False
            for p_id in pos_edge:
                if(p_id not in segment):
                    invalid_seam = True
                    break
            if(invalid_seam):
                continue
            for p_id in neg_edge:
                if(p_id not in segment):
                    invalid_seam = True
                    break
            if(invalid_seam):
                continue

        pos_points, neg_points = split(adj_dict, pos_edge, neg_edge, segment)


        # calculate the RMSE after aligning the newly split segments
        mses = []
        tot_rmse = 0

        reg_p2p = align_multi(m.point_array, pos_points, target, False)
        weight = len(pos_points)
        mses.append(pow(reg_p2p.inlier_rmse, 2) * weight)


        reg_p2p = align_multi(m.point_array, neg_points, target, False)
        weight = len(neg_points)
        mses.append(pow(reg_p2p.inlier_rmse, 2) * weight)

        if(mses[0] > mses[1]):
            mses[1] *= penalty
        else:
            mses[0] *= penalty
        
        tot_rmse = sum(mses)

        if(segment is None):
            tot_rmse = pow(tot_rmse / len(m.points), 1/2)
        else:
            tot_rmse = pow(tot_rmse / len(segment), 1/2)


        # update all points in pos_edge and neg_edge
        for p_id in pos_edge:
            # if current seam is the best explored seam passing through the point, update point's value
            if(p_id not in point_scores_rmse or point_scores_rmse[p_id] > tot_rmse):
                point_scores_rmse[p_id] = tot_rmse
        for p_id in neg_edge:
            # if current seam is the best explored seam passing through the point, update point's value
            if(p_id not in point_scores_rmse or point_scores_rmse[p_id] > tot_rmse):
                point_scores_rmse[p_id] = tot_rmse
    
    # points have now been classified based on the RMSE of seams passing through them

    # remove outlier values where the ICP algorithm messes up (i.e. returns 0)
    del_list = []
    for key in point_scores_rmse: 
        if(point_scores_rmse[key] == 0): del_list.append(key)

    for val in del_list: del point_scores_rmse[val]

    # determine the range of values present in the RMSE values
    sort_vals = sorted(list(point_scores_rmse.values()))
    maxi = sort_vals[int(len(sort_vals)*0.95)]
    mini = sort_vals[int(len(sort_vals)*0.05)]
    span = maxi - mini

    # initialize PointCloud object for visualization
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(m.point_array)

    # assign colours to each point
    colours = []
    for i in range(len(source.points)):
        # if point was never initialized (never intersected by valid seam) assign grey colour
        if(i not in point_scores_rmse): 
            colours.append([0.5,0.5,0.5])
        else: 
            # colour ranges from red (closest to max value) to green (closest to min value)
            coef = (point_scores_rmse[i]-mini)/span
            coef = min(max(coef, 0), 1)
            colours.append([1-coef, coef, 0])

    # assign the colours to the source model
    source.colors = o3d.utility.Vector3dVector(colours)

    # visualize
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([source, coord_frame],
                                     zoom=0.5,
                                     front=[0.9288, -0.2951, -0.2242],
                                     lookat=[0, 1, 1],
                                     up=[0, 0, 1])
    
    return