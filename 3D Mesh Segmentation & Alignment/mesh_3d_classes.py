# Model Class

import numpy as np
import open3d as o3d

class ModelPoint:
    """
    Point Data Structure
    """
    def __init__(self, x, y, z, id):
        self.loc = np.array([x, y, z])
        self.edges = []
        self.id = id
    
    def add_edge(self, e):
        self.edges.append(e)


class ModelEdge:
    """
    Edge Data Structure
    """
    def __init__(self, p1, p2, id):
        self.p1 = p1
        self.p2 = p2
        self.f = []
        self.id = id
        self.size = np.linalg.norm(self.p1.loc - self.p2.loc)
        self.conc = 1
        self.weight = self.size
    
    def add_face(self, f1):
        # if(len(self.f) > 1):
        #     return
        self.f.append(f1)

class ModelFace:
    """
    Face Data Structure"""
    def __init__(self, p1, p2, p3, edges, id, normal=[0,0,0]):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.edges = edges
        self.id = id
        self.normal = np.array(normal)

class Model:
    """
    Represents a Model, Contains Points, Edges, and Faces"""
    def __init__(self, f_name):
        f_name = f_name + ".obj"
        self.points = None
        self.edges = None
        self.faces = None
        self.max_edge = None
        self.thres = None
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = None, None, None, None, None, None
        self.point_array = None
        # initialize above variables
        self.read_from_obj(f_name)
        self.compute_bounds_and_array()
    
    # read from .obj file, get points, edges, faces, and compute max edge length
    def read_from_obj(self, f_name):
        # Obj File Format:
        # 1. Headings
        # 2. Points, noted as "v x y z" where x y z are the 3D coordinates
        # 3. Faces, noted as "f a b c" where a b c are the corresponding points in the file, index starts at 1
        # 4. Footer

        # Note: Obj files are intentionally generated in simplest form, no normals, no no texture info

        # read obj file lines
        f = open(f_name)
        text = f.read()
        f.close()
        lines = text.split("\n")

        # iterate to first point in file, marked with "v"
        points = []
        i = 0
        while(lines[i] == "" or lines[i][0] != "v"):
            i += 1
        lines = lines[i:]

        # iterate, store all point locations, stop when "f" (faces) are reached
        for i, line in enumerate(lines):
            if(line == ""):
                continue
            if(line[0:2] == "v "):
                vals = line.split(" ")
                points.append(ModelPoint(float(vals[1]), float(vals[2]), float(vals[3]), len(points)))
            elif(line[0] == "f"):
                f_pos = i
                break

        lines = lines[f_pos:]

        edges = []
        edge_cache = {}
        faces = []
        num_points = len(points)
        max_edge = 0

        # iterate through faces, create edge and face data structures
        for i in range(len(lines)):
            if(lines[i] == ""):
                continue
            if(lines[i][0] == "f"):
                vals = lines[i].split(" ")
                # read indices for the 3 points in the face
                ind1 = int(vals[1]) - 1
                ind2 = int(vals[2]) - 1
                ind3 = int(vals[3]) - 1
                p1 = points[ind1]
                p2 = points[ind2]
                p3 = points[ind3]
                # compute seed corresponding to the two points in the edge (ind1*N+ind2, ind1 < ind2)
                seed12 = min(ind1, ind2)*num_points + max(ind1, ind2)
                seed23 = min(ind2, ind3)*num_points + max(ind2, ind3)
                seed13 = min(ind1, ind3)*num_points + max(ind1, ind3)
                # check edge against cache, create new edge if not present, connect edge to points, update max edge
                if(seed12 not in edge_cache):
                    edges.append(ModelEdge(p1, p2, len(edges)))
                    edge_cache[seed12] = edges[-1]
                    p1.add_edge(edges[-1])
                    p2.add_edge(edges[-1])
                    if(edges[-1].size > max_edge):
                        max_edge = edges[-1].size
                if(seed23 not in edge_cache):
                    edges.append(ModelEdge(p2, p3, len(edges)))
                    edge_cache[seed23] = edges[-1]
                    p2.add_edge(edges[-1])
                    p3.add_edge(edges[-1])
                    if(edges[-1].size > max_edge):
                        max_edge = edges[-1].size
                if(seed13 not in edge_cache):
                    edges.append(ModelEdge(p3, p1, len(edges)))
                    edge_cache[seed13] = edges[-1]
                    p3.add_edge(edges[-1])
                    p1.add_edge(edges[-1])
                    if(edges[-1].size > max_edge):
                        max_edge = edges[-1].size
                
                # compute normal for face (arbitrary sign, analyzed and possibly inverted with compute_intra_weights)
                norm = np.cross(p1.loc - p2.loc, p2.loc - p3.loc)
                norm = norm / np.linalg.norm(norm)

                faces.append(ModelFace(p1, p2, p3, [edge_cache[seed12], edge_cache[seed13], edge_cache[seed23]], len(faces), norm))
                edge_cache[seed12].add_face(faces[-1])
                edge_cache[seed13].add_face(faces[-1])
                edge_cache[seed23].add_face(faces[-1])
        
        self.points = points
        self.edges = edges
        self.faces = faces
        self.max_edge = max_edge
        
        return

    # compute bounds around the point cloud, and store all point locations as an array 
    def compute_bounds_and_array(self):
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf
        
        for point in self.points:
            if(point.loc[0] < self.xmin): self.xmin = point.loc[0]
            if(point.loc[0] > self.xmax): self.xmax = point.loc[0]
            if(point.loc[1] < self.ymin): self.ymin = point.loc[1]
            if(point.loc[1] > self.ymax): self.ymax = point.loc[1]
            if(point.loc[2] < self.zmin): self.zmin = point.loc[2]
            if(point.loc[2] > self.zmax): self.zmax = point.loc[2]
        
        self.point_array = []
        for point in self.points:
            self.point_array.append(point.loc)
        self.point_array = np.array(self.point_array)

        # length of largest bounding box dimension, used for halting longer seams in seam generation
        self.thres = sorted([self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin])[2]
        
        return
    
    # compute concavity measure for each edge in model
    def compute_intra_weights(self):

        # determine the point with the highest x value, select face connected to it with highest x component of normal vector
        cur_id = -1
        for p in self.points:
            if(p.loc[0] == self.xmax):
                max_x = -np.inf
                for edge in p.edges:
                    for face in edge.f:
                        cur_x = abs(face.normal[0])
                        if(cur_x > max_x):
                            max_x = cur_x
                            cur_id = face.id
                break
        
        # Note: the above code is important for determining the inward/outward direction of the mesh

        # the face containing the xmax point (point with highest x coordinate) with the highest magnitude 
        # x component of its normal vector must face in the positive x direction

        # thus, can check if previously computed normal vector is the correct direction, or if
        # it needs to be flipped

        if(cur_id == -1):
            raise Exception("Bounds do not match point locations")
        
        init_queue = []
        visited_edges = set()

        # if selected face's x component of its normal vector is negative, *-1
        x = np.array([1,0,0])
        face = self.faces[cur_id]

        if(np.dot(face.normal, x) < 0):
            face.normal = face.normal * -1

        # add the 3 edges on the face to the exploration queue
        for edge in face.edges:
            # determine the 3 points on the plane, p3 being the non-edge point
            p1 = edge.p1.loc
            p2 = edge.p2.loc
            if(face.p1.id not in [edge.p1.id, edge.p2.id]):
                p3 = face.p1.loc
            elif(face.p2.id not in [edge.p1.id, edge.p2.id]):
                p3 = face.p2.loc
            else:
                p3 = face.p3.loc
            
            # find unit vector perpendicular to edge and face normal, facing from within triangle towards edge
            inward = np.cross(face.normal, (p2-p1))
            if(np.dot(inward, (p2-p3)) < 0):
                inward *= -1
            inward /= np.linalg.norm(inward)
            # add edge to visited queue, and add relevent info to exploration queue
            #####init_queue.append([cur_id, edge.id, np.cross(inward, face.normal), face.normal])
            init_queue.append([cur_id, edge.id, inward, face.normal])
            visited_edges.add(edge.id)
        
        # BFS exploration, setting normal vectors for faces and exterior dihedral angles for edges
        while(len(init_queue) > 0):
            edge = self.edges[init_queue[0][1]]
            # remove edges that don't have two faces connecting to them (holes in mesh)
            if(len(edge.f) < 2):
                del init_queue[0]
                continue
            # get the face id of the other face connected to the edge
            for face in edge.f:
                if(face.id != init_queue[0][0]):
                    new_id = face.id
                    break

            # determine the 3 points on the plane, p3 being the non-edge point
            face = self.faces[new_id]
            p1 = edge.p1.loc
            p2 = edge.p2.loc
            if(face.p1.id not in [edge.p1.id, edge.p2.id]):
                p3 = face.p1.loc
            elif(face.p2.id not in [edge.p1.id, edge.p2.id]):
                p3 = face.p2.loc
            else:
                p3 = face.p3.loc

            # find unit vector perpendicular to edge and face normal, facing from within triangle towards edge
            inward = np.cross(face.normal, (p2-p1))
            if(np.dot(inward, (p2-p3)) < 0):
                inward *= -1
            inward /= np.linalg.norm(inward)
            
            # to determine direction of normal vector for this face, need to compare to normal vector of previous face
            
            # the cross product of the prev normal vector with prev inward vector should be the opposite direction
            # to the cross product of the current normal vector with the current inward vector

            # if cross product results are same direction, switch direction of current face normal
            cross1 = np.cross(face.normal, inward)
            cross2 = np.cross(self.faces[init_queue[0][0]].normal, init_queue[0][2])
            if(np.dot(cross1, cross2) > 0):
                face.normal *= -1

            # if the current inward vector and the previous face's normal vector are similar, 
            # the faces meet at a convex angle, set concavity to 0
            # need to check first because arccos will correspond to both a concave and convex angle
            if(np.dot(init_queue[0][3], inward) > 0):
                edge.conc = 1
            else:
                # compute angle between the two normal vectors of the faces
                cos_angle = np.arccos(np.dot(face.normal, init_queue[0][3]) / (np.linalg.norm(face.normal) * np.linalg.norm(init_queue[0][3])))
                # angle between faces = pi - angle between normals
                #edge.conc = 1 - (cos_angle / np.pi)
                edge.conc = 1
            
            # set edge weight using concavity and the length of the edge
            edge.weight = pow(edge.conc, 8) * edge.size
            if(not edge.weight > 0):
                edge.weight = 0

            # pop first value from queue
            del init_queue[0]

            # add the three edges on face to exploration queue if not already visited
            for edge in face.edges:
                if(edge.id not in visited_edges):
                    # 3 points, p3 is non-edge point
                    p1 = edge.p1.loc
                    p2 = edge.p2.loc
                    if(face.p1.id not in [edge.p1.id, edge.p2.id]):
                        p3 = face.p1.loc
                    elif(face.p2.id not in [edge.p1.id, edge.p2.id]):
                        p3 = face.p2.loc
                    else:
                        p3 = face.p3.loc
                    
                    # unit vector perpendicular to edge and face normal, facing towards edge
                    inward = np.cross(face.normal, (p2-p1))
                    if(np.dot(inward, (p2-p3)) < 0):
                        inward *= -1
                    inward /= np.linalg.norm(inward)
                    init_queue.append([new_id, edge.id, inward, face.normal])
                    visited_edges.add(edge.id)
        
        # once exploration queue is empty, all values are computed, return
        return
    
    def visualize_edges(self, weights=None):

        lines = []
        concs = []
        for edge in self.edges:
            lines.append([edge.p1.id, edge.p2.id])
            concs.append(edge.conc)
        
        if(weights is None):
            weights = concs
        
        sorted_weights = sorted(weights)
        mini = sorted_weights[int(len(sorted_weights)*0.05)]
        maxi = sorted_weights[int(len(sorted_weights)*0.95)]
        span = maxi - mini

        # assign colours to each point
        colours = []
        for i in range(len(weights)):
            # colour ranges from red (closest to max value) to blue (closest to min value)
            coef = (weights[i]-mini)/span
            coef = min(max(coef, 0), 1) # ensure between 0 and 1
            colours.append([1-coef, 0, coef])
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.point_array),
            lines=o3d.utility.Vector2iVector(lines)
        )

        line_set.colors = o3d.utility.Vector3dVector(colours)

        o3d.visualization.draw_geometries([line_set])

        return