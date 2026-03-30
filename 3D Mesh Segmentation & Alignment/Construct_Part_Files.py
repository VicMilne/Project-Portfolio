# Construct New Object Files
# Provide the option of either O.G. file or the downsampled file

from mesh_3d_classes import Model
import pickle
import open3d as o3d
import numpy as np

# # construct new obj file
txt = "# File output by Segmentation Algorithm\n\n"
header = ""
footers = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
seg_schema = ""

segmented_fname = "X.obj"

if(".obj" in segmented_fname):
    segmented_fname = segmented_fname[:-4]

m1 = Model(segmented_fname)
f = open(seg_schema, 'rb')
segs = pickle.load(f)
f.close()

for i, seg in enumerate(segs):

    txt = "# File output by Segmentation Algorithm\n\n"

    p_dict = {}
    for j, point_id in enumerate(seg):
        point = m1.points[point_id]
        temp_str = "v {0:.6f} {1:.6f} {2:.6f}\n".format(point.loc[0], point.loc[1], point.loc[2])
        txt += temp_str
        p_dict[point_id] = j+1

    txt += "\n"

    for face in m1.faces:
        if(face.p1.id in p_dict and face.p2.id in p_dict and face.p3.id in p_dict):
            temp_str = "f {0} {1} {2}\n".format(p_dict[face.p1.id], p_dict[face.p2.id], p_dict[face.p3.id])
            txt += temp_str

    f = open("{header}_{footer}.obj".format(header=header, footer=footers[i]), 'w')

    f.write(txt)
    f.close()