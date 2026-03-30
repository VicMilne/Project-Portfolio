# Segmenter Class

import numpy as np
import open3d as o3d
import multiprocessing
from itertools import repeat
from split_align_funcs import split, align_multi

class Model_Segmenter:
    def __init__(self, source_pts, target_pts, adj_dict, pot_seams, cur_segs):
        self.source_pts = source_pts
        self.target_pts = target_pts
        self.adj_dict = adj_dict
        self.pot_seams = pot_seams
        self.cur_segs = cur_segs
        self.candidates = {}
        self.best = -1
        self.adj_segs = []
    
    def update_candidates_parallel(self):

        num_threads = multiprocessing.cpu_count() - 2

        seam_list = []
        for seam_num in self.pot_seams:
            if(seam_num not in self.candidates):
                seam_list.append(seam_num)
        
        seam_chunks = np.array_split(np.array(seam_list), num_threads)

        with multiprocessing.Pool(processes=num_threads) as pool:
            dict_list = pool.map(self.update_candidates, seam_chunks)
        for d in dict_list:
            self.candidates.update(d)
        
        top_thres = sorted(self.candidates.values())[int(len(self.candidates)*0.95)][0]
        seam_list = []
        for seam in self.candidates:
            if(self.candidates[seam][0] > top_thres):
                seam_list.append(seam)
        
        seam_chunks = np.array_split(np.array(seam_list), num_threads)

        with multiprocessing.Pool(processes=num_threads) as pool:
            dict_list = pool.starmap(self.update_candidates, zip(seam_chunks, repeat(False)))
        
        finalists = {}
        for d in dict_list:
            finalists.update(d)
        self.best = max(finalists, key=finalists.get)


    def update_candidates(self, seam_list, downsample=True):
        # model: model object
        # target: o3d.geometry.PointCloud object, target to be aligned to
        # candidates: dict with potential seam number keys and [mse_reduction, seg#] value pairs
        # pot_seams: dict of potential segmentation boundaries [pos_edge, neg_edge, penalty]
        # cur_segs: existing segments in the model

        target = o3d.geometry.PointCloud()

        if(downsample):
            down_list = np.random.choice(list(range(len(self.target_pts))), int(len(self.target_pts)/4), replace=False)
            target_points = self.target_pts[np.array(list(down_list))]
        else:
            target_points = self.target_pts

        target.points = o3d.utility.Vector3dVector(target_points)
        target.estimate_normals()

        new_candidates = {}

        init_seg_mses = []
        for seg in self.cur_segs:
            # check mse of segment before spliting
            reg_p2p = align_multi(self.source_pts, seg, target, downsample)
            weight = len(seg)
            init_seg_mses.append(pow(reg_p2p.inlier_rmse, 2) * weight)
        

        for seam_num in seam_list:

            # determine segment of the first point on the positive side of the seam
            split_seg = -1
            for i, seg in enumerate(self.cur_segs):
                if(self.pot_seams[seam_num][0][0] in seg):
                    split_seg = i
            

            # ensure that all points on either side of the seam are within the same segment
            # i.e. seam doesn't cross any other seams
            valid = True
            # positive side
            for val in self.pot_seams[seam_num][0]:
                if(val not in self.cur_segs[split_seg]):
                    valid = False
                    break
            if(not valid):
                continue
            # negative side
            for val in self.pot_seams[seam_num][1]:
                if(val not in self.cur_segs[split_seg]):
                    valid = False
                    break
            if(not valid):
                continue

            # initialize candidate (corresponding to the seam number)
            new_candidates[seam_num] = [0, split_seg]


            # add initial mse (reg_p2p returns rmse per point, square to get mse, then need to multiply by # of points)
            new_candidates[seam_num][0] += init_seg_mses[split_seg]

            pos_points, neg_points = split(self.adj_dict, self.pot_seams[seam_num][0], self.pot_seams[seam_num][1], self.cur_segs[split_seg])

            mses = []
            # check alignment error of positive point cloud
            reg_p2p = align_multi(self.source_pts, pos_points, target, downsample)
            weight = len(pos_points)

            mses.append(pow(reg_p2p.inlier_rmse, 2) * weight)


            # check alignment error of negative point cloud
            reg_p2p = align_multi(self.source_pts, neg_points, target, downsample)
            weight = len(neg_points)

            mses.append(pow(reg_p2p.inlier_rmse, 2) * weight)


            # adjust the less meaningful segment's MSE value (almost always the smaller segment) by the penalty factor
            if(mses[0] > mses[1]):
                mses[1] *= self.pot_seams[seam_num][2]
            else:
                mses[0] *= self.pot_seams[seam_num][2]

            # subtract the MSE after splitting from the before result
            # higher result means a higher reduction in mse
            new_candidates[seam_num][0] -= sum(mses)

        # return after evaluating all missing candidate seams
        return new_candidates

    # returns list of segment adjacencies
    def get_segment_list(self):
        # cur_segs: list of segments (sets of point numbers)
        # adj_segs: list of point pairs, where each point 
        seg_list = []
        for border in self.adj_segs:
            seg1 = -1
            seg2 = -1
            # determine what segment the first adjacent point belongs to
            for i, seg in enumerate(self.cur_segs):
                if(border[0] in seg):
                    seg1 = i
                    break
            # determine what segment the second adjacent point belongs to
            for i, seg in enumerate(self.cur_segs):
                if(border[1] in seg):
                    seg2 = i
                    break
            # append segment pair to list, convention: smaller segment # first
            if(seg1 > seg2):
                seg_list.append([seg2, seg1])
            else:
                seg_list.append([seg1, seg2])
        
        return seg_list
    
    def perform_split(self, num_segs, update=True):
        # model: model object
        # target: o3d.geometry.PointCloud object, target to be aligned to
        # candidates: dict, keys=potential seam numbers, values = [mse_reduction, seg_num]
        # cur_segs: list if sets of points for each segment
        # pot_seams: dict of possible segmentations (one associated with each point)
        # adj_segs: list of pairs of segments, each pair indicates the segments border each other
        # num_segs: target number of segs to aim for while spliting

        while(len(self.cur_segs) < num_segs):

            pos_edge = self.pot_seams[self.best][0]
            neg_edge = self.pot_seams[self.best][1]

            pos_points, neg_points = split(self.adj_dict, pos_edge, neg_edge, self.cur_segs[self.candidates[self.best][1]])
            
            # update current segs list, set split segment to just the pos points, add new seg with neg points
            self.cur_segs[self.candidates[self.best][1]] = pos_points
            self.cur_segs.append(neg_points)

            # add new pair of points to the adjacency list (one on pos side of seam, one on neg, connected)
            p1_id = pos_edge[0]
            # check all points connected to p1
            for p2_id in self.adj_dict[p1_id]:
                if(p2_id in neg_edge):
                    self.adj_segs.append([p1_id, p2_id])
                    break

            # the split changes the validity and mse reduction of all seams within the old segment
            # need to remove and recompute all candidates from the two "new" segments
            recompute_segs = [self.candidates[self.best][1], len(self.cur_segs)-1]

            # create list of all candidates to be removed
            del_list = []
            for seam_num in self.candidates:
                if(self.candidates[seam_num][1] in recompute_segs):
                    del_list.append(seam_num)
            
            # remove the candidates
            for k in reversed(del_list):
                del self.candidates[k]
            
            # update the now missing candidates
            # can override if update will be called outside the function (as in refine_segs)
            if(update):
                self.update_candidates_parallel()
            
        return
    
    def refine_segs(self):
        num_segs = len(self.cur_segs)
        for j in range(num_segs-1):
            
            # initialize segment list, pairs of bordering segments
            seg_list = self.get_segment_list()

            # pop first segment boundary in list (adj_segs[0] corresponds to seg_list[0])
            chosen_seg = seg_list[0]
            del self.adj_segs[0]

            # merge the two segments, store in the first segment, delete the second segment
            self.cur_segs[chosen_seg[0]].update(self.cur_segs[chosen_seg[1]])
            del self.cur_segs[chosen_seg[1]]

            ### update candidates
            del_list = []
            for seam_num in self.candidates:
                # if candidate corresponds to either of the merged segments, must be recomputed, add to delete list
                if(self.candidates[seam_num][1] in chosen_seg):
                    del_list.append(seam_num)
                # if seg_num > the deleted segment, reduce by 1 (corresponding seg is 1 earlier in seg_list)
                if(self.candidates[seam_num][1] > chosen_seg[1]):
                    self.candidates[seam_num][1] -= 1
            
            # delete candidates in delete list
            for seam_num in del_list:
                del self.candidates[seam_num]
            
            # update the deleted candidates with new values
            self.update_candidates_parallel()

            
            # only update the candidates within perform_split if on the final iteration
            # otherwise, the update above could unnecessarily do double work
            if(j == num_segs-2):
                update = True
            else:
                update = False

            # perform one split to return to the desired number of segments
            # replaces earlier merged seam with best available seam given the current set of segments
            self.perform_split(num_segs, update)
    
        return