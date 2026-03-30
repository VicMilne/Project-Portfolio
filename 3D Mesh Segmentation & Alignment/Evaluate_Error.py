# Evaluate_Error

# Reads in list of potential segmentations as well as a "true" segmentation, evaluates the accuracy (correctly classified points) of each segmentation

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
import pickle
import itertools
from mesh_3d_classes import Model

f_name = ""
parts = ["", ""]

header = ""
segmentations_list = ["splitnc_2", "splitnc_3", "splitnc_4", "refinenc_4", "splitnc_5", "splitnc_6", "splitnc_7", "splitnc_8", "refinenc_8"]

# file contains many functions for implementing the Hungarian algorithm for assigning the best relationships between pairs of segments


def attempt_solve(best_locs, req, current):
    """
    Recursive function for assigning segs of A to segs of B, given list optimal B segs for each A seg

    - Reads in a list of optimal assignment choices in B (assignments with zero "error") for each seg in A
    - Attempts to assign a unique B seg for each A seg, checks all possible combinations until targeted # of assignments is reached
    - Returns False if no valid full assignment is found

    Params
    ------
    best_locs: list of lists of ints
        sublists of optimal B segs for each A seg
    req: int
        Number of A-B assignments required to successfully return
    current: list
        current set of assignments, built recursively
    """

    # if size of current set equally the required number, successful assignment found, return True
    if(len(current) == req):
        return current, True
    # search through each potential match for the current segment
    for val in best_locs[0]:
        # if segment has already been assigned, can't reuse
        if(val in current):
            continue
        # continue the search by recursively calling attempt_solve on list[1:], with selected value added to current
        result, success = attempt_solve(best_locs[1:], req, current + [val])
        # if success is returned True, recursion found a valid assignment set, pass back up the call stack
        if(success):
            return result, success
    # if no valid successful sets found for this assignment of current, return False
    return [], False

def create_mapping(result, num_r, num_c):
    """
    Creates a mapping of values in result to the index of that value, removes redundant mappings

    - Reads in a list of numbers representing segments
    - each number maps to it's index in the list
    - Ex: [4,5,6], 4 -> 0, 5 -> 1, 6 -> 2
    - the number of associations should be the min(num_r, num_c)
        - only keep keys < num_rows and values < num_cols

    Params
    ------
    result: list of mapping keys, index of each key is its value
    """

    # create the initial mapping
    mapping = {}
    for i in range(len(result)):
        mapping[result[i]] = i
    del_list = []

    # if less rows than columns, keys higher than num_rows are present, need to remove them
    if(num_r < num_c):
        for key in mapping:
            if(key >= num_r):
                del_list.append(key)
    # if less columns than rows, values higher than num_cols are present, need to remove them
    elif(num_r > num_c):
        for key in mapping:
            if(mapping[key] >= num_c):
                del_list.append(key)

    # remove necessary keys/values from mapping
    for key in del_list:
        del mapping[key]
    
    return mapping


def init_covered(starred, all_zeros, matrix_size):
    """
    Sets all zeros to "covered" if they share a column with a starred zero, otherwise "uncovered"

    Params
    ------
    starred: set()
        set of starred zero locations (locations are ints, rows#*len_columns+column#)
    all_zeros: set()
        set of all zero locations (locations are ints, rows#*len_columns+column#)
    matrix_size: int
        number of rows and columns in the original matrix
    """
    covered = set()
    uncovered = []
    covered_cols = []

    # for each starred zero, check entire column for zeros
    for val in starred:
        # extract column number from location int
        col = val % matrix_size
        for i in range(matrix_size):
            # if location in column is a zero, cover this zero
            if(i*matrix_size+col in all_zeros):
                covered.add(i*matrix_size+col)
                all_zeros.remove(i*matrix_size+col)
                covered_cols.append(col)
    
    # add the remaining zeros to uncovered
    for val in all_zeros:
        uncovered.append(val)
    
    # remove duplicate values, get list of columns covered
    covered_cols = list(set(covered_cols))
    
    return covered, uncovered, covered_cols

def row_col_cover(matrix):
    """
    Find minimum number of rows + columns that, when selected, cover all zeros in the matrix

    Params
    ------
    matrix: mxn numpy array, represents the costs of pairing mi and ni for all mi in segmentation M and ni in segmentation N
    """
    matrix_size = len(matrix)
    # covering
    tot_zeros = 0

    # inital logic for assigning (*) values to certain 0s in the matrix
    # - when a 0 is starred, no other 0s in its row or column may be starred
    # - greedily star 0s until no valid 0s remain
    starred = set()
    all_zeros = set()
    rows = list(range(matrix_size))
    cols = list(range(matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            # finding zeros in the matrix
            if(matrix[i][j] == 0):
                # also maintain list of all zero locations
                tot_zeros += 1
                all_zeros.add(i*matrix_size+j)
                # if i and j are still valid rows/columns
                if(i in rows and j in cols):
                    starred.add(i*matrix_size+j)
                    # invalidate the row and column
                    cols.remove(j)
                    rows.remove(i)
    
    # initial "covering" function, covers all zeros that are or share a column with a starred 0
    covered, uncovered, covered_cols = init_covered(starred, all_zeros, matrix_size)
    covered_rows = []

    # prime are another required label for 0s performing the coverage algorithm
    primed = set()

    # while not all 0s in the matrix are covered
    while(len(uncovered) > 0):
        # get first uncovered 0 and assign it prime
        loc = uncovered[0]
        primed.add(loc)
        row = int(loc/matrix_size)
        col = loc % matrix_size

        # identify if a starred zero shares the row with this zero
        star_row = False
        for star_col in range(row*matrix_size, (row+1)*matrix_size):
            if(star_col in starred):
                star_row = True
                break
        
        # if shares row with * zero, switch coverage such that the starred zero covers the row, and no longer covers its column
        # now, the new prime zero is covered, but any zeros in the starred zero's column are no longer covered
        if(star_row):
            star_col = star_col % matrix_size
            covered_cols.remove(star_col)
            covered_rows.append(row)
            # update the covered zeros given the changed coverage schema
            for i in range(matrix_size):
                if(i*matrix_size+star_col in covered and i != row):
                    covered.remove(i*matrix_size+star_col)
                    uncovered.append(i*matrix_size+star_col)
            for i in range(matrix_size):
                if(row*matrix_size+i in uncovered):
                    covered.add(row*matrix_size+i)
                    uncovered.remove(row*matrix_size+i)
            
            # now that the primed zero is covered, go back to top of the loop
            continue
        
        # no starred zero found, convert current primed zero to starred and cascade this effect

        # maintain list of zeros to change
        visited = [loc]

        # find if a starred zero shares a column with the current primed zero
        row = -1
        for i in range(matrix_size):
            if(i*matrix_size+col in starred):
                row = i

        # only continue process if there's a starred zero in the current column
        while(row >= 0):
            # add this zero to the list
            visited.append(row*matrix_size+col)

            # find the primed zero that shares the row with this starred zero
            for i in range(row*matrix_size, (row+1)*matrix_size):
                if(i in primed):
                    col = i % matrix_size
                    break
            # add this zero to the list
            visited.append(row*matrix_size+col)

            # once again, find if a starred zero shares the column with this primed zero
            row = -1
            for i in range(matrix_size):
                if(i*matrix_size+col in starred):
                    row = i
        
        # visited is now a queue of zeros that traces between primed and starred zero (by column then row repeating)
        
        # all primed zeros become starred and any starred zeros are unlabeled
        for i in range(len(visited)):
            if(i % 2 == 0):
                starred.add(visited[i])
            else:
                starred.remove(visited[i])

        # empty the prime set and remove all coverage
        del primed
        primed = set()
        for val in covered:
            uncovered.append(val)
        del covered

        # redo the initial covering function, row with an adjusted set of starred zeros
        all_zeros = set(uncovered)
        covered, uncovered, covered_cols = init_covered(starred, all_zeros, matrix_size)
        covered_rows = []
    
    # all zeros are now covered, return the selected rows and columns that make up the coverage
    return covered_rows, covered_cols


def hungarian(matrix):
    """
    Optimization function for unique assignment of row-column pairs, to minimize the cost of assignment

    Params
    ------
    matrix: mxn numpy array, represents shared points between segment mi and segment nj for mi 
    """
    matrix = np.asarray(matrix)

    # algorithm is written handle matrices where m >= n
    # transpose if this is not true (optimal assignments are symmetrical, can invert back at the end of the function)

    num_c = len(matrix[0])
    num_r = len(matrix)
    if(num_r < num_c):
        matrix = np.r_[matrix, np.zeros((num_c-num_r, num_c))]
    elif(num_r > num_c):
        matrix = np.c_[matrix, np.zeros((num_r, num_r-num_c))]

    matrix_size = len(matrix)

    # want to maximize number of shared points, but algorithm attempts to find minimum cost of assignment
    # so, negate all values, maxima are now minima
    matrix *= -1

    ### Step 1: Subtract minimum value of each row from all values in the row, then attempt to solve

    # subtract the mins from each row
    for row in matrix:
        row -= row.min()
    
    # collect locations of zeros for each column
    zero_locs = []
    for i in range(matrix_size):
        zero_locs.append(list(np.where(matrix[:, i] == 0)[0]))
    
    # find potential zero cost assignment in the matrix
    result, success = attempt_solve(zero_locs, matrix_size, [])

    # if assignment found, format mapping dictionary, invert if transposed at the beginning of the function
    if(success):
        # create and return mapping, removing all references to any rows or columns of zeros added to make a square matrix
        return create_mapping(result, num_r, num_c)


    ### Step 2: Subtract minimum value of each column from all values in the column, then attempt to solve
    
    # subtract the mins from each column
    for i in range(matrix_size):
        matrix[:, i] -= matrix[:, i].min()
    
    # collect locations of zeros for each column
    zero_locs = []
    for i in range(matrix_size):
        zero_locs.append(list(np.where(matrix[:, i] == 0)[0]))
    
    # find potential zero cost assignment in the matrix
    result, success = attempt_solve(zero_locs, matrix_size, [])

    # if assignment found, format mapping dictionary, invert if transposed at the beginning of the function
    if(success):
        # create and return mapping, removing all references to any rows or columns of zeros added to make a square matrix
        return create_mapping(result, num_r, num_c)
    
    ### Step 3: Iteratively perform minimum row+column 0 covers, then apply transform to the matrix
    #         - continually apply transformations until a zero cost assignment is possible

    num_loops = 0
    # terminate if number of transformations exceeds 1000 (for a reasonably sized matrix, should never get close to 1000)
    while(num_loops < 1000):
        num_loops += 1
        # find the minimum set of rows and columns that contain all 0s in the matrix
        rows, cols = row_col_cover(matrix)
        # if the resulting number of rows+columns is equal to the width of the matrix, a zero cost assignment must be present
        if(len(rows) + len(cols) == len(matrix[0])):
            # collect locations of zeros for each column
            zero_locs = []
            for i in range(matrix_size):
                zero_locs.append(list(np.where(matrix[:, i] == 0)[0]))

            # find zero cost assignment in the matrix
            result, success = attempt_solve(zero_locs, matrix_size, [])
            
            # create and return mapping, removing all references to any rows or columns of zeros added to make a square matrix
            return create_mapping(result, num_r, num_c)

        # find the minimum value in the matrix that is uncovered by the rows+columns
        min_uncovered = np.inf
        for i in range(matrix_size):
            if(i in rows):
                continue
            for j in range(matrix_size):
                if(j in cols):
                    continue
                if(matrix[i][j] < min_uncovered):
                    min_uncovered = matrix[i][j]
        
        # add this value to every element covered by both a row and a column
        # subtract this value from every uncovered element
        for i in range(matrix_size):
            for j in range(matrix_size):
                if(i in rows and j in cols):
                    matrix[i][j] += min_uncovered
                elif(i not in rows and j not in cols):
                    matrix[i][j] -= min_uncovered
        
        # loop back and find row column cover of the new matrix

    # if no result is found within 1000 transformations, raise exception
    raise Exception("Could not find minimum assignment for matrix")



if(__name__ == "__main__"):
    if(".obj" in f_name):
        f_name = f_name[:-4]
    
    # initialize model object using the input obj file
    m1 = Model(f_name)

    segs = []

    added = set()

    # read in every segment (part) file
    for part in parts:
        segs.append(set())
        part_model = Model("{f_name} {part}".format(f_name=f_name, part=part))

        # add all point locations in part to cmp_set
        cmp_set = set()
        for point in part_model.points:
            cmp_set.add(str(round(point.loc[0], 3)) + str(round(point.loc[1], 3)) + str(round(point.loc[2], 3)))
        
        # check if each point in m1 corresponds to the currently select part
        for p in m1.points:
            string = str(round(p.loc[0], 3)) + str(round(p.loc[1], 3)) + str(round(p.loc[2], 3))
            # check if p's location hits in the cache
            if(string in cmp_set and p.id not in added):
                segs[-1].add(p.id)
                added.add(p.id)


    # iterate through the list of candidate segmentations, store success rate of each (# correct points / # total points)
    all_perfs = []
    for suffix in segmentations_list:
        f = open(header + suffix, "rb")
        my_segs = pickle.load(f)

        
        # for each candidate segment, find its overlap with each real segment, store in matrix
        overlaps = []
        for i in range(len(my_segs)):
            overlaps.append([])
            for j in range(len(segs)):
                overlaps[-1].append(len(my_segs[i].intersection(segs[j])))
        
        # use the hungarian algorithm to find the best mapping between my_segs and segs (most overlapping points)
        mapping = hungarian(np.copy(overlaps))

        best_overlap = sum([overlaps[i][mapping[i]] for i in mapping])

        perform = best_overlap / len(m1.point_array)
        all_perfs.append(perform)

    check = 0




### test cases for hungarian ###
# x = np.asarray([[0,3,0,4], [6,0,1,0], [0,3,7,2], [0,2,4,1]])
# x = np.asarray([[0,0,0,0,1,0], [2,8,9,5,0,4], [0,3,7,2,5,4], [0,5,11,13,7,8], [0,3,10,15,13,7], [0,12,8,16,7,9]])
# x = np.asarray([[5,3,2,4], [6,8,1,9], [2,3,7,2], [7,2,4,1]])
# x = np.asarray([[0,0,0,0,1], [2,8,9,5,0], [0,3,7,2,5], [0,5,11,13,7], [0,3,10,15,13], [0,12,8,16,7]])
# all were tested by first multiplying by negative 1 (to find the "min")

### test with ###
# diff = x.shape[0] - x.shape[1]
# if(diff < 0):
#     x1 = np.concatenate((x, np.zeros((abs(diff), x.shape[1]))), axis=0)
# elif(diff > 0):
#     x1 = np.concatenate((x, np.zeros((x.shape[0], diff))), axis=1)
# else:
#     x1 = x
# num_assign = len(x1)
# options = list(range(num_assign))
# best_result = np.inf
# for comb in itertools.permutations(options, num_assign):
#     result = 0
#     for i in range(num_assign):
#         result += x1[i][comb[i]]
#     if(result < best_result):
#         best_result = result
#         best_comb = comb