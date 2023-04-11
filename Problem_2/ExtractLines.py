#!/usr/bin/env python3

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, 
                                                             rho, 
                                                             (startIdx, endIdx), 
                                                             params['MIN_POINTS_PER_SEGMENT'],
                                                             params['LINE_POINT_DIST_THRESHOLD'])
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, indices, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        indices: tuple with (idxStart, idxEnd) representing the starting
                and ending indices of the segment to be split
        MIN_POINTS_PER_SEGMENT: parameter describing min pts per segement.
        LINE_POINT_DIST_THRESHOLD: parameter describing the max distance a point
            can be from a line before the line is split
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########

    # Define a recursive function
    # The arguments alpha_list, r_list and idx_list are empty lists that get populated with
    # the values of alpha, r and segment indices
    def RecursiveSplit(theta, rho, indices, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD, alpha_list, r_list, idx_list):
        alpha_fit, r_fit = FitLine(theta[indices[0]:indices[1]], rho[indices[0]:indices[1]])
        if (indices[1] - indices[0]) <= MIN_POINTS_PER_SEGMENT:
            alpha_list.append(alpha_fit)
            r_list.append(r_fit)
            idx_list.append([indices[0], indices[1]])
        else:
            s = FindSplit(theta[indices[0]:indices[1]], rho[indices[0]:indices[1]], alpha_fit, r_fit, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD)
            if s == -1:
                alpha_list.append(alpha_fit)
                r_list.append(r_fit)
                idx_list.append([indices[0], indices[1]])
            else:
                RecursiveSplit(theta, rho, (indices[0], indices[0] + s), MIN_POINTS_PER_SEGMENT,
                               LINE_POINT_DIST_THRESHOLD, alpha_list, r_list, idx_list)
                RecursiveSplit(theta, rho, (indices[0] + s, indices[1]), MIN_POINTS_PER_SEGMENT,
                               LINE_POINT_DIST_THRESHOLD, alpha_list, r_list, idx_list)

    # Initializing lists
    alpha = []
    r = []
    idx = []
    # Calling the recursive function
    RecursiveSplit(theta, rho, indices, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD, alpha, r, idx)
    # Converting the lists to numpy arrays
    alpha = np.array(alpha)
    r = np.array(r)
    idx = np.array(idx)

    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, MIN_POINTS_PER_SEGMENT, LINE_POINT_DIST_THRESHOLD):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. You should check if any point exists that satisfies both criteria.
    Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        MIN_POINTS_PER_SEGMENT: parameter describing min pts per segement.
        LINE_POINT_DIST_THRESHOLD: parameter describing the max distance a point
            can be from a line before the line is split
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).

    Hint: the output is the index of the first element in the second line. 
    For instance, if theta = np.array([0,1,2,3,4,5,6]) and splitIdx = 3, 
    then the two line segments will be 
        theta1 = theta[0:splitIdx] = np.array([0,1,2]) and 
        theta2 = theta[splitIdx:] = np.array([3,4,5,6]).
    '''
    ########## Code starts here ##########

    perp_distances = np.abs(np.multiply(rho, np.cos(theta - alpha)) - r)
    viable_points_idx = []  # list which will be populated with indices of viable points
    for i in range(len(perp_distances)):
        len_segment1 = i
        len_segment2 = len(rho) - i
        if min(len_segment1, len_segment2) >= MIN_POINTS_PER_SEGMENT and perp_distances[i] >= LINE_POINT_DIST_THRESHOLD:
            viable_points_idx.append(i)
    if len(viable_points_idx) > 0:
        # Line can be split
        splitIdx = viable_points_idx[np.argmax(perp_distances[viable_points_idx])]
    else:
        # Line cannot be split
        splitIdx = -1

    ########## Code ends here ##########
    return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads). Should be between -pi and pi.
        r: 'r' of best fit for range data (1 number) (m). Should be positive.
    '''
    ########## Code starts here ##########

    n = len(theta)  # number of beams/measurements
    numer = np.dot(np.square(rho), np.sin(2 * theta)) - 2/n * (
            np.dot(rho, np.cos(theta)) * np.dot(rho, np.sin(theta)))
    denom = np.dot(np.square(rho), np.cos(2 * theta)) - 1/n * (
            np.dot(rho, np.cos(theta)) ** 2 - np.dot(rho, np.sin(theta)) ** 2)
    alpha = 0.5 * np.arctan2(numer, denom) + np.pi/2
    r = 1/n * np.dot(rho, np.cos(theta - alpha))

    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########

    N_lines = len(alpha)  # number of lines

    # I create copies of alpha, r and pointIdx as they will be altered as the function is executed.
    # Instead, their copies will now be altered.
    alpha_c = np.copy(alpha)
    r_c = np.copy(r)
    pointIdx_c = np.copy(pointIdx)

    success = False
    while not success:
        success = True
        # Initializing
        alphaOut = np.array([])
        rOut = np.array([])
        pointIdxOut = np.zeros((0, 2), dtype=int)
        for i in range(N_lines - 1):
            idx1 = pointIdx_c[i, 0]
            idx2 = pointIdx_c[i + 1, 1]
            alpha_fit, r_fit = FitLine(theta[idx1:idx2], rho[idx1:idx2])
            s = FindSplit(theta[idx1:idx2], rho[idx1:idx2], alpha_fit, r_fit,
                          params['MIN_POINTS_PER_SEGMENT'], params['LINE_POINT_DIST_THRESHOLD'])
            if s == -1:
                # Accept the merge
                success = False
                alphaOut = np.hstack((alphaOut, alpha_fit, alpha_c[i + 2:]))
                rOut = np.hstack((rOut, r_fit, r_c[i + 2:]))
                pointIdxOut = np.vstack((pointIdxOut, [idx1, idx2], pointIdx_c[i + 2:, :]))
                N_lines = len(alphaOut)
                alpha_c = np.copy(alphaOut)
                r_c = np.copy(rOut)
                pointIdx_c = np.copy(pointIdxOut)
                break
            else:
                # Do not merge
                alphaOut = np.hstack((alphaOut, alpha_c[i]))
                rOut = np.hstack((rOut, r_c[i]))
                pointIdxOut = np.vstack((pointIdxOut, pointIdx_c[i, :]))

    # Adding the parameters for the last line
    alphaOut = np.hstack((alphaOut, alpha_c[N_lines - 1]))
    rOut = np.hstack((rOut, r_c[N_lines - 1]))
    pointIdxOut = np.vstack((pointIdxOut, pointIdx_c[N_lines - 1, :]))

    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2  # minimum number of points per line segment
    MAX_P2P_DIST = 0.35  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    # filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show()

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
