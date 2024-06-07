import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import copy

import warnings

warnings.filterwarnings("ignore")


def transitive_aln_pair(aln_pair1: np.ndarray, aln_pair2: np.ndarray):
    """
    Given two aligned pairs (i, j) and (j, k) and output (i, k)

    Args:
        aln_pair1:  ndarray of a residue alignment pair from pLM-BLAST
        aln_pair2:  ndarray of a residue alignment pair from pLM-BLAST
    """
    idx1 = np.setdiff1d(aln_pair1, aln_pair2)
    idx2 = np.setdiff1d(aln_pair2, aln_pair1)

    # make sure the smaller number is assigned to the x dim
    if idx1[0] < idx2[0]:
        aln_pair3 = np.array([idx2[0], idx1[0]], dtype=np.int32)
    else:
        aln_pair3 = np.array([idx1[0], idx2[0]], dtype=np.int32)

    return aln_pair3


def fill_in_transitivity(path1: np.ndarray, path2: np.ndarray):
    """
    Iterate the identified suboptimal alignment paths to explore transitive alignment pairs
    Iterating one path is enough?

    Args:
        path1: ndarray of one suboptimal alignment
        path2: ndarray of another suboptimal alignment
    """

    transitive_pair_list = []

    # iterate path1
    for res_pair in path1:
        res1_index = res_pair[0]
        res2_index = res_pair[1]
        # iterate path2
        for search_pair in path2:
            # matched pair detected
            if ((res1_index in search_pair) != (res2_index in search_pair)) and (res1_index != res2_index) and (
                    search_pair[0] != search_pair[1]):
                transitive_pair = transitive_aln_pair(res_pair, search_pair)
                transitive_pair_list.append(transitive_pair)
            else:
                pass

    # remove redundancy
    transitive_pair_list = list(set(tuple(arr) for arr in transitive_pair_list))
    transitive_pair_list = [np.array(arr) for arr in transitive_pair_list]

    return transitive_pair_list


def find_right_corner(pair_array: np.ndarray):

    """
    find a right corner aligned pair to start finding the trace

    Args:
        pair_array: ndarray of aligned pairs remained to be add to traces
    """

    sums = pair_array[:, 0] + pair_array[:, 1]
    max_index = np.argmax(sums)
    start_pair = pair_array[max_index]

    return start_pair


def find_neighbor(current_pair: np.ndarray, pair_list: list):

    """
    find a neighbor of a given aligned pair

    Args:
        current_pair: the aligned pair used to search its neighbors
        pair_list: list of aligned pairs remained to be add to traces
    """

    neighbor_list = []
    for pair in pair_list:
        # vertical or translational
        if [(current_pair[0] - pair[0]) + (current_pair[1] - pair[1])] == 1:
            neighbor_list.append(pair)
        # diagnoal
        if (current_pair[0] - pair[0]) == 1 and (current_pair[1] - pair[1]) == 1:
            neighbor_list.append(pair)

    return neighbor_list


def find_transitive_trace_from_filled_matrix(pair_list: list):

    """
    Given a transitive aligned residue pair array, identify the transitive traces it contains

    Args:
        pair_list: a transitive aligned residue pair list
    """

    transitive_alignment_trace = defaultdict(list)

    wait_pair_list = copy.deepcopy(pair_list)

    path_idx = 0
    while len(wait_pair_list) > 0:

        # find the right corner start pair and start a new trace
        wait_pair_array = np.vstack(wait_pair_list)
        trace_start = find_right_corner(wait_pair_array)
        wait_pair_list = [arr for arr in wait_pair_list if not np.array_equal(arr, trace_start)]
        neighbor_pair_list = find_neighbor(trace_start, wait_pair_list)
        transitive_alignment_trace[path_idx].append(trace_start)

        # iterate pair list to complete current trace
        while len(neighbor_pair_list) != 0:
            assert len(neighbor_pair_list) == 1
            current_pair = neighbor_pair_list[0]
            wait_pair_list = [arr for arr in wait_pair_list if not np.array_equal(arr, current_pair)]
            transitive_alignment_trace[path_idx].append(current_pair)
            neighbor_pair_list = find_neighbor(current_pair, wait_pair_list)

        path_idx += 1

    return transitive_alignment_trace


def score_transitive_trace(sub_matrix: np.ndarray, trace: np.ndarray):

    """
    Score the transitive traces based on substitution matrix - take the average of all aligned pairs

    Args:
        sub_matrix: substitution matrix of cosine similarity
        trace: an array of transitive trace
    """

    trace_score = sub_matrix[trace[:, 0], trace[:, 1]]
    trace_score = trace_score.mean()

    return trace_score


def score_repeat_trace(sub_matrix: np.ndarray, trace: np.ndarray):

    """
    Score the transitive traces based on substitution matrix - take the average of all aligned pairs

    Args:
        sub_matrix: substitution matrix of cosine similarity
        trace: an array of transitive trace
    """

    new_trace = copy.copy(trace)
    print(new_trace.shape)
    print(new_trace[:, 0])
    trace_score = sub_matrix[new_trace[:, 0], new_trace[:, 1]]
    trace_score = trace_score.mean()

    return trace_score


def transitive_result(results: np.ndarray, sub_matrix: np.ndarray, dist2diag_threshold: float = 5.0, transitive_trace_threshold: float = 0.0):

    """
    Find transitive traces for each pair of suboptimal alignment
    Args:
        sub_opt_result: result dataframe from plm-blast after filtering
        :param transitive_trace_threshold:
    """

    # only consider the lower triangle
    half_results = results[results['y1'] >= results['x1']]
    subopt_aln_dict = half_results.set_index('pathid').to_dict(orient='index')
    half_results = half_results[['indices', 'score', 'mode']]
    subopt_aln_pair = list(combinations(list(subopt_aln_dict), 2))

    for pair in subopt_aln_pair:
        aln1 = subopt_aln_dict[pair[0]]['indices']
        aln2 = subopt_aln_dict[pair[1]]['indices']
        #         aln1_score = subopt_aln_dict[pair[0]]['score']
        #         aln2_score = subopt_aln_dict[pair[0]]['score']
        #         transitive_trace_score = 0.25*min(aln1_score, aln2_score)
        transitive_pair_list = fill_in_transitivity(aln1, aln2)
        transitive_trace_dict = find_transitive_trace_from_filled_matrix(transitive_pair_list)

        # save transitive traces in the same dataframe
        for trace in transitive_trace_dict.values():
            trace = np.vstack(trace)
            trace = trace[trace[:, 0].argsort()]
            transitive_trace_score = score_transitive_trace(sub_matrix, trace)
            new_trace = [trace, transitive_trace_score, 'transitive']
            new_trace = pd.DataFrame([new_trace], columns=half_results.columns)
            half_results = pd.concat([half_results, new_trace], ignore_index=True)

    # one more filter based on the distance between detected traces and diagonal
    dist2diag = [np.abs(np.diff(trace, axis=1).mean()) for trace in half_results['indices']]
    half_results['dist_to_diag'] = dist2diag
    half_results = half_results[half_results['dist_to_diag'] >= dist2diag_threshold]
    # add length label
    trace_len = [trace.shape[0] for trace in half_results['indices']]
    half_results['len'] = trace_len
    half_results = half_results[half_results['score'] >= transitive_trace_threshold]
    half_results['pathid'] = range(len(half_results))

    return half_results
