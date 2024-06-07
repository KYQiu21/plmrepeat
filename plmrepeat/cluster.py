import sys
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from typing import List, Union
import warnings

warnings.filterwarnings("ignore")


# Functions to process plmblast-format result dataframe and remove trace redundancy

def clean_trace(data: pd.DataFrame):
    """
    cluster traces and output a filtered dataframe
    Args:
        data: pd.DataFrame of plmblast output format
    """

    dist_array = create_trace_dist(data)
    results = cluster_trace(data, dist_array, threshold=15.0, column=['len'])

    return results


def clean_trace2(data: pd.DataFrame):
    """
    cluster traces and output a filtered dataframe
    Args:
        data: pd.DataFrame of plmblast output format
    """

    dist_array = create_trace_dist2(data)
    results = cluster_trace_no_remove(data, dist_array, threshold=15.0, column=['len'])

    return results


def cluster_generator(results):
    """

    :param results:
    :return:
    """

    cluster_lists = dict()
    for index, row in results.iterrows():
        cluster = row['cluster']
        indices = row['indices']

        if cluster not in cluster_lists:
            cluster_lists[cluster] = []

        cluster_lists[cluster].append(indices)

    trace_cluster_list = [np.vstack(cluster) for cluster in cluster_lists.values()]
    trace_cluster_range_list = [(np.amin(cluster, axis=0)[0], np.amax(cluster, axis=0)[0]) for cluster in trace_cluster_list]
    trace_cluster_range_list = sorted(trace_cluster_range_list, key=lambda x: x[0])
    trace_cluster_range_list = [item for item in trace_cluster_range_list if abs(item[0] - item[1]) >= 10]

    return trace_cluster_range_list


def my_filter_result_dataframe(data: pd.DataFrame, column: Union[str, List[str]] = ['score'],
                               score_threshold: float = 0.0, len_threshold: int = 0, dist2diag_threshold: float = 10,
                               resi_idx_list=None, ignore_diagonal=True, ignore_diagonal_overlap=True) -> \
        pd.DataFrame:
    """
    keep spans with the biggest score and len
    Args:
        data: (pd.DataFrame)
        column: entries used to do filtering
        score_threshold: score threshold of considered suboptimal alignments
        dist2diag_threshold: a distance threshold involved in removing the redundant shifted traces close to the diagonal
        resi_idx_list: a list of residue index which have been assigned in a repeat so not considered in next search
        ignore_diagonal: ignore the trivial self-alignment with score 1.0
        ignore_diagonal_overlap: a more controllable option to filter redundant shifted traces close to the diagonal (consider
                                 both dist2diag and score)
                                 :param len_threshold:
    """

    # if not an empty dataframe
    if data.shape[0] != 0:
        data = data.sort_values(by=['len'], ascending=False)
        indices = data.indices.tolist()
        data['y1'] = [yx[0][0] for yx in indices]
        data['x1'] = [yx[0][1] for yx in indices]
        data['score'] = data['score'].round(3)

        if isinstance(column, str):
            column = [column]
        resultsflt = list()
        iterator = data.groupby(['y1', 'x1'])
        for col in column:
            for groupid, group in iterator:
                tmp = group.nlargest(1, [col], keep='first')
                resultsflt.append(tmp)
        resultsflt = pd.concat(resultsflt)
        # drop duplicates sometimes
        resultsflt = resultsflt.drop_duplicates(
            subset=['pathid', 'i', 'len', 'score'])
        # filter
        resultsflt = resultsflt.sort_values(by=['score'], ascending=False)
        resultsflt = resultsflt[resultsflt['score'] >= score_threshold]
        # one more filter based on the distance between detected traces and diagonal
        dist2diag = [np.abs(np.diff(trace, axis=1).mean()) for trace in resultsflt['indices']]
        resultsflt['dist_to_diag'] = dist2diag
        resultsflt = resultsflt[resultsflt['score'] >= score_threshold]
        resultsflt = resultsflt[resultsflt['dist_to_diag'] >= dist2diag_threshold]
        resultsflt = resultsflt[resultsflt['y1'] >= resultsflt['x1']]

        if ignore_diagonal:
            resultsflt = resultsflt[resultsflt['score'] != 1.0]

        if ignore_diagonal_overlap:
            # resultsflt = resultsflt[resultsflt['dist_to_diag'] >= resultsflt['len']]
            # a more conserved condition to trade-off FP and TP
            resultsflt = resultsflt[(resultsflt['dist_to_diag'] >= resultsflt['len']) | (
                    (resultsflt['dist_to_diag'] < resultsflt['len']) & (resultsflt['score'] >= 0.4))]

        nogap_len_list = [min(len(set(trace[:, 0].tolist())), len(set(trace[:, 1].tolist()))) for trace in
                          resultsflt['indices'].tolist()]
        resultsflt['nogap_len'] = nogap_len_list
        resultsflt = resultsflt[resultsflt['nogap_len'] >= len_threshold]

        # delete detected repeats that are located in the already assigned region
        if resi_idx_list is not None:
            resultsflt = resultsflt[
                resultsflt['indices'].apply(lambda arr: all(item[0] not in resi_idx_list for item in arr))]

        resultsflt['pathid'] = range(len(resultsflt))

    # just return the empty dataframe
    else:
        resultsflt = data

    return resultsflt


def manhattan_dist_point2trace(point: np.ndarray, trace: np.ndarray):
    """
    Compute the manhattan distance between one point and a set of points

    Args:
        point: an array of a start/end point of one trace
        trace: an array of a trace (aka a set of points)
    """

    point2trace_dist = np.sum(np.abs(point - trace), axis=1)

    return point2trace_dist


def manhattan_dist_point2point(point1: np.ndarray, point2: np.ndarray):
    """
    Compute the manhattan distance between one point and a set of points

    Args:
        point1 & point2: arrays of a start/end point of one trace
    """

    x1, y1 = point1
    x2, y2 = point2
    distance = abs(x1 - x2) + abs(y1 - y2)

    return distance


def create_trace_dist(results: pd.DataFrame):
    """
    Create a distance matrix of all traces

    Args:
        results: Dataframe of pLM-BLAST results
    """

    trace_id_list = results['pathid'].tolist()
    trace_list = results['indices'].tolist()
    trace_pair_list = list(combinations(list(trace_id_list), 2))
    taxi_dist_array = np.zeros((len(trace_id_list), len(trace_id_list)))

    for pair in trace_pair_list:
        trace1_id = pair[0]
        trace1 = trace_list[trace1_id]
        trace1_start = trace1[0]
        trace1_end = trace1[-1]
        trace2_id = pair[1]
        trace2 = trace_list[trace2_id]
        trace2_start = trace2[0]
        trace2_end = trace2[-1]

        trace1_start_to_trace2_dist = min(manhattan_dist_point2trace(trace1_start, trace2))
        trace1_end_to_trace2_dist = min(manhattan_dist_point2trace(trace1_end, trace2))
        trace1_to_trace2_dist = max(trace1_start_to_trace2_dist, trace1_end_to_trace2_dist)

        trace2_start_to_trace1_dist = min(manhattan_dist_point2trace(trace2_start, trace1))
        trace2_end_to_trace1_dist = min(manhattan_dist_point2trace(trace2_end, trace1))
        trace2_to_trace1_dist = max(trace2_start_to_trace1_dist, trace2_end_to_trace1_dist)

        trace1_trace2_dist = min(trace1_to_trace2_dist, trace2_to_trace1_dist)
        taxi_dist_array[trace1_id][trace2_id] = trace1_trace2_dist
        taxi_dist_array[trace2_id][trace1_id] = trace1_trace2_dist

    return taxi_dist_array


def create_trace_dist2(results: pd.DataFrame):
    """
    Create a distance matrix of all traces

    Args:
        results: Dataframe of pLM-BLAST results
    """

    trace_id_list = results['pathid'].tolist()
    trace_list = results['indices'].tolist()
    trace_pair_list = list(combinations(list(trace_id_list), 2))
    taxi_dist_array = np.zeros((len(trace_id_list), len(trace_id_list)))

    for pair in trace_pair_list:
        trace1_id = pair[0]
        trace1 = trace_list[trace1_id]
        trace1_start = trace1[0]
        trace1_end = trace1[-1]
        trace2_id = pair[1]
        trace2 = trace_list[trace2_id]
        trace2_start = trace2[0]
        trace2_end = trace2[-1]

        trace1_start_to_trace2_dist = min(manhattan_dist_point2trace(trace1_start, trace2))
        trace1_end_to_trace2_dist = min(manhattan_dist_point2trace(trace1_end, trace2))
        trace1_to_trace2_dist = min(trace1_start_to_trace2_dist, trace1_end_to_trace2_dist)

        trace2_start_to_trace1_dist = min(manhattan_dist_point2trace(trace2_start, trace1))
        trace2_end_to_trace1_dist = min(manhattan_dist_point2trace(trace2_end, trace1))
        trace2_to_trace1_dist = min(trace2_start_to_trace1_dist, trace2_end_to_trace1_dist)

        trace1_trace2_dist = min(trace1_to_trace2_dist, trace2_to_trace1_dist)
        taxi_dist_array[trace1_id][trace2_id] = trace1_trace2_dist
        taxi_dist_array[trace2_id][trace1_id] = trace1_trace2_dist

    return taxi_dist_array


def cluster_trace(results: pd.DataFrame, dist_array: np.ndarray, threshold: float = 15.0,
                  column: Union[str, List[str]] = ['len']):
    """
    Cluster traces according to distances for removing redundancy

    Args:
        results:  Dataframe of pLM-BLAST results
        dist_array: distance array of each trace pair
        threshold: distance threshold to be considered in the same cluster
        column: features used to select a representative from a cluster
    """

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                         distance_threshold=threshold)
    trace_label = clustering.fit_predict(dist_array)
    results['cluster'] = trace_label

    if isinstance(column, str):
        column = [column]
    resultsflt = list()
    iterator = results.groupby(['cluster'])
    for col in column:
        for groupid, group in iterator:
            tmp = group.nlargest(1, [col], keep='first')
            resultsflt.append(tmp)

    resultsflt = pd.concat(resultsflt)

    # reset the pathid column
    resultsflt['pathid'] = range(len(resultsflt))

    return resultsflt


def cluster_trace_no_remove(results: pd.DataFrame, dist_array: np.ndarray, threshold: float = 15.0,
                            column: Union[str, List[str]] = ['len']):
    """
    Cluster traces according to distances for removing redundancy

    Args:
        results:  Dataframe of pLM-BLAST results
        dist_array: distance array of each trace pair
        threshold: distance threshold to be considered in the same cluster
        column: features used to select a representative from a cluster
    """

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                         distance_threshold=threshold)
    trace_label = clustering.fit_predict(dist_array)
    results['cluster'] = trace_label

    # reset the pathid column
    results['pathid'] = range(len(results))

    return results
