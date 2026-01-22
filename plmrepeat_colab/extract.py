from collections import defaultdict
from utils import *
import warnings

warnings.filterwarnings("ignore")


def fill_diagonal(score_matrix: np.ndarray):

    """

    :param score_matrix:
    :return:
    """

    for idx in range(score_matrix.shape[0]):
        score_matrix[idx][idx] = 1.0

    return score_matrix


def sum_trace_score_for_pair(trace_df: pd.DataFrame):
    """
    Each aligned residue pair will be scored based on all traces and return a dictionary of aligned
    residue pairs and their scores

    Args:
        trace_df: a dataframe containing all traces with 'indices', 'score', 'mode' format
    """

    aln_pair_score_dict = defaultdict(int)
    for idx, trace_row in trace_df.iterrows():
        trace = trace_row['indices']
        score = trace_row['score']
        for aln_pair in trace:
            aln_pair = (aln_pair[0], aln_pair[1])
            aln_pair_score_dict[aln_pair] += score

    return aln_pair_score_dict


def find_length_to_diagonal(pair_score_dict: defaultdict):
    """
    Sum all scores lying at the same distance to the matrix diagonal and
    the distance with the highest sum becomes the putative tandem repeat length L

    Args:
        pair_score_dict: a dictionary of aligned residue pairs and their scores
    """

    align_pair_list = list(pair_score_dict.keys())
    pair_score_list = list(pair_score_dict.values())
    dist2diag_list = [pair[0] - pair[1] for pair in align_pair_list]
    dist2diag_cul_dist_dict = defaultdict(int)
    for i in range(len(dist2diag_list)):
        dist2diag_cul_dist_dict[dist2diag_list[i]] += pair_score_list[i]

    max_dist_length = max(dist2diag_cul_dist_dict, key=dist2diag_cul_dist_dict.get)
    sort_dist2diag_cul_dist_dict = defaultdict(int, sorted(dist2diag_cul_dist_dict.items(), key=lambda item: item[1],
                                                           reverse=True))

    return max_dist_length, sort_dist2diag_cul_dist_dict


def matrix2dict(score_matrix: np.ndarray):
    """
    Convert a score matrix to a dictionary
    """
    score_dict = defaultdict(int)

    matrix_size = score_matrix.shape[0]
    for i in range(matrix_size):
        for j in range(i + 1):
            if score_matrix[i][j] != 0:
                score_dict[(i, j)] = score_matrix[i][j]

    return score_dict


def dict2matirx(score_dict: dict, seq_len: int):
    """
    Convert a score matrix to a dictionary
    """
    score_matrix = np.zeros((seq_len, seq_len))

    for pair, score in score_dict.items():
        score_matrix[pair[0]][pair[1]] = score
        score_matrix[pair[1]][pair[0]] = score

    return score_matrix


# def sliding_window_trust(score_matrix: np.ndarray, win_len: int, seq_len: int):
#     """
#     Use the sliding window of length L identified in the distance to diagonal to scan the sequence
#     self-comparison matrix; each aligned residue pair is scored using previously built dictionary
#
#     Args:
#         pair_score_dict: a dictionary of aligned residue pairs and their scores
#         win_len: putative length of the repeat length
#         seq_len: length of the SSA matrix
#     """
#
#     # sliding the SSA matrix
#     for pos in range(seq_len - win_len):
#         profile_score = 0
#         win_start = pos
#         win_end = pos + win_len
#         for column in range(win_start, win_end):
#             column_score = score_matrix[:, column]
#             non_zero_row = np.where(column_score != 0)[0]
#             for row in non_zero_row:
#                 # current position is row, column,
#                 # the considered environment is [row, column-L/2] to [row, column+L/2]
#                 row_value = score_matrix[row][column]
#                 env_min = max(0, column - half_win_len)
#                 env_max = min(seq_len - 1, column + half_win_len)
#                 env = score_matrix[row, np.r_[env_min:column, column + 1:env_max]]
#                 env_max_pair_value = max(env)
#                 reduce_row_value = row_value - env_max_pair_value
#                 profile_score += reduce_row_value
#
#         profile_score_dict[pos].append(profile_score)
#
#     return profile_score_dict


def sliding_window_hhrepid(score_matrix: np.ndarray, win_len: int, seq_len: int):
    """
    Use the sliding window of length L identified in the distance to diagonal to scan the sequence
    self-comparison matrix; each aligned residue pair is scored using previously built dictionary;
    scores of column are weighted by their position - central, higher; border, lower

    Args:
        pair_score_dict: a dictionary of aligned residue pairs and their scores
        win_len: putative length of the repeat length
        seq_len: length of the SSA matrix
    """

    profile_score_dict = defaultdict(list)

    # sliding the SSA matrix
    for pos in range(seq_len - win_len):

        profile_score = 0
        win_start = pos
        win_end = pos + win_len

        for column in range(win_start, win_end):

            column_score = score_matrix[:, column].sum()

            if (column - win_start) <= win_len / 2:
                column_weight = (column - win_start) - 1 / 2
            else:
                column_weight = win_len - (column - win_start) + 1 / 2

            column_score = column_weight * column_score
            profile_score += column_score

        profile_score_dict[pos].append(profile_score)

    return profile_score_dict


def rep_repeat_creator(start: int, repeat_len: int, score_matrix: np.ndarray, emb: np.ndarray):
    """
    Create a concat embedding of the representative repeat region;
    each column j is a residue - a (1024, ) vector - this vector is generated by the sum/average of
    L (length of the repeat) residue-wise embeddings, weighted by the score_matrix and folllowed by
    normalization

    Args:
        start: the start index of the representative repeat
        repeat_len: the length of the representative repeat
        score_matrix: the score array of each aligned pair with a shape LxL
        emb: embedding of the query sequence
    """

    repeat_start = start
    repeat_end = start + repeat_len
    emb_column_list = []

    # iterate each column of the representative region
    for i in range(repeat_start, repeat_end):

        score_weight = score_matrix[:, i]
        score_weight_sum = np.sum(score_weight)

        if score_weight_sum != 0:
            emb_column = np.dot(score_weight, emb) / score_weight_sum
            # print("score_weight", score_weight.shape, "emb", emb.shape, "emb_column", emb_column.shape)
        else:
            emb_column = emb[i]

        emb_column_list.append(emb_column)

    emb_repeat = np.vstack(emb_column_list)

    return emb_repeat


def create_rep_hit_idx_dict(aln):
    """
    function used to create an index mapping between representative repeat and hit region for ONE repeat

    Args:
        aln: an alignment of index array from a plmblast format dataframe
    """

    former_aln_pair_hit_idx = None
    rep_idx = aln[:, 1]
    single_rep_site_dict = {site_idx: [] for site_idx in list(set(rep_idx))}
    for aln_pair in aln:
        aln_pair_rep_idx, aln_pair_hit_idx = aln_pair[1], aln_pair[0]
        # this is used to record the gap position in hits
        if former_aln_pair_hit_idx is not None:
            if aln_pair_hit_idx == former_aln_pair_hit_idx:
                single_rep_site_dict[aln_pair_rep_idx].append('-')
            else:
                single_rep_site_dict[aln_pair_rep_idx].append(aln_pair_hit_idx)
                former_aln_pair_hit_idx = aln_pair_hit_idx
        else:
            single_rep_site_dict[aln_pair_rep_idx].append(aln_pair_hit_idx)
            former_aln_pair_hit_idx = aln_pair_hit_idx

    return single_rep_site_dict


def create_repeat_aln(results, fl_seq, rep_range, refine_seq_dict):
    """
    function to receive the result dataframe of repeat extraction and output a repeat alignment

    Args:
        results: dataframe in plmblast output format
        fl_seq: string of full-length sequence
        rep_range: the representative repeat range
        refine_seq_dict:
    """

    rep_start, rep_end = rep_range[0], rep_range[1]
    repeat_seq = fl_seq[rep_start:rep_end]

    aln_idx_list = results['indices'].tolist()
    repeat_num = len(aln_idx_list)
    rep_idx_range = list(set([idx for aln in aln_idx_list for idx in aln[:, -1].tolist()]))
    rep_start_idx, rep_end_idx = min(rep_idx_range), max(rep_idx_range)

    # rep_site_msa_dict: a dictionary where keys are site indices of the representative repeat and each value of the key
    # is a list of lists - each sublist is the corresponding site indices of the hit repeat region

    rep_site_msa_dict = defaultdict(dict)
    for site_idx in range(rep_start_idx, rep_end_idx + 1):
        for repeat_name in range(1, repeat_num + 1):
            rep_site_msa_dict[site_idx][str(repeat_name)] = []

    # iterate detected repeats
    for idx, aln in enumerate(aln_idx_list):
        # first create a mapping between the representative and hit index
        single_rep_site_dict = create_rep_hit_idx_dict(aln)

        # iterate representative site index
        for site_idx in list(single_rep_site_dict.keys()):
            single_hit_site_list = single_rep_site_dict[site_idx]
            rep_site_msa_dict[site_idx][str(idx + 1)] = single_hit_site_list

    # iterate sites to output detected repeats based on the rep_site_msa_dict
    aligned_repeat_seq_dict = {repeat_name: '' for repeat_name in ['rep'] + [str(i + 1) for i in range(repeat_num)]}
    # print("rep_site_msa_dict", rep_site_msa_dict)
    for site_idx, all_hit_idx in rep_site_msa_dict.items():
        # print("site_idx", site_idx, "all_hit_idx", all_hit_idx)
        max_hit_idx_len = max([len(hit_idx) for hit_name, hit_idx in all_hit_idx.items()])
        rep_gap_num = max_hit_idx_len - 1

        # first write the representative sequence
        aligned_repeat_seq_dict['rep'] += repeat_seq[site_idx] + '-' * rep_gap_num

        for hit_name, hit_idx in all_hit_idx.items():
            # print("hit_name", hit_name, "hit_idx", hit_idx)
            refine_seq = refine_seq_dict[hit_name]
            # print("len of refine_seq", len(refine_seq))
            site_num = len(hit_idx)
            hit_gap_num = max_hit_idx_len - site_num

            for single_hit_idx in hit_idx:
                if single_hit_idx == '-':
                    aligned_repeat_seq_dict[hit_name] += '-'
                else:
                    aligned_repeat_seq_dict[hit_name] += refine_seq[single_hit_idx]

            # add gaps
            aligned_repeat_seq_dict[hit_name] += '-' * hit_gap_num

    return aligned_repeat_seq_dict


def mask_score_matrix(score_matrix: np.ndarray, mask_idx: list) -> np.ndarray:
    """
    Mask a score matrix based on already detected repeats

    Args:
        score matrix: the score matrix from both local and transitive traces
        mask_idx: idx list of residue which have participated in other repeats
        :rtype: object
    """

    for res_idx in mask_idx:
        score_matrix[res_idx, :] = 0
        score_matrix[:, res_idx] = 0

    return score_matrix


def check_overlap(range_list: list, overlap_threshold: 0.2):
    """
    Check to what extent adjacent detected repeats overlap with each other by counting the difference between
    the residue index of *the start of the next trace* and *the end of trace before*

    Args:
        range_list: list of tuples, each tuple contains the start and end of a repeat instace
        :param overlap_threshold:
    """

    idx_difference_list = [range_list[i + 1][0] - range_list[i][1] for i in range(len(range_list) - 1)]
    idx_difference_list = [False if diff > -2 else True for diff in idx_difference_list]

    if len(range_list) == 1:
        return True
    else:
        overlap_ratio = idx_difference_list.count(True) / (len(range_list)-1)

        if overlap_ratio <= overlap_threshold:
            return False
        else:
            return True


def mask_idx_list(trace_cluster_range_list):

    mask_resi_idx_list = []
    for resi_range in trace_cluster_range_list:
        for resi_idx in range(resi_range[0], resi_range[1]+1):
            mask_resi_idx_list.append(resi_idx)

    mask_resi_idx_list = list(set(mask_resi_idx_list))

    return mask_resi_idx_list


def check_adjacent_range(c_range, b_range, a_range):
    c_range_start, c_range_end = c_range[0], c_range[1]

    # check the repeat before
    if b_range is not None:
        b_range_start, b_range_end = b_range[0], b_range[1]

        if b_range_end <= c_range_start:
            new_b_range = (b_range_start, b_range_end)
        else:
            if b_range_start >= c_range_start:
                new_b_range = None
            else:
                new_b_range = (b_range_start, c_range_start)
    else:
        new_b_range = None

    # check the repeat after
    if a_range is not None:
        a_range_start, a_range_end = a_range[0], a_range[1]

        if c_range_end <= a_range_start:
            new_a_range = (a_range_start, a_range_end)
        else:
            if c_range_start >= a_range_start:
                new_a_range = None
            else:
                new_a_range = (c_range_end, a_range_end)
    else:
        new_a_range = None

    return new_b_range, new_a_range


def remove_overlapping_ranges(range_list, score_list):
    idx_list = [i for i in range(len(range_list))]
    idx_range_dict = dict(zip(idx_list, range_list))
    idx_mark_dict = {idx: False for idx in idx_list}
    print(idx_mark_dict)
    sorted_data = sorted(zip(score_list, idx_list), reverse=True)
    print(sorted_data)

    # check repeat range in the order of score
    for i in range(len(sorted_data)):
        _, repeat_idx = sorted_data[i]
        print("repeat_idx", repeat_idx)
        repeat_range = idx_range_dict[repeat_idx]
        before_pos, after_pos = repeat_idx - 1, repeat_idx + 1

        if before_pos < 0:
            before_range = None
        else:
            before_range = idx_range_dict[before_pos]

        if after_pos > len(range_list) - 1:
            after_range = None
        else:
            after_range = idx_range_dict[after_pos]

        print("before_range", before_range)
        print("after_range", after_range)

        if repeat_range is not None:
            new_before_range, new_after_range = check_adjacent_range(repeat_range, before_range, after_range)

            print("new_before_range", new_before_range)
            print("new_after_range", new_after_range)

            # update repeat ranges dict
            if before_pos >= 0 and not idx_mark_dict[before_pos]:
                idx_range_dict[before_pos] = new_before_range
            if after_pos <= len(range_list) - 1 and not idx_mark_dict[after_pos]:
                idx_range_dict[after_pos] = new_after_range

            # mark this range to avoid being changed
            idx_mark_dict[repeat_idx] = True

    return idx_range_dict






