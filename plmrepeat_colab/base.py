import numpy as np
from pandas import DataFrame

from alignment import *
from cluster import *
from transitivity import *
from extract import *
from utils import *
import pickle
import os


class RepeatDetector:
    """
    main class for conducting a repeat search

    for each searching iteration, this class performs repeat detection procedure as below:
    1. plmblast self-sequence alignment
    2. add transitive traces to trace matrix
    3. estimate the repeat length and position of sliding window - this generates a list of possible
    lengths to be tested
    4. compute the weighted embedding for the representative repeat, plmblast the full-length
    embedding for scan repeat instances and refine boundary of each detected repeat
    5. generate alignment of repeats, score matrix with sliding window, relevant statistics, mask score
    6. perform another round of this workflow until no significant repeats are found

    Parameters to set:
    win_len: length of sliding windows (for plmblast)
    min_span: minimum span length (for plmblast)
    sigma_factor: sigma factor (for plmblast)
    ssa_score_threshold: score threshold of self-sequence alignments in pLM-BLAST
    transitive_score_threshold: score threshold of transitive alignments
    scan_score_threshold: score threshold of repeat scanning (compare embeddings of the representative repeat and the full-length sequence) in pLM-BLAST
    repeat_length_threshold: average length threshold of final refined repeat
    repeat_avg_length_threshold: average length (except for the longest one) threshold of final refined repeat
    repeat_score_threshold: average score threshold of final refined repeat
    overlap_threshold: overlap ratio of detected repeats


    Problems to be solved:
    1. how many checkpoints are needed to proceed in this workflow?
    2. where to decide there are no repeats to be detected
    3. ...

    Checkpoints needed to be added to this class:
    1. result dataframe with only one line cannot be clustered

    Improvements of pLM-Repeat workflow by cases:
    1. improve score threshold of scanned repeats to search more than one round to AVOID false negatives caused by redundant shifted alignments (case: 1560, 729)
    2. beta-solenoid redundant shifted alignments (case: 2IU8_A)

    """
    length_overlap_dict: dict

    def __init__(self,
                 emb,
                 seq,
                 win_len=15,
                 min_span=15,
                 ssa_sigma_factor=2.0,
                 ssa_score_threshold=0.25,
                 transitive_score_threshold=0.0,
                 scan_score_threshold=0.25,
                 scan_sigma_factor=0.2,
                 scan_with_weighted=True,
                 repeat_length_threshold=10,
                 repeat_score_threshold=0.0,
                 repeat_avg_length_threshold=10,
                 overlap_threshold=0.3,
                 use_transitivity=True,
                 mask_matrix=None,
                 weight_repeat_emb=True,
                 globalmode=True,
                 draw_during_procedure=False,
                 out_dir=None
                 ):

        self.emb = emb
        self.seq = seq
        self.win_len = win_len
        self.min_span = min_span
        self.ssa_sigma_factor = ssa_sigma_factor
        self.ssa_score_threshold = ssa_score_threshold
        self.transitive_score_threshold = transitive_score_threshold
        self.scan_score_threshold = scan_score_threshold
        self.scan_sigma_factor = scan_sigma_factor
        self.scan_with_weighted = scan_with_weighted
        self.repeat_length_threshold = repeat_length_threshold
        self.repeat_score_threshold = repeat_score_threshold
        self.repeat_avg_length_threshold = repeat_avg_length_threshold
        self.overlap_threshold = overlap_threshold
        self.mask_matrix = mask_matrix
        self.weight_repeat_emb = weight_repeat_emb
        self.update_score_matrix = False
        self.use_transitivity = use_transitivity
        self.globalmode = globalmode
        self.resi_idx_list = []  # the initial masked residue list is empty for the first detection operation

        # mark
        self.norepeat = False
        self.overlap = False
        self.score_matrix = np.zeros((len(seq), len(seq)))
        self.length_repeat_info_dict = None
        self.estimate_repeat_length_list = None
        self.aln_pair_score_dict = None
        self.extract_results = None
        self.sort_dist2diag_dist_dict = None
        self.rep_repeat_range = None
        self.draw_during_procedure = draw_during_procedure
        self.out_dir = out_dir
        self.high_ssa_score = 0.4
        self.high_ssa_hit = False

        # get plmblast (with transitivity) results and list of estimated repeat length
        self._get_ssa_alignment(self.emb, self.seq)
        if not self.norepeat:
            if self.use_transitivity:
                print("Use transitivity!")
                self._get_full_alignment(self.emb, self.seq)
            else:
                self.ssa_with_transitive_results = self.ssa_result

        # print(self.ssa_result)
        if self.ssa_result is not None:
            print(self.ssa_result['score'])
            if max(self.ssa_result['score'].tolist()) >= self.high_ssa_score:
                self.high_ssa_hit = True

    def _get_ssa_alignment(self, emb, seq):
        """
        execute plmblast self-sequence alignment to obtain suboptimal alignments

        :param emb: embedding of the full-length sequence
        :param seq: string of the full-length sequence
        """

        results, sub_matrix = self_alignment(emb, seq, window=self.win_len, min_span=self.min_span,
                                             sigma_factor=self.ssa_sigma_factor)
        results = my_filter_result_dataframe(results, column=['len'], score_threshold=self.ssa_score_threshold,
                                             ignore_diagonal_overlap=True)
        self.sub_matrix = sub_matrix

        # CHECKPOINT: if there is only one alignment then no need to proceed forward
        if results.shape[0] <= 1:
            self.norepeat = True
            self.ssa_result = None
        else:
            results = clean_trace(results)
            # judge again after clustering
            if results.shape[0] <= 1:
                self.norepeat = True
                self.ssa_result = None
            else:
                self.ssa_result = results

    def _get_full_alignment(self, emb, seq):
        """
        execute transitivity and length search

        :param emb: embedding of the full-length sequence
        :param seq: string of the full-length sequence
        """

        # Step 2: add transitive traces
        transitive_results = transitive_result(self.ssa_result, self.sub_matrix, transitive_trace_threshold=self.transitive_score_threshold)
        transitive_results = clean_trace(transitive_results)
        assert transitive_results.shape[0] >= self.ssa_result.shape[0]
        self.ssa_with_transitive_results = transitive_results

    def get_len_list(self, repeat_class_idx):

        # Step 3: estimate repeat length (this should give a list of estimated length)
        # directly calculate the score matrix from plmblast results in the first search of repeats
        if not self.update_score_matrix:
            aln_pair_score_dict = sum_trace_score_for_pair(self.ssa_with_transitive_results)
            score_matrix = dict2matirx(aln_pair_score_dict, len(self.seq))
            self.score_matrix = score_matrix
            self.aln_pair_score_dict = aln_pair_score_dict

        # in the next iterations of repeat search - new score matrix is generated by masking the assigned residue index in the original score matrix
        else:
            self.score_matrix = mask_score_matrix(self.score_matrix, self.resi_idx_list)
            self.aln_pair_score_dict = matrix2dict(self.score_matrix)

        # CHECKPOINT: no need to proceed ahead if aln_pair_score_dict is empty
        if len(self.aln_pair_score_dict) == 0:
            self.norepeat = True
        else:
            max_dist_length, sort_dist2diag_dist_dict = find_length_to_diagonal(self.aln_pair_score_dict)
            max_aln_length = max(self.ssa_result['len'])

            # consider length with scores higher than max_score/2
            max_score = sort_dist2diag_dist_dict[max_dist_length]
            estimate_repeat_length_list = [length for length, score in sort_dist2diag_dist_dict.items() if score >= 0/4*max_score and len(self.seq) > length >= self.min_span] + [max_aln_length]
            estimate_repeat_length_list = [length for length in estimate_repeat_length_list if len(self.seq)/2 >= length]
            self.estimate_repeat_length_list = estimate_repeat_length_list

            # dictionary to record overlap
            length_overlap_dict = {length: False for length in estimate_repeat_length_list}
            self.length_overlap_dict = length_overlap_dict

            # dictionary to record detected repeat info
            self.length_repeat_info_dict = defaultdict(dict)

            # draw and save score matrix for checking
            if self.draw_during_procedure:
                save_sub_matrix_path = f'{self.out_dir}/sub_matrix{repeat_class_idx}.jpg'
                draw_sub_matrix(self.sub_matrix, save=True, save_path=save_sub_matrix_path)

                save_score_matrix_path = f'{self.out_dir}/score_matrix{repeat_class_idx}.jpg'
                draw_sub_matrix(self.score_matrix, save=True, sub=False, save_path=save_score_matrix_path)

                # draw_trace(results=self.ssa_result, seq=self.seq)
                # draw_trace(results=self.ssa_with_transitive_results, seq=self.seq)

            print("Alignment matrix built!")

    def search_one_len(self, estimate_repeat_length):

        """
        function to execute repeat detection after obtaining suboptimal alignments with one estimated length

        :param estimate_repeat_length:
        :return:
        """

        # Step 4: determine the position of sliding window
        profile_score_dict = sliding_window_hhrepid(self.score_matrix, estimate_repeat_length, len(self.seq))
        # self.score_matrix = fill_diagonal(self.score_matrix)

        # print("score_matrix", self.score_matrix, self.score_matrix.shape, np.all(self.score_matrix == 0.0), np.sum(self.score_matrix), "profile_score_dict", profile_score_dict)
        max_win_start = max(profile_score_dict, key=profile_score_dict.get)
        max_win_end = max_win_start + estimate_repeat_length
        self.rep_repeat_range = (max_win_start, max_win_end)
        print("window position: ", max_win_start, max_win_end)

        if self.draw_during_procedure:
            draw_repeat_matrix(self.score_matrix, (max_win_start, max_win_end))

        # Step 5: generate the representative repeat embedding
        weight_repeat_emb = rep_repeat_creator(max_win_start, estimate_repeat_length, self.score_matrix, self.emb).astype(np.float32)
        origin_repeat_emb = self.emb[max_win_start:max_win_end].astype(np.float32)

        # Step 6: scan for repeat instances
        if self.scan_with_weighted:
            extract_results, repeat_diagonal_row, score_matrix = scan_repeat(emb=self.emb,
                                                                             seq=self.seq,
                                                                             repeat=[max_win_start, max_win_end],
                                                                             repeat_emb=weight_repeat_emb,
                                                                             sigma_factor=self.scan_sigma_factor)
        else:
            extract_results, repeat_diagonal_row, score_matrix = scan_repeat(emb=self.emb,
                                                                             seq=self.seq,
                                                                             repeat=[max_win_start, max_win_end],
                                                                             repeat_emb=origin_repeat_emb,
                                                                             sigma_factor=self.scan_sigma_factor)

        # CHECKPOINT: are there any repeats detected in scan_repeat???
        # param resi_idx_list is used to filter those traces containing sites that have been included in one kind of repeat
        # param len_threshold is used to avoid situation where the representative repeat is much longer than the detected ones
        extract_results = my_filter_result_dataframe(extract_results, column=['score'], score_threshold=self.scan_score_threshold, len_threshold=estimate_repeat_length*1/4, ignore_diagonal=False, resi_idx_list=self.resi_idx_list)
        self.extract_results = extract_results
        # print("scan repeat results: ", extract_results)

        # draw repeat extraction for check
        if extract_results.shape[0] >= 1 and self.draw_during_procedure:
            draw_extract_repeat(extract_results, len(self.seq), estimate_repeat_length)

        return max_win_start, max_win_end, weight_repeat_emb, origin_repeat_emb

    def define_boundary(self, estimate_repeat_length, max_win_start, max_win_end, weight_repeat_emb, origin_repeat_emb):
        """

        :param estimate_repeat_length:
        :param max_win_start:
        :param max_win_end:
        :param weight_repeat_emb:
        :param origin_repeat_emb:
        :return:
        """
        # print(self.extract_results, list(self.extract_results), self.extract_results['indices'].tolist()[0])
        extract_results = clean_trace(self.extract_results)
        sorted_extract_results = extract_results.sort_values(by='y1')
        sorted_extract_results['pathid'] = range(len(sorted_extract_results))

        # CHECKPOINT: each 'clean_trace' operation need an input dataframe more than one trace
        if sorted_extract_results.shape[0] > 1:
            final_extract_results = clean_trace2(sorted_extract_results)
        else:
            final_extract_results = sorted_extract_results
        trace_cluster_range_list = cluster_generator(final_extract_results)
        # trace_cluster_range_list.append((max_win_start, max_win_end))
        resi_idx_list = mask_idx_list(trace_cluster_range_list)
        print("trace_cluster_range_list", trace_cluster_range_list)

        # CHECKPOINT: overlap!!!!!!!
        have_overlap = check_overlap(trace_cluster_range_list, overlap_threshold=self.overlap_threshold)
        # if this length doesn't satisfy overlap requirement, record this in the length_overlap_dict and no need to proceed forward
        if have_overlap:
            print(f'length {estimate_repeat_length} and longer have overlaps!')

            # update the overlap dictionary
            self.length_overlap_dict = {length: True if length >= estimate_repeat_length else overlap for
                                        length, overlap in self.length_overlap_dict.items()}

        else:
            # Step 7: refine detected repeat boundary
            extract_results, refine_seq_dict, all_aln_score = refine_repeat(self.emb,
                                                                            self.seq,
                                                                            [max_win_start, max_win_end],
                                                                            trace_cluster_range_list,
                                                                            weight_repeat_emb,
                                                                            globalmode=self.globalmode)

            origin_extract_results, origin_refine_seq_dict, origin_all_aln_score = refine_repeat(self.emb,
                                                                                                 self.seq,
                                                                                                 [max_win_start, max_win_end],
                                                                                                 trace_cluster_range_list,
                                                                                                 origin_repeat_emb,
                                                                                                 globalmode=self.globalmode)

            # Step 8: build repeat alignments and output everything
            # print("extract_results", extract_results, "refine_seq_dict", refine_seq_dict)
            # remove any overlap based on score

            repeat_seq_dict = create_repeat_aln(extract_results, self.seq, [max_win_start, max_win_end], refine_seq_dict)

            # Step 9: finally summarize the detected information on this specific length estimation
            repeat_num = len(repeat_seq_dict) - 1
            repeat_len_list = [len(repeat_seq.replace('-', '')) for name, repeat_seq in repeat_seq_dict.items() if name != 'rep']
            repeat_len = np.mean(repeat_len_list)
            repeat_total_len = repeat_num * repeat_len
            avg_origin_aln_score = (sum(origin_all_aln_score) - max(origin_all_aln_score)) / (len(origin_all_aln_score) - 1)
            if len(repeat_len_list) > 1:
                avg_origin_aln_len = (sum(repeat_len_list) - max(repeat_len_list)) / (len(repeat_len_list) - 1)
            else:
                avg_origin_aln_len = sum(repeat_len_list) - max(repeat_len_list)
            avg_coverage_to_rep = np.mean([repeat_len/estimate_repeat_length for repeat_len in repeat_len_list])
            print("check score matrix")
            rep_repeat_range = copy.copy(self.rep_repeat_range)
            sub_matrix = copy.copy(self.sub_matrix)
            score_matrix = copy.copy(self.score_matrix)
            result_info_dict = {"repeat_num": repeat_num,
                                "repeat_len": repeat_len,
                                "repeat_total_len": repeat_total_len,
                                "trace_cluster_range_list": trace_cluster_range_list,
                                "mask_resi_idx_list": resi_idx_list,
                                "repeat_seq_dict": repeat_seq_dict,
                                "weight_aln_score_list": all_aln_score,
                                "origin_aln_score_list": origin_all_aln_score,
                                "avg_origin_aln_score": avg_origin_aln_score,
                                "avg_origin_aln_len": avg_origin_aln_len,
                                "avg_coverage_to_rep": avg_coverage_to_rep,
                                "sub_matrix": sub_matrix,
                                "score_matrix": score_matrix,
                                "rep_repeat_range": rep_repeat_range}

            # update the repeat information dictionary and mask_index_list for next search only when average length of repeat is higher than a threshold
            # score can be used to screen significant results as well
            # only detection with more than one trace is recorded
            if result_info_dict['repeat_len'] > self.repeat_length_threshold and result_info_dict['avg_origin_aln_score'] > self.repeat_score_threshold and result_info_dict['avg_origin_aln_len'] > self.repeat_avg_length_threshold and len(result_info_dict['origin_aln_score_list']) > 1:
                self.length_repeat_info_dict[estimate_repeat_length] = result_info_dict
                with open(f'{self.out_dir}one_result.pkl', 'wb') as f:
                    pickle.dump(result_info_dict, f)

    def search_all_len(self, score_matrix, repeat_class_idx):
        """
        function to execute a full search on repeats - perform search_one_len for all repeat length in the list and store relevant information

        """

        # update score_matrix of RepeatDector class
        self.score_matrix = score_matrix
        print("Consider these lengths: ", self.estimate_repeat_length_list)

        # iterate the length list
        for repeat_len in self.estimate_repeat_length_list:
            print(f"Now try length {repeat_len}")

            # only go ahead if the length doesn't have overlap
            if not self.length_overlap_dict[repeat_len]:

                # search repeats with this length
                max_win_start, max_win_end, weight_repeat_emb, origin_repeat_emb = self.search_one_len(repeat_len)
                # only go ahead if scan_repeat generates positive hits
                if self.extract_results.shape[0] > 1:
                    self.define_boundary(repeat_len, max_win_start, max_win_end, weight_repeat_emb, origin_repeat_emb)

        # if no significant repeat info has been recorded in this round of search - then no repeat anymore!
        if len(self.length_repeat_info_dict) == 0:
            self.norepeat = True
        else:
            # do some summary check to select one estimated length and corresponding repeat info and update mask_index_list based on it
            length_with_max_total_len = max(self.length_repeat_info_dict, key=lambda x: self.length_repeat_info_dict[x]['repeat_total_len'])
            self.resi_idx_list += self.length_repeat_info_dict[length_with_max_total_len]['mask_resi_idx_list']


class AllRepeatDector(RepeatDetector):
    """
    main class to do iterative search for repeats
    """

    def __init__(self,
                 emb,
                 seq,
                 win_len=15,
                 min_span=15,
                 ssa_sigma_factor=2.0,
                 ssa_score_threshold=0.25,
                 transitive_score_threshold=0.0,
                 scan_score_threshold=0.25,
                 scan_sigma_factor=2.0,
                 scan_with_weighted=True,
                 repeat_length_threshold=10,
                 repeat_score_threshold=0.0,
                 repeat_avg_length_threshold=10,
                 overlap_threshold=0.3,
                 use_transitivity=True,
                 mask_matrix=None,
                 weight_repeat_emb=True,
                 globalmode=True,
                 draw_during_procedure=True,
                 out_dir=None):

        super().__init__(emb, seq, win_len, min_span, ssa_sigma_factor, ssa_score_threshold, transitive_score_threshold, scan_score_threshold, scan_sigma_factor, scan_with_weighted,
                         repeat_length_threshold, repeat_score_threshold, repeat_avg_length_threshold, overlap_threshold, use_transitivity, mask_matrix, weight_repeat_emb, globalmode, draw_during_procedure, out_dir)
        self.emb = emb
        self.seq = seq
        self.all_repeat_info_dict = defaultdict(dict)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # @staticmethod
    def search_all_repeat(self):

        # iteratively search repeats - 'norepeat' is a mask for deciding stop searching or not
        repeat_class_idx = 0
        while not self.norepeat:

            print(f"Start searching for repeat class {repeat_class_idx}")
            print("These residues are masked in this iteration: ", self.resi_idx_list)

            # update the possible length list, the score matrix and the aln_pair_score_dict
            self.get_len_list(repeat_class_idx)

            # status of 'norepeat' may be changed here
            if not self.norepeat:

                # update score matrix and relevant inputs for detection procedure
                # assert self.resi_idx_list is not None
                # change score matrix inside class RepeatDector
                # score_matrix = mask_score_matrix(self.score_matrix, self.resi_idx_list)

                # conduct search for repeats (possible length list and alignment result are ready)
                self.search_all_len(self.score_matrix, repeat_class_idx)

                # record the information of this kind of repeat in self.all_repeat_info_dict
                self.all_repeat_info_dict[repeat_class_idx] = self.length_repeat_info_dict

                repeat_class_idx += 1
                # once at least one iteration is run then update_score_matrix is set as True
                self.update_score_matrix = True

            # if 'norepeat' is marked as True in get_len_list, stop everything
            else:
                break
