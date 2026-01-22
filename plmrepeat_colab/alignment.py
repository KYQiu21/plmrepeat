import sys
import pandas as pd
import numpy as np
import torch

# sys.path.append('/content/plmrepeat_colab/new-pLM-BLAST-main')
import alntools as aln
import warnings

warnings.filterwarnings("ignore")


def self_alignment(emb, seq, bfactor=1, sigma_factor=2, window=10, min_span=10, gap_penalty=0,
                   column='score'):
    """
    Perform the sequence self-alignment by pLM-BLAST
    """

    # load sequence
    seq1, seq2 = seq, seq

    # load embedding
    emb1, emb2 = emb, emb

    # run plm-blast
    # cosine similarity substitution matrix
    sub_matrix = aln.base.embedding_local_similarity(emb1, emb2)
    # traces from SW based on scoring matrix
    paths = aln.alignment.gather_all_paths(sub_matrix, gap_penalty=gap_penalty, bfactor=bfactor)
    # alignments from traces
    spans_locations = aln.prepare.search_paths(
        sub_matrix, paths=paths, window=window, sigma_factor=sigma_factor,
        globalmode=False, min_span=min_span)
    # save as a df
    results = pd.DataFrame(spans_locations.values())
    results['i'] = 0

    return results, sub_matrix


def scan_repeat(emb: torch.Tensor, seq: str, repeat: [int, int], repeat_emb: np.ndarray,
                bfactor=1, sigma_factor=2.0, window=15, min_span=15, gap_penalty=0, column='score'):
    """
    Perform pLM-BLAST between identified representative repeat and the full-length sequence
    """

    start = repeat[0]
    end = repeat[1]
    fl_seq = seq
    repeat_seq = seq[start:end]
    print("Representative repeat", repeat_seq)
    fl_emb = emb

    repeat_sub_matrix = aln.base.embedding_local_similarity(repeat_emb, fl_emb)
    paths, score_matrix = aln.alignment.gather_all_paths(repeat_sub_matrix,
                                                         gap_penalty=gap_penalty,
                                                         bfactor=bfactor,
                                                         with_scores=True)
    spans_locations = aln.prepare.search_paths(
        repeat_sub_matrix, paths=paths, window=window, min_span=min_span, sigma_factor=sigma_factor,
        globalmode=False)

    results = pd.DataFrame(spans_locations.values())
    results['i'] = 0

    # add the original diagonal of the representative repeat
    repeat_diagonal = np.arange(start, end + 1)
    repeat_diagonal = np.column_stack((repeat_diagonal, repeat_diagonal))
    repeat_diagonal[:, 1] -= start
    # print("repeat_diagonal", repeat_diagonal)

    if results.shape[0] != 0:
        # this is the list of residue which are detected in this repeat profile
        resi_idx_list = list(set(np.vstack(results['indices'].tolist())[:, 0].tolist()))

    else:
        resi_idx_list = []

    repeat_diagonal_row = {'pathid': 0, 'spanid': 0, 'span_start': 0, 'span_end': 0, 'indices': repeat_diagonal,
                           'score': 0.99, 'len': end - start, 'mode': 'local', 'i': 0,
                           'y1': start, 'x1': 0, 'dist_to_diag': 0, 'nogap_len': end - start}

    # print("before ", results)
    # results = results.append(repeat_diagonal_row, ignore_index=True)
    # results.loc[len(results)] = repeat_diagonal_row
    results = pd.concat([results, pd.DataFrame([repeat_diagonal_row])], ignore_index=True)
    # print("after ", results)

    return results, repeat_diagonal_row, score_matrix


def refine_repeat(emb: torch.Tensor, seq: str, rep_repeat: [int, int], refine_region_list: list[[int, int]],
                  repeat_emb: np.ndarray, bfactor='global', sigma_factor=2, window=10,
                  min_span=10, gap_penalty=0, column='score', globalmode=True):
    """
    Perform global pLM-BLAST between identified representative repeat and the defined region
    (deemed as containing one repeat)
    """

    rep_start = rep_repeat[0]
    rep_end = rep_repeat[1]
    repeat_seq = seq[rep_start:rep_end]

    all_results = pd.DataFrame()
    all_alns = []
    all_aln_score = []
    refine_seq_dict = dict()
    refine_seq_list = []

    # Iterate each region to obtain the refined repeat
    for idx, refine_region in enumerate(refine_region_list):

        refine_start = refine_region[0]
        refine_end = refine_region[1]
        refine_seq = seq[refine_start:refine_end]
        refine_emb = emb[refine_start:refine_end]

        # print(repeat_seq, len(repeat_seq), repeat_emb.shape, refine_seq, len(refine_seq), refine_emb.shape)

        repeat_sub_matrix = aln.base.embedding_local_similarity(repeat_emb, refine_emb)
        paths, score_matrix = aln.alignment.gather_all_paths(repeat_sub_matrix,
                                                             gap_penalty=gap_penalty,
                                                             bfactor=bfactor,
                                                             with_scores=True)

        spans_locations = aln.prepare.search_paths(
            repeat_sub_matrix, paths=paths, window=window, min_span=min_span, sigma_factor=sigma_factor,
            globalmode=globalmode)

        results = pd.DataFrame(spans_locations.values())
        results['i'] = 0

        if results.shape[0] != 0:
            # this is the list of residue which are detected in this repeat profile
            resi_idx_list = list(set(np.vstack(results['indices'].tolist())[:, 0].tolist()))
            row = results.iloc[0]
            # best_aln = aln.alignment.draw_alignment(row.indices, refine_seq, repeat_seq, output='str')
            # all_alns.append(best_aln)
            all_aln_score.append(results['score'][0])
            refine_seq_list.append(refine_seq)
            # refine_seq_dict[str(idx + 1)] = refine_seq

        else:
            resi_idx_list = []

        for repeat_name, repeat_seq in enumerate(refine_seq_list):
            refine_seq_dict[str(repeat_name+1)] = repeat_seq

        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results, refine_seq_dict, all_aln_score
