import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO


# functions for drawing

def draw_trace(results: pd.DataFrame, seq: str):
    plt.figure(figsize=(8, 8), dpi=400)
    matrix = np.zeros((len(seq), len(seq)), dtype=np.int32)

    for path in results['indices']:
        for coord in path:
            x, y = coord
            matrix[x, y] = 1
            matrix[y, x] = 1

    for i in range(len(seq)):
        matrix[i, i] = 1

    plt.imshow(matrix, cmap='Blues')
    # plt.show()


def draw_sub_matrix(sub_matrix: np.ndarray, sub=True, save=False, save_path=None):
    plt.figure(figsize=(10, 8), dpi=500)
    if sub:
        plt.imshow(sub_matrix, cmap='bwr')
        plt.colorbar()
        plt.title('Substitution Matrix')
    else:
        plt.imshow(sub_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('Score Matrix')

    if save:
        plt.savefig(save_path)
    # plt.show()


def draw_extract_repeat(results: pd.DataFrame, seq_len: int, repeat_len: int):
    plt.figure(dpi=400)
    matrix = np.zeros((seq_len+1, repeat_len+1), dtype=np.int32)

    for path in results['indices']:
        for coord in path:
            x, y = coord
            matrix[x, y] = 1

    plt.title('Extracted repeats', fontsize=7)
    plt.imshow(matrix, cmap='Blues')
    # plt.show()


def draw_repeat_matrix(score_matrix: np.ndarray, repeat_range: (int, int), save=False, save_path=None):

    seq_len = score_matrix.shape[0]
    for i in range(seq_len):
        score_matrix[i, i] = 1

    plt.figure(figsize=(10, 8), dpi=500)
    plt.imshow(score_matrix, cmap='Greys')
    plt.axvline(repeat_range[0], linestyle='dashdot')
    plt.axvline(repeat_range[1], linestyle='dashdot')

    # plt.axvline(43, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axvline(87, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axvline(144, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axvline(195, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axvline(466, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axvline(507, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(43, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(87, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(144, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(195, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(466, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)
    # plt.axhline(507, c='palegreen', alpha=0.5, linestyle='-', linewidth=0.5)

    plt.title("Sliding window on the score matrix", fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar()

    if save:
        plt.savefig(save_path)
    # plt.show()


def select_length(all_result_info_dict: dict, metrics='repeat_total_len'):
    """
    select the length of repeat reporting the best extraction

    metrics: which metrics used to select the repeat length

    """

    print(f"{metrics} is used to select repeat length!")
    max_idx = max(all_result_info_dict.keys(), key=lambda k: all_result_info_dict[k][metrics])
    select_result_info_dict = all_result_info_dict[max_idx]
    repeat_range_list = select_result_info_dict['trace_cluster_range_list']

    return select_result_info_dict, repeat_range_list


def save_result(result_info_dict: dict, idx: int, draw: True, save_path: str):
    """
    save outputs generated during thw workflow

    """

    if draw:
        # draw the substitution matrix
        draw_sub_matrix(result_info_dict['sub_matrix'], sub=True, save=True,
                        save_path=os.path.join(save_path, f'sub_matrix{idx}'))
        # draw the score matrix
        draw_sub_matrix(result_info_dict['score_matrix'], sub=False, save=True,
                        save_path=os.path.join(save_path, f'score_matrix{idx}'))
        # draw the sliding window
        draw_repeat_matrix(result_info_dict['score_matrix'], result_info_dict['rep_repeat_range'], save=True,
                           save_path=os.path.join(save_path, f'sliding_window{idx}'))

    # save all repeat alignment fasta
    repeat_seq_dict = result_info_dict['repeat_seq_dict']
    repeat_seq_score = result_info_dict['weight_aln_score_list']
    repeat_range = result_info_dict['trace_cluster_range_list']
    new_repeat_seq_dict = dict()
    for seq_id, seq in repeat_seq_dict.items():
        print(seq_id, seq)
        if seq_id != 'rep':
            seq_score = repeat_seq_score[int(seq_id)-1]
            repeat_start, repeat_end = repeat_range[int(seq_id)-1][0], repeat_range[int(seq_id)-1][1]
            new_repeat_seq_dict[seq_id + '|' + str(seq_score) + '|' + str(repeat_start) + '-' + str(repeat_end)] = seq
        else:
            new_repeat_seq_dict[seq_id] = seq

    records = [SeqIO.SeqRecord(seq, id=seq_id) for seq_id, seq in new_repeat_seq_dict.items()]
    SeqIO.write(records, os.path.join(save_path, f'all_repeat_alignment{idx}.fasta'), 'fasta')
