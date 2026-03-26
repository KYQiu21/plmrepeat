# Protein sequence representations for repeat detection

## Install
To use pLM-Repeat, you first need to set up the python environment.
1. Create a new conda environment:
```
conda create -p your_path_to_new_env python=3.9
conda activate your_path_to_new_env
```
2. Install dependencies:
```
pip install -r requirement.txt
cd path_to_plmrepeat
pip install -e .
```
## Introduction
The workflow of pLM-Repeat can be briefly described as follows:
1. Perform self-sequence alignment using [pLM-BLAST (Kaminski et al., Bioinformatics, 2023)](https://academic.oup.com/bioinformatics/article/39/10/btad579/7277200). For more details on pLM-BLAST, please check the reference and [its repository](https://github.com/labstructbioinf/pLM-BLAST).
2. Add transitive traces
3. Build the score matrix based on all traces
4. Estimate possible lengths of repeat via the cumulative score to the diagonal
5. Determine the positions of the representative repeat using a sliding window with the estimated length
6. Compute a weighted embedding of the representative repeat
7. Conduct another round of pLM-BLAST search between representative embedding and full-length sequence embedding to extract repeat instances
8. Output alignment (score) matrix, repeat ranges, and multiple sequence alignment (experimental)

![image](https://github.com/KYQiu21/plmrepeat/blob/main/figure/figure1_workflow.jpg)

## Performance
![image](https://github.com/KYQiu21/plmrepeat/blob/main/figure/figure3_performance_case.jpg)

## Usage
Run pLM-Repeat to detect repeats
```
cd path_to_plmrepeat
plmrepeat-run --emb path_to_embedding_file --seq path_to_sequence_file --out path_to_output_dir --transitivity --draw
e.g. plmrepeat-run --emb ./example/2QJ6_A.emb --seq ./example/2QJ6_A.fasta --out ./example/2QJ6_A/ --transitivity --draw
```
See help information
```
plmrepeat-run --help
```
The colab demo version is available here: https://colab.research.google.com/drive/1ouBwciiXy7HPnddut15JAGAmREWaqaZ7. In this colab notebook, you can run pLM-Repeat in an interactive way, i.e. selecting optimal repeat length based on the score/substitution matrix plot.

## Output
Example on protein 4R36_A:\
![image](https://github.com/KYQiu21/plmrepeat/blob/main/figure/example.png)

Residue range of detected repeats:
```
(31, 46), (49, 64), (67, 82), (97, 112), (121, 136), (139, 156), (157, 172), (175, 190)
```
Repeat alignment (Please note that the multiple alignment is created by directly combining pairwise alignments of representative repeat v.s. detected repeat, so the concatenated multiple alignment could be far away from optimal):
```
>representative repeat
ALIGNGCIVGNSTKMAGE
>repeat1
AKIGENVEIAPFVYI---
>repeat2
VVIGDNNKIMANANI---
>repeat3
SRIGNGNTIFPGAVI---
>repeat4
AEIGDNNLIRENVTI---
>repeat5
TIVGNNNLLMEGVHV---
>repeat6
ALIGNGCIVGNSTKMAG-
>repeat7
IIIDDNAIISANVLM---
>repeat8
CRVGGYVMIQGGCRF---
```

## Contact
This repository is under active construction. Please get in touch via kaiyu.qiu@tuebingen.mpg.de if you encounter any issues when using the pipeline once it is ready. Many thanks!
