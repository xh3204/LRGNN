
# LRGNN
Speedup Robust Graph Structure Learning with Low-Rank Information
![Speedup Robust Graph Structure Learning with Low-Rank Information](system_model.png "Model Architecture")

## Installation
1. conda create -n lrgnn python=3.9.11
2. conda activate lrgnn
3. conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
4. pip install -U tensorly
5. pip install deeprobust
6. pip install torch_geometric
7. pip install torch_sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

## How to run
```bash
### LRGNN
python main.py --dataset cora

### LRGNN(S)
python main.py --dataset cora --sparse

```
## Citation
If you find this repository, e.g., the paper, code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{xu2021speedup,
  title={Speedup robust graph structure learning with low-rank information},
  author={Xu, Hui and Xiang, Liyao and Yu, Jiahao and Cao, Anqi and Wang, Xinbing},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2241--2250},
  year={2021}
}
```
