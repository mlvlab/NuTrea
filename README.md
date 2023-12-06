# NuTrea
Official implementation of NeurIPS 2023 paper, "NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA" by Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, and Hyunwoo J. Kim.

<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/mlvlab/NuTrea/main/assets/main_fig.png">
</p>


## Setup

1. **Environment**

```
git clone https://github.com/mlvlab/NuTrea.git
cd NuTrea
```
```
conda create -n nutrea python=3.8
conda activate nutrea
```
```
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg

pip install -r src/requirements.txt
```

2. **Datasets**

Download dataset [here](https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv).
Unzip data files and place under 'data/' directory as
```
NuTrea/
  ├── src/
  ├── checkpoints/
  ├── ...
  └── data/
       ├── webqsp/
       ├── CWQ/
       ├── metaqa-1hop/
       ├── metaqa-2hop/
       └── metaqa-3hop/
```

3. (optional) **Checkpoints** 

Download the pretrained checkpoint(s) and place under "checkpoints/" directory.

| Dataset |  Metric | Hit@1 | F1  |  Checkpoint   |
|:--------------:|:---------------:|:------------:|:---------------:|:------------:|
|  WebQuestionsSP  |    Hit@1    |  77.43   |  71.01   | [drive](https://drive.google.com/file/d/1RS6lXdv0_hRu3EdP8wE63LRLQTC5erNP/view?usp=drive_link)  |
|  WebQuestionsSP  |    F1    |  76.88    |  72.70  | [drive](https://drive.google.com/file/d/1Gz9wdAOGCi7gGKKLTYSnoxeJ2X4AQlfW/view?usp=drive_link)  |
|  ComplexWebQuestions  |    Hit@1    |  53.61   |  49.41   | [drive](https://drive.google.com/file/d/1ldTP3yiyCX3W_oq2byX41SmxseH5Dhp8/view?usp=drive_link)  |
|  ComplexWebQuestions  |    F1    |   53.16   |  49.53   | [drive](https://drive.google.com/file/d/1fNidJ_Exbw0TeM4QS8e05WOgAVltp88Y/view?usp=drive_link)  |
|  MetaQA-1hop  |    Hit@1    |  97.40    |   97.53   | [drive](https://drive.google.com/file/d/1z6gUY3Ayb4NR0ruGxT77y8RlZSvRg2ky/view?usp=drive_link)  |
|  MetaQA-1hop  |    F1    |  97.25    |   97.62   | [drive](https://drive.google.com/file/d/1ZbR6WJkd9OkHNzjmxjpRWKBoB73pC9c4/view?usp=drive_link)  |
|  MetaQA-2hop  |    Hit@1    |  99.99    |   99.82    | [drive](https://drive.google.com/file/d/1h__vAqWONr6Doa1I4yi96HP41i5UuRye/view?usp=drive_link)  |
|  MetaQA-2hop  |    F1    |   99.99    |   99.82    | [drive](https://drive.google.com/file/d/1drEEy3yGPpS_xz66maHfEZEdMcbhylVD/view?usp=drive_link)  |
|  MetaQA-3hop  |    Hit@1    |   98.89   |  87.06     | [drive](https://drive.google.com/file/d/1eJ5uoCv34V-5bGUJG_cQ7EAkAkaCw_8D/view?usp=drive_link)  |
|  MetaQA-3hop  |    F1    |    98.89    |   87.06    | [drive](https://drive.google.com/file/d/19-khAGYkYNhJevXPqdDcY2al5PZi5E7a/view?usp=drive_link)  |

* Note, MetaQA-2hop and -3hop Hit@1 and F1 model checkpoints contain identical parameters, respectively.
* Also note that train / eval performance may vary across different GPU and environments.

## Run
Runnable scripts are in the "runs/" directory. To run evaluation with checkpoints,
```
sh runs/wqp_eval.sh
```
To train the model,
```
sh runs/wqp_train.sh
```
Run different scripts to test different datasets and parameters.


## UPDATES
**2023.12** Initial code release


## Citation

```
@inproceedings{choi2023nutrea,
  title={NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA},
  author={Choi, Hyeong Kyu and Lee, Seunghun and Chu, Jaewon and Kim, Hyunwoo J.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## License
Code is released under MIT License.
