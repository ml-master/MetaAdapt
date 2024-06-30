# Introduction

This is a reproduction of ACL 2023 Paper [MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning](https://arxiv.org/abs/2305.12692)

<img src=pics/intro.png>

## Data & Requirements

原论文的数据集都是公开的，可以在网上找到他们。<br>
这个仓库的数据集是基于 GossipCop 进行改编的,你可以在此找到他们。(https://github.com/junyachen/Data-examples)<br>
请确保每种类型的数据集下包含**real.json**和**fake.json**，你可以使用`dataprocess.py`对数据集进行预处理。<br>
要运行代码，需要 PyTorch 和 Transformers，具体请参阅 requirements.txt 了解运行环境。

## Run MetaAdapt

```bash
python src/metaadapt.py --source_data_path=PATH/TO/SOURCE --source_data_type=SOURCE_DATASET --target_data_path=PATH/TO/TARGET --target_data_type=TARGET_DATASET --output_dir=OUTPUT_DIR;
```

执行上述命令（带参数）以调整错误信息检测模型，从 FEVER、GettingReal、GossipCop、LIAR 和 PHEME 中选择源数据集，从 GossipCop_Content_Based、GossipCop_Style_Based、GossipCop_Story_Based 和 GossipCop_Integration_Based 中选择目标数据集。采用的模型是 RoBERTa，元学习的功能版本写在 roberta_utils.py 中。训练好的模型和评估指标可以在 OUTPUT_DIR 中找到。我们提供了一个从 GossipCop_Origin 调整到 GossipCop_Story_Based 的示例命令，其中包含以下学习率和温度参数：

```bash
python src/metaadapt.py --source_data_path=data/GossipCop_Origin --source_data_type=gossipcop --target_data_path=data/GossipCop_Story_Based --target_data_type=gossipcop_story_based --learning_rate_meta=1e-5 --learning_rate_learner=1e-5 --softmax_temp=0.1 --output_dir=gossipcop_story;
```

## Performance

### 下面展示了所有源-目标组合的 10 样本跨域性能。

<img src=pics/result.png width=1000><br>

### 消融实验

<img src=pics/消融实验.png width=1000>

## Citing

如果使用此方法请参照下列引用。

```
@inproceedings{yue2023metaadapt,
  title={MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning},
  author={Yue, Zhenrui and Zeng, Huimin and Zhang, Yang and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```
