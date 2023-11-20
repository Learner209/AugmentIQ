<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="https://notes.sjtu.edu.cn/uploads/upload_e5ec409384816312ff0d7e0371be76d2.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" />
    <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white">
    <img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" />
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLearner209%2FAugmentIQ&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</p>

# AugmentIQ: Revolutionizing Image Quality Assessment with Advanced Data Augmentation and Dynamic Data Loading Techniques

The official code repository for the paper: **AugmentIQ: Revolutionizing Image Quality Assessment with Advanced Data Augmentation and Dynamic Data Loading Techniques.**

> 📣 Attention please:
> The README.md file and the paper has been updated, please checkout the newest version! 😉

## Introduction:

**AugmentIQ** represents a paradigm shift in the realm of Image Quality Assessment (IQA). This innovative model is a fusion of advanced methodologies from Re-IQA, which offers nuanced image quality measurement techniques, and ImageReward, known for its incisive alignment evaluation between images and textual prompts. Unlike traditional models, **AugmentIQ** excels in assessing both the aesthetic and technical quality of images and their semantic congruence with given textual descriptors, embodying a dual-capability framework that marks a significant advancement in automated image evaluation.

In essence, **AugmentIQ** is more than an addition to the compendium of IQA methodologies; it is a groundbreaking approach that aligns with the complexities of modern image generation and processing technologies. Its development signifies a new era in image quality assessment, one that is attuned to both the aesthetic beauty and the semantic relevance of images.

🤗 Please [cite AugmentIQ](https://github.com/Learner209/AugmentIQ) in your publications if it helps with your work. Please star🌟 this repo to help others notice AugmentIQ if you think it is useful. It really means a lot to our open-source research. Thank you! BTW, you may also like [`ImageReward`](https://github.com/THUDM/ImageReward), [`ReIQA`](https://github.com/avinabsaha/ReIQA), the two great open-source repositories upon which we built our architecture.

> 📣 Attention please:
> Due to the time limit, the implementation in this repo may not achieve the best result, and also considering we haven't running extensive parameters fintuning process due to time and resource limit,the best results may still be on the way ! 😉

## ❖ Contributions and Performance

⦿ **`Contributions`**:

-   Our integrated model, synthesizing the methodsologies of Re-IQA and ImageReward, represents the next step in this evolutionary path. It not only incorporates the technical advancements in assessing image fidelity and aesthetic quality but also introduces a novel dimension of evaluating text-image semantic congruence.

-   This integration signifies a broader trend in IQA research, one that acknowledges the multi-dimensional nature of image quality in the age of AI and seeks to develop assessment tools that are as dynamic and multifaceted as the images they evaluate.

## ❖ Brief Graphical Illustration of Our Methodology

Here we only show the main component of our method: the joint-optimization training approach combining three encoders while frozening their own weights.
For the detailed description and explanation, please read our full paper if you are interested.

<b>Fig. 1: Training approach</b>

## ❖ Repository Structure

The implementation of AugmentIQ is in dir [`AugmentIQ`](https://github.com/Learner209/AugmentIQ/tree/dev/augmentIQ).Please install it via `pip install -e .` or `python setup.py install`. Due to the time and resource limit, we haven't performed extensive enough parameter finetuning experiments, if you like this repo, please feel free to fork and PR to help us improve it ! 💚 💛 🤎.

## ❖ Development Environment

We run on `Ubuntu 22.04 LTS` with a system configured with a NVIDIA RTX 3090 GPU.

-   Please checkout the dev branch first.

```bash
git checkout dev
```

-   Use conda to create a env for **AugmentIQ** and activate it.

```bash
conda create -n AugmentIQ python=3.8
conda activate AugmentIQ
```

-   Install the necessary dependencies in the conda env

```bash
pip install -r requirements.txt
```

-   Then install **AugmentIQ** as a package

```
cd AugmentIQ
pip install -e .
```

## ❖ Datasets

We run on two datasets, more specifically, [`AGIQA-3k-Database`](https://github.com/lcysyzxdxc/AGIQA-3k-Database) and [`AIGCIQA2023`](https://github.com/wangjiarui153/AIGCIQA2023)
Here are some samples taken randomly from the dataset:
![](https://notes.sjtu.edu.cn/uploads/upload_008f07d38bc0f91e024906ae92024bd3.png)
![](https://notes.sjtu.edu.cn/uploads/upload_81fc68d9f7bdd0f2133de738e951759c.png)

Now the directory tree should be the following:

```
- AIGC-3k
    - image
    - data.csv
- AIGCIQA-2023
    - DATA
    - Image
      - allimg
    - prompts.xlsx
```

## ❖ Pretrained Models

Please refer to the [`Re-IQA`](https://github.com/avinabsaha/ReIQA) repository to download the `content_aware_r50.pth` and the `quality_aware_r50.pth`, and put them under the directory `$ROOT/IQAx/IQAx/re-iqa_ckpts/`. Also please take a tour to the [`ImageReward`](https://github.com/THUDM/ImageReward) repo and download `ImageReward.pt` and `med_config.json` and put them under the `$ROOT/IQAx/ImageReward/pretrained_model`.

## ❖ Finetuned models

Our finetuned models can be obtained at [jbox](https://jbox.sjtu.edu.cn/l/y1VSE2), the readers can check the `demo*.py` for reference(load a `IQAtrainer` class and then run the inference function).

## ❖ Quick Run

<details open>
  <summary><b>👉 Click here to see the example 👀</b></summary>

Please run the below commands to finetune the pretrained models on AIGCIQA-2023 dataset.

```bash
python $ROOT_DIRECTORY/augmentIQ/demo_AIGCIQA.py --aug --n_args=4 --gpu=$gpu
```

Similary on the AIGC-3k dataset.

```bash
python $ROOT_DIRECTORY/augmentIQ/demo_AIGC3K.py --aug --n_args=4 --gpu=$gpu
```

</details>

❗️Note that paths of datasets and saving dirs may be different on personal computers, please check them in the configuration files.

## ❖ Experimental Results

The training curves and validation curves of our model on AIGCIQA-2023 dataset and are shown below:

![](https://notes.sjtu.edu.cn/uploads/upload_da3be361cac4ffd6cc7ee5e8b1fcb437.png)
![](https://notes.sjtu.edu.cn/uploads/upload_fbb7ab3d5beb3c26c43efb6972e8ef42.png)

The training curves of our model on AIGC-3k dataset are shown below:

![](https://notes.sjtu.edu.cn/uploads/upload_bd6ff7f84b16c563e3fb7af246cd05a0.png)
![](https://notes.sjtu.edu.cn/uploads/upload_cc6e4d6bdb3ee523f4dd81ded4ae9103.png)

The metrics on test dataset is Spearmans Rank Correlation Coefficient(SRCCle), Pearson Correlation Coefficient(PLCC):

| Metrics/Dataset | AIGCIQA-2023(content) | AIGCIQA-2023(text_alignment) | AIGCIQA-3K(content) | AIGCIQA-3K(text_alignment) |
| :-------------- | :-------------------: | ---------------------------: | ------------------: | -------------------------- |
| PLCC            |        0.4051         |                       0.5002 |              0.4241 | 0.5523                     |
| SRCC            |        0.4046         |                       0.4719 |              0.4001 | 0.4956                     |

> 📣 Attention please:
> I have arrived at some conclusions after running some parameter fine-tuning results, the problem space of different questions can be quite different, so though my two heads(content and quality aware head and text alignment head) are very heavy and took a while to forward a tensor, its performance on transfer learning across datasets(such as AIGCIQA2023,AIGC3k,DiffusionDB,ImageReward,Pick-A-Pic) can be out of expectation, also it is noteworthy that during the process of constructing the dataset, different settings, prompts, rating schemas can make a big difference in outcomes, so according to my opinion, better results can be achieved by training the model from scratch, instead of frozening weights of some pretrained models designed to perform well only on specific datasets and then fine-tuning some final layers.

> Also it is noteworthy that the choices of loss function is also pivotal in the final results, the text to image alignment task can be achieved by training on the SPCC, PLCC metrics directly, but the content and quality related task seldom boasts a well-formulated metrics, we believe that in our task, a better metrics needs to be chosen for better results. 😉

## ❖ Acknowledgments

I extend my heartfelt gratitude to the esteemed faculty and dedicated teaching assistants of CS3324 for their invaluable guidance and support throughout my journey in image process- ing. Their profound knowledge, coupled with an unwavering commitment to nurturing curiosity and innovation, has been instrumental in my academic and personal growth. I am deeply appreciative of their efforts in creating a stimulating and enriching learning environment, which has significantly contributed to the development of this paper and my under- standing of the field. My sincere thanks to each one of them for inspiring and challenging me to reach new heights in my studies.

### ✨Stars/forks/issues/PRs are all welcome!

<details open>
<summary><b><i>👏 Click to View Contributors: </i></b></summary>

![Stargazers repo roster for @Learner209/AugmentIQ](http://reporoster.com/stars/dark/Learner209/AugmentIQ)

</details>

## ❖ Last but Not Least

If you have any additional questions or have interests in collaboration,please take a look at [my GitHub profile](https://github.com/Learner209) and feel free to contact me 😃.
