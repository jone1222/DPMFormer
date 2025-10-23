# [ICCV 2025]  Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation

### [** Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation**](https://openaccess.thecvf.com/content/ICCV2025/html/Jeon_Exploiting_Domain_Properties_in_Language-Driven_Domain_Generalization_for_Semantic_Segmentation_ICCV_2025_paper.html)
>[Seogkyu Jeon], [Kibeom Hong]†, [Hyeran Byun]†\
>Yonsei University, Sookmyung Women's University\
>ICCV 2025

## Environment
### Requirements
- The requirements can be installed with:
  
  ```bash
  conda create -n tqdm python=3.9 numpy=1.26.4
  conda activate tqdm
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install xformers==0.0.20
  pip install mmcv-full==1.5.3 
  ```
### Pre-trained VLM Models
- Please download the pre-trained CLIP and EVA02-CLIP and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | CLIP | `ViT-B-16.pt` |[official repo](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B` |[official repo](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#eva-02-clip-series)|

### Checkpoints
- You can download **dpmformer** model checkpoints:

  | Model | Config | Link |
  |-----|-----|:-----:|
  | `tqdm-clip-vit-b-gta` | [config](https://github.com/jone1222/DPMFormer/blob/main/configs/dpmformer/dpmformer_clip_vit-l_1e-5_20k-g2c-512.py) |[download link](https://drive.google.com/file/d/1xijvHa6e5nLHDcI2RvJBLtGOWKVitAXA/view?usp=sharing)|
  | `tqdm-eva02-clip-vit-l-gta` | TBD | TBD |


## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files](https://github.com/ByeongHyunPak/tqdm/tree/main/configs/_base_/datasets) following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```
## Train & Test
- Check scripts in the "scripts/"

## Citation
If you find our code helpful, please cite our paper:
```bibtex
@article{jeon2025dpm,
  title     = {Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation},
  author    = {Jeon, Seogkyu and Hong, Kibeom, and Byun, Hyeran},
  journal   = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2025}
}
```

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DAFormer](https://github.com/lhoyer/DAFormer)
- [TLDR](https://github.com/ssssshwan/TLDR)
