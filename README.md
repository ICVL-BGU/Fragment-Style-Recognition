# Recognizing Artistic Style of Archaeological Image Fragments Using Deep Style Extrapolation

This repository contains all the code for training and evaluating the models from our paper "Recognizing Artistic Style of Archaeological Image Fragments Using Deep Style Extrapolation" (read it [here](https://arxiv.org/pdf/2501.00836), to be presented at HCII 2025.

## Setup
1. Clone this repository.
2. To install the required dependencies run ```pip install -r requirements.txt```.
3. If you wish to use the proposed POMPAAF dataset, download it from [here](https://bgu365.sharepoint.com/:f:/s/ICVL/ElyJxN--aONDsd83cVwu4FABsPQqGKrV_3HYb480omJHHA?e=sna8Bh) and place the extracted folders in the `pompaaf/` directory.

## Configuration
Edit the `config.yml` file to set your preferred training and evaluation parameters. Key parameters include:
- `do_train`: Set to `true` to train the model.
- `do_eval`: Set to `true` to evaluate the model.
- `max_epochs`: Maximum number of training epochs (training will use an "early stopping" mechanism).
- `data_dir`: Path to the dataset directory.
- `n_styles`: Number of styles in your dataset.
- `ckpt_dir`: Path to the directory containing model weights.
- `model_type`: Either `"proposed"`, `"ft"`, or `"cnn"` to use each approach presented in the paper.
- `model_name`: To uniquely identify your model in the `ckpt_dir`.
- `sx_name`: Use `"new"` to train a Style Extrapolator from scratch or supply a checkpoint file in the `ckpt_dir`.

More parameters and further explanations can be examined in `config.yml`.

## Usage
After defining your configurations, simply run ```python main.py config.yml``` to train/evaluate/both.

## Citation
If you are using this code or the POMPAAF dataset, please cite:
```
@article{elkin2025recognizing,
  title={Recognizing Artistic Style of Archaeological Image Fragments Using Deep Style Extrapolation},
  author={Elkin, Gur and Shahar, Ofir Itzhak and Ohayon, Yaniv and Alali, Nadav and Ben-Shahar, Ohad},
  journal={arXiv preprint arXiv:2501.00836},
  year={2025}
}
```

