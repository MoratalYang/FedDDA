# Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation (ICML 25)

This repository is built for the paper Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation (ICML 25)

## 📦 Requirements

- Python 3.8+
- Pytorch 1.10.0+

To install requirements:

```
pip install -r requirements.txt
```

## 📁 Data Preparation

You need to manually download and unzip data under `data/` file catalog. Remember to set the correct data path via the `--root` argument when running experiments. The file structure looks like:

### Example
```
data/
 └── Office31/
      ├── amazon/
      ├── dslr/
      └── webcam/
```

### Data List
- [Office31](https://www.kaggle.com/datasets/mei1963/office31/data)
- [OfficeHome](https://www.kaggle.com/datasets/iamjanvijay/officehome)
- [PACS](https://www.kaggle.com/datasets/nickfratto/pacs-dataset)
- [DomainNet](https://www.kaggle.com/datasets/mei1963/domainnet)


## 🚀 How to Run

You can run **federated_main.py** with specific arguments. After the experiments, all the results are finished and save to `output/`.

### Example

```bash
python federated_main.py \
    --trainer FEDDDA \
    --dataset Office31 \
    --device_id 0 \
    OPTIM.MAX_EPOCH 1
```

### Key Arguments

| Argument          | Description                                   |
| ----------------- | --------------------------------------------- |
| `--trainer`       | Training method                               |
| `--dataset`       | Dataset name                                  |
| `--device_id`     | GPU device ID                                 |
| `OPTIM.MAX_EPOCH` | Maximum number of training epochs             |

For more detailed configuration settings, refer to the `configs/` directory and the extended command-line arguments.

## 📚 Citation

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{yang2025FedDDA,
  title={Federated Disentangled Tuning with Textual Prior Decoupling and Visual Dynamic Adaptation},
  author={Yang, Yihao and Huang, Wenke and Wan, Guancheng and Yang, Bin and Ye, Mang},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
