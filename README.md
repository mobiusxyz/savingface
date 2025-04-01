# savingface: Humphead Wrasse Individual Identification

This repository contains the open-source code for training the artificial intelligence models described in the paper:

**Leveraging artificial intelligence for photo identification to aid CITES enforcement in combating illegal trade of the endangered humphead wrasse (_Cheilinus undulatus_)**  
*C. Y. Hau, W. K. Ngan, Y. Sadovy de Mitcheson*  
_Frontiers in Ecology and Evolution_, 2025  
DOI: [10.3389/fevo.2025.1526661](https://doi.org/10.3389/fevo.2025.1526661)  
Link: [https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2025.1526661/full](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2025.1526661/full)

## Models

This repository provides the code for training two key deep learning models:

1.  **Facial Pattern Extraction:** A [YOLOv8](https://github.com/ultralytics/ultralytics)-based model trained to detect and extract the left and right facial patterns from images of humphead wrasse. The model achieved a 99% success rate in the study.
2.  **Individual Identification:** A [ResNet-50](https://pytorch.org/hub/pytorch_vision_resnet/)-based convolutional neural network (CNN) retrained using a triplet loss function. This model compares extracted facial patterns to identify individual fish. 

## Getting Started

*(Placeholder: Add instructions on setting up the environment, installing dependencies (e.g., `requirements.txt`), and preparing the necessary datasets.)*

```bash
# Example dependency installation
pip install -r requirements.txt
```

## Training

*(Placeholder: Add detailed instructions on how to run the training scripts for both the extraction and identification models. Include necessary command-line arguments or configuration file details.)*

```bash
# Example training command for YOLOv8 extraction model
# The training process uses an IPython Notebook (.ipynb), which can be run directly in environments like Google Colab.
# Refer to the notebook for specific instructions.

# Example training/evaluation command for ResNet-50 identification model
python train/evaluation/triplet_loss/train.py
```

## Usage

*(Placeholder: Add instructions on how to use the trained models for inference, either via provided scripts or by integrating them into other applications like the 'Saving Face' app.)*

## Citation

If you use this code or the associated models in your research, please cite the original paper:

```bibtex
@article{hau2025savingface,
  title={Leveraging artificial intelligence for photo identification to aid CITES enforcement in combating illegal trade of the endangered humphead wrasse ({_Cheilinus undulatus_})},
  author={Hau, C. Y. and Ngan, W. K. and Sadovy de Mitcheson, Y.},
  journal={Frontiers in Ecology and Evolution},
  volume={13},
  year={2025},
  pages={1526661},
  doi={10.3389/fevo.2025.1526661}
}
```

## Contact

For questions about the research and methodology, please refer to the contact information in the original publication.
