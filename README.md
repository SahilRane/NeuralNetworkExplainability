This repository containe PyTorch code used for fine-tuning the vision transformer (ViT) and ResNet34 on the CIFAR-10 dataset, and applying GRAD-CAM (Class Activation Mapping) and LRP (Layer-wise relevance propagation) for explainability.

requirements.txt: Contains a list of packages that can be installed in the virtual environment.
training.ipynb: Notebook that describes how to train ViT with HuggingFace, AlexNet and ResNet34
traincnn.py: Contains functions used to train AlexNet and ResNet34
cnn-explain.ipynb: Notebook that demonstrates Grad-CAM explainability on the ResNet34 fine-tuned in training.ipynb
transformer-explain.ipynb: Notebook that trains ViT implementation compatible with LRP and demonstrates LRP explainability.

Transformer-Explainability: Modified code from Hila Chefer implementing LRP
