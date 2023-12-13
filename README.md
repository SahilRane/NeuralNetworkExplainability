## Explainable Transfer Learning for CNNs and Transformers
Authors: Forrest Bicker, Sahil Rane, Rohan Subramanian

This repository containe PyTorch code used for fine-tuning the vision transformer (ViT) and ResNet34 on the CIFAR-10 dataset, and applying GRAD-CAM (Class Activation Mapping) and LRP (Layer-wise relevance propagation) for explainability.

Code:
<ul>
  <li>requirements.txt: Contains a list of packages that can be installed in the virtual environment using `pip install -r requirements.txt`.</li>
  <li>training.ipynb: Notebook that describes how to train ViT with HuggingFace, AlexNet and ResNet34.</li>
  <li>traincnn.py: Contains functions used to train AlexNet and ResNet34</li>
  <li>cnn-explain.ipynb: Notebook that demonstrates Grad-CAM explainability on the ResNet34 fine-tuned in training.ipynb</li>
  <li>transformer-explain.ipynb: Notebook that trains ViT implementation compatible with LRP and demonstrates LRP explainability.</li>
</ul>

Directories:
<ul>
  <li>models: Contains saved model objects for ResNet34 and ViT fine-tuned on CIFAR-10 that can be reloaded for use.</li>
  <li>Transformer-Explainability: Modified code from Hila Chefer implementing LRP</li>
</ul>

This was part of our Neural Networks Final Project (CS152 at Harvey Mudd College).
