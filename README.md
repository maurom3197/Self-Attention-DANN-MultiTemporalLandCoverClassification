# Self-Attention DANN for Multi-Temporal Land Cover Classification
This repository contains the official code release for the paper: "Domain-Adversarial Training of Self-Attention Based Networks for Land Cover Classification using Multi-Temporal Sentinel-2 Satellite Imagery" 
Official citation: Martini, M.; Mazzia, V.; Khaliq, A.; Chiaberge, M. Domain-Adversarial Training of Self-Attention-Based Networks for Land Cover Classification Using Multi-Temporal Sentinel-2 Satellite Imagery. Remote Sens. 2021, 13, 2564. https://doi.org/10.3390/rs13132564
paper url: https://www.mdpi.com/2072-4292/13/13/2564

## Description of the project
In this study, we investigate adversarial training of deep neural networks to bridge the domain discrepancy between distinct geographical zones. In particular, we perform a thorough analysis of domain adaptation applied to challenging multi-spectral, multi-temporal data, accurately highlighting the advantages of adapting state-of-the-art self-attention-based models for LC&CC to different target zones where labeled data are not available. Our experimentation demonstrated significant performance and generalization gain in applying domain-adversarial training to source and target regions with marked dissimilarities between the distribution of extracted features.

The repository is organized as follows:
- **training** contains all the code for adversarial training of vision transformers for land cover classification
- **visualization** 2D and 3D plot visualization of extracted features
- **testing** 

## The Visual representation of land crops classification on zone 3 (Ille-et-Vilain) of the BreizhCrops dataset
For each sub-image we show the complete region and a sub-area to facilitate the visualization of the advantage obtained by the proposed methodology. In particular, on the left the crops predictions without our domain adaptation mechanism are shown, while in the center the same predictions performed adopting DANN are proposed. On the right, ground truth labeled crops can be visualized. The improvement in the classification with DANN is evident, especially in the reduction of misclassification of wheat and meadows. :
![alt text](/images/map_zone3.jpg "Ille-et-Vilain Map Classification")

## Deep Neural Network architecture
Overview of the overall framework to train a Transformer encoder with domain-adversarial training. The multi-spectral temporal sequence X_{t×b} is first linearly projected and fused with a position encoding. Subsequently, the self-attention-based model manipulates the input series and, through a max operation applied to the last layer of the encoder, is possible to extract a token from the output sequence. Finally, gradients derived by LC&CC and Domain classifiers train the network while keeping close the distribution of source and target domains. 
<img src="/images/dann_transformer.png" width="65%" height="65%">

**Tested software versions**
- TensorFlow 2.6.x
- Keras 2.6.x
- PyTorch 1.8.1+cu102
