# AE-Reconstruction-And-Feature-Based-AD

**Framework for the AE reconstruction and feature based anomaly detection**

## How does the proposed anomaly detection works?

This repository implements the reconstruction-based anomaly detection method presented in the article [*Toward phytoplankton parasite detection using autoencoders*](https://link.springer.com/article/10.1007/s00138-023-01450-x). The proposed technique utilizes an autoencoder trained on the non-anomalous (OK) data, which is later used to reconstruct dataset containing both OK and anomalous (NOK) data. Various features are extracted using a comparison between the original and reconstructed data, which are classified under an assumption, that the difference will be more significant in case of the NOK data. The scheme is shown in the figure bellow:

![Reconstruction framework scheme](https://github.com/boortel/AE-Reconstruction-And-Feature-Based-AD/assets/33236294/c4955068-3825-469a-a177-fa949e33bd4c)

One-class classification is performed using the standard *Scikit* libraries based on [this]([https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py]) example. The anomaly detection threshold for each technique is derived using the equal error rate on the ROC curve as shown in the figure bellow:

![AUC_EER](https://github.com/boortel/AE-Reconstruction-And-Feature-Based-AD/assets/33236294/931f3642-d6de-47d0-9c4e-d0d9f373acd9)


## Which autoencoders are implemented?

The framework implements five autoencoders cores and six convolutional encoder-decoder pairs of various complexity and depth. The autoencoder cores are:

- Basic core 1 (BAE1): a direct connection of the encoder-decoder pair.
- Basic core 2 (BAE2): a basic scheme with the inserted fully-connected layers.
- Variational core 1 (VAE1): a basic variational autoencoder derived from [Keras Example](https://keras.io/examples/generative/vae/).
- Variational core 2 (VAE2): VAE1 scheme with the inseted fully-connected layers.
- Vector-quantised core (VQVAE1): vector-quantised autoencoder derived from [Keras Example](https://keras.io/examples/generative/vq_vae/).

Modifications of the BAE2 and VAE2 cores are illustrated on the figure bellow:

![BAE2 and VAE2 cores scheme](https://github.com/boortel/AE-Reconstruction-And-Feature-Based-AD/assets/33236294/5a8630ca-7844-4e96-b790-8884c94b9c71)


## How to run the code on sample data?

- Run *pip install requirements.txt* within the fresh conda Python 3.9. enviroment. Reinstall the TF2 following the instructions at: https://www.tensorflow.org/install/pip

- Download the V2 dataset at: [*Industry Biscuit (Cookie) dataset*](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset) and run the attached script *DatasetFolder.py* to get a folder structured dataset. For the expriments, we used training dataset of 1000 OK samples, validation dataset of 500 OK samples and test dataset of 200 OK and 200 NOK samples.

- Set the evaluation flag in the script *FrameworkTrain.py* and run it to train the selected autoencoders. Models has to be trained before evaluation and each model has to have a corresponding *ini* file in the *init* directory. All those files will be processed by the main function and this script will automatically create subdirectory in *data* to store the model weights and evaluation results.

- Use the script *FrameworkEvaluate.py* to perform evaluation on the unknown data. Set the *saveImgToFile* argument to True if you would like to have your data sorted as *OK / NOK*.

- Create custom models in the modules *ModelSaved.py* and *ModelLayers.py*. Create corresponding *ini* files in the *init* directory.

## Please note!

This repository is still under development and it is based the research presented in the following repositories:

- [CAE-VAE-AD-Preprocessing](https://github.com/boortel/CAE-VAE-AD-Preprocessing)
- [SIFT-and-SURF-based-AD](https://github.com/boortel/SIFT-and-SURF-based-AD)

## References

Please cite the article *Toward phytoplankton parasite detection using autoencoders* in your further work:

```
@inproceedings{BUT171163,
    @Article{Bilik2023,
    author={Bilik, Simon
    and Batrakhanov, Daniel
    and Eerola, Tuomas
    and Haraguchi, Lumi
    and Kraft, Kaisa
    and Van den Wyngaert, Silke
    and Kangas, Jonna
    and Sj{\"o}qvist, Conny
    and Madsen, Karin
    and Lensu, Lasse
    and K{\"a}lvi{\"a}inen, Heikki
    and Horak, Karel},
    title={Toward phytoplankton parasite detection using autoencoders},
    journal={Machine Vision and Applications},
    year={2023},
    month={Sep},
    day={13},
    volume={34},
    number={6},
    pages={101},
    issn={1432-1769},
    doi={10.1007/s00138-023-01450-x},
    url={https://doi.org/10.1007/s00138-023-01450-x}
}
```

