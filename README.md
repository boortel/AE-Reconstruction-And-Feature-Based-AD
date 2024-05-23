# AE-Reconstruction-And-Feature-Based-AD

**Framework for the AE reconstruction and feature based AD**

- Run *pip install requirements.txt* within the fresh conda Python 3.9. enviroment. Reinstall the TF2 following the instructions at: https://www.tensorflow.org/install/pip

- Download the V2 dataset at: [https://www.kaggle.com/imonbilk/industry-biscuit-cookie-dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset) and run the attached script *DatasetFolder.py* to get a folder structured dataset. For the expriments, we used training dataset of 1000 OK samples, validation dataset of 500 OK samples and test dataset of 200 OK and 200 NOK samples.

- Set the training and evaluation flags in the script *main.py* and run it. Models has to be trained before evaluation and each model has to have a corresponding *ini* file in the *init* directory. All those files will be processed by the main function and this script will automatically create subdirectory in *data* to store the model weights and evaluation results.

- Create custom models in the modules *ModelSaved.py* and *ModelLayers.py*. Create corresponding *ini* files in the *init* directory.

**This repository is still under development and it is based on:**
- https://github.com/boortel/CAE-VAE-AD-Preprocessing
- https://github.com/boortel/SIFT-and-SURF-based-AD

## References

Please cite following paper in your further work:

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

}
```

