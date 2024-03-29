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
  author="Šimon {Bilík}",
  title="Feature space reduction as data preprocessing for the anomaly detection",
  address="Brno University of Technology, Faculty of Electrical Engineering",
  booktitle="Proceedings I of the 27th Conference STUDENT EEICT 2021",
  chapter="171163",
  howpublished="online",
  institution="Brno University of Technology, Faculty of Electrical Engineering",
  year="2021",
  month="april",
  pages="415--419",
  publisher="Brno University of Technology, Faculty of Electrical Engineering",
  type="conference paper"
}
```

