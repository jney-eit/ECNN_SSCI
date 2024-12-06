# ECNN: A low complexity, Adjustable CNN for Industrial Pump Monitoring Using Vibration Data

Repository contains code to train and test ECNN, CNN and Threshold model proposed in the SSCI paper "ECNN: A low complexity, Adjustable CNN for Industrial Pump Monitoring Using Vibration Data".

**Note**: The original dataset used in the paper was provided by KSB and cannot be published due to confidentiality reasons. However, the provided code can also be used with other datasets, if provided in the correct parquet format. 
Therefore, please modify the dataset_path variable in the parameters.yaml file. 


## 📋 HowTo 

For each of the different algorithms a separate parameters file is provided and can be execute like: 

`python main.py <name_of_parameters_file>`.

- [parameters_enhanced_cnn.yaml](parameters/parameters_enhanced_cnn.yaml) contains the configuration for the CNN enhanced with the average difference between the current sample and the mean of the normal samples of the current pump.

- [parameters_threshold.yaml](parameters/parameters_threshold.yaml) contains the configuration for the thresholding algorithm that tries to find a threshold between the mean of the normal sampels and the mean of the anormal samples.

- [parameters_cnn.yaml](parameters/parameters_cnn.yaml) contains the configuration for the default CNN without any addtional input 


## 📂 Directory Structure


**output_files**: Contains configurations, errors and outputs of trained models 

**parameters**: Contains model configuration as yaml file that is loaded before training. Allows to set model, channel, training and other properties.

**trained_models**: Contains saved pytorch models 

**main.py**: Starting point of application. Loads parameters file and starts training/testing etc.

**model.py**: Contains all the different NN Pytorch models

**tester.py**: Contains all functions used for evaluation of model

**trainer**: Contains different trainer files to apply different training strategies to the model 

**sweeper.py**: Contains all functions used to train and evaluate multiple model configurations by sweeping over specific parameters

**utils.py**: General auxiliary functions

**visualization.py**: Contains functions to plot/visualize results 


