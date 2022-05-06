# Biased_Deepfake_Detectors

Repo for "Detecting Racial and Gender Bias in Deepfake Detectors" group project in CS 523 Spring 22 - Boston University

Our goal is to detect racial and gender bias in recent state of art deep fake detectors on videos of humans. We do so by running various deep fake detectors on the FakeAVCeleb [1] dataset which has videos of different racial and gender groups and evaluate the performance of the detectors on different groups to identify any bias present. 

We borrowed these models from the following repositories and papers. We also adapted code from these repositories to run these models on our evaluation datasets.

* N. Bonettini, E. D. Cannas, S. Mandelli, L. Bondi, P. Bestagini and S. Tubaro, "Video Face Manipulation Detection Through Ensemble of CNNs," 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 5012-5019, doi: 10.1109/ICPR48806.2021.9412711. 
https://github.com/polimi-ispl/icpr2020dfdc

* Davide Coccomini, Nicola Messina, Claudio Gennaro, Fabrizio Falchi.  “Combining EfficientNet and Vision Transformers for Video Deepfake Detection”, arXiv.org,2021. 
https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection

Models from the first paper make use of ensemble CNN models, more precisely EfficientNetB4 CNN architectures, with an attention mechanism to see which local part of the image contributes the most to the predictions of the CNN models. These models are 

* XceptionNet 
* EfficientNetB4 
* EfficientNetB4ST 
* EfficientNetAutoAttnB4
* EfficientNetAutoAttnB4ST


Models from the second paper] use an Efficient Net CNN architecture combined with vision transformers, but contrary to the first paper, these do not use ensembles. The following model from this paper was used by us

* Cross Efficient Vision Transformer

### Our Approach on Videos
Our evaluation dataset consists of videos. To get predictions of deep fake detectors on videos, we use the following approach

1. Take 30 frames from each video
2. Use the Blazeface [9] face extractor to extract a face from each frame
3. Pass each frame to the deep fake detector to get  a score between 0 and 1. A score of zero means that the image is real and a score of 1 means that the video is fake. 
4. Average the scores of all 30 frames to get a score for video. 

We adapted code from https://github.com/polimi-ispl/icpr2020dfdc and https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection as mentioned above to evaluate these models on our evaluation datasets. 

### Predictions
The predictions of each model on our test sets are in the directory `Predictions/pred_ensembles` and `Predictions/preds_cross_efficient_transformer.txt` file. The notebook `video_predictions_colab.ipynb` was used to get predictions from the ensemble models. This notebook clones the required github repo, installs the required libraries and downloads all the model weights required. 

### Results and Metrics

We computed the following metrics after dividing the predictions into three test sets. 
1. Accuracy
2. Precision, Recall, F1
3. ROC curves (TPR vs FPR)
4. AUC of ROC curves

The code for calculating these metrics is in the notebooks `calculate_metrics_faceswap.ipynb` , `calculate_metrics_wav2lip.ipynb` and `calculate_metrics_fsgan.ipynb`. Detailed tables and plots of metrics are in these models and in the directories `Results/tables` and `Results/plots`.

###






### References:

[1] Hasam Khalid, Shahroz Tariq, Minha Kim, Simon S. Woo FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset, arXiv.org, 2021.
