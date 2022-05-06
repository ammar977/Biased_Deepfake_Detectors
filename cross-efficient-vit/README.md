# Combining EfficientNet and Vision Transformers for Video Deepfake Detection

Code for Video Deepfake Detection model from "Combining EfficientNet and Vision Transformers for Video Deepfake Detection" available on Arxiv and was submitted to ICIAP 2021 [<a href="https://arxiv.org/abs/2107.02612">Pre-print PDF</a>]. Using this repository it is possible to train and test the two main architectures presented in the paper, Efficient Vision Transformers and Cross Efficient Vision Transformers, for video deepfake detection.
The architectures exploits internally the <a href="https://github.com/lukemelas/EfficientNet-PyTorch">EfficientNet-Pytorch</a> and <a href="https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch">ViT-Pytorch</a> repositories.

**Compared to the original git, we modified and simplified the input structure to be conform to FakeAVCeleb gender and ethnic structure.**

**The main modified files are :**
```
preprocessing\detect_faces.py
preprocessing\extract_crops.py
cross-efficient-vit/test.py
```
# Setup
Setup Python environment using conda:
```
conda env create --file environment.yml
conda activate deepfakes
export PYTHONPATH=.
```

# Get the data
It is the FakeAVCeleb dataset, available upon request.

# Preprocess the data
The preprocessing phase is based on <a href="https://github.com/selimsef/dfdc_deepfake_challenge">Selim Seferbekov implementation</a>.

In order to perform deepfake detection it is necessary to first identify and extract faces from all the videos in the dataset.
Detect the faces inside the videos:
```
cd preprocessing
python3 detect_faces.py --data_path '/home/cosminciausu/Documents/cs523/project/data/FakeAVCeleb_v1.2/FakeVideo-RealAudio/' --processes 10
```
The input data structure assumes one folder per ethnicty and one sub folder per gender.
The root folder is the type of data, fake or real video.

The extracted boxes will be saved inside the "path/to/videos/boxes" folder.
In order to get the best possible result, make sure that at least one face is identified in each video. If not, you can reduce the threshold values of the MTCNN on line 38 of face_detector.py and run the command again until at least one detection occurs.
At the end of the execution of face_detector.py an error message will appear if the detector was unable to find faces inside some videos.

If you want to manually check that at least one face has been identified in each video, make sure that the number of files in the "boxes" folder is equal to the number of videos. To count the files in the folder use:
```
cd path/to/videos/boxes
ls | wc -l
```

Extract the detected faces obtaining the images:
```
python3 extract_crops.py --data_path  '/home/cosminciausu/Documents/cs523/project/data/FakeAVCeleb_v1.2/FakeVideo-RealAudio/' --output_path '/home/cosminciausu/Documents/cs523/project/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/FakeAVCeleb/FakeVideo-RealAudio'
```

Repeat detection and extraction for all the different fake/real vids of your dataset.

We suggest to exploit the --output_path parameter when executing extract_crops.py to build the folders structure properly.

# Evaluate
Move into the choosen architecture folder you want to evaluate and download the pre-trained model:

(Cross Efficient ViT)
```
cd cross-efficient-vit
wget http://datino.isti.cnr.it/efficientvit_deepfake/cross_efficient_vit.pth
```


If you are unable to use the previous urls you can download the weights from [Google Drive](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1?usp=sharing).


Then, issue the following commands for evaluating a given model giving the pre-trained model path and the configuration file available in the config directory:
```
python3 test.py --model_path 'cross_efficient_vit.pth' --config 'configs/architecture.yaml'  
```

**By default it assumes the availability of a txt file containing the videos relative path for the predictions.
It is hard coded, please find the file definition at line 171**


# Reference
```
@misc{coccomini2021combining,
      title={Combining EfficientNet and Vision Transformers for Video Deepfake Detection}, 
      author={Davide Coccomini and Nicola Messina and Claudio Gennaro and Fabrizio Falchi},
      year={2021},
      eprint={2107.02612},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

