# Arousal prediction model
## Info
Model is used for predict human arousal level of from video. 

Inputs are sequence of single images as numpy array with RGB colormap.

Model is based on InceptionV3 feature extractor (ImageNet pretrained) and LSTM model (not pretrained).
Feature extractor process every single image to numpy array of features.
Next, features goes to LSTM, which trained with sequences of 30 frames.
So, every predict is a result of last 30 frame analysis (model is initialize with zeros features).

## Train
The training process consists of 3 stages:
* Dataset preparation. Train videos extract into frames. 
Next, it's necessary to cut out a person on each frame. 
Finally, prepare `data_file.csv` and move all frames to directories in `data`. 
Repository contains all necessary scripts.
* Run `extract_features.py` (see -h flag for details). 
It will extract all frames into sequences of features and save them as `.npy` files by 30 per 
file in `data/sequences`.
* Run `train.py` (see -h flag for details).
It will train LSTM model with features extracted on previous step.


### Metrics
Mean squared error (MSE): 0.036

Mean squared logarithmic error (MSLE): 0.0186

Mean absolute error (MAE): 0.158


## Demo
### Run instructions
Download the [weights file](https://drive.google.com/file/d/1F5yMv4BPOuJyjUMDBQooWFYQfFx2k1DD/view?usp=sharing)

Download the short [demonstration video](https://drive.google.com/file/d/1-cwAeye0304RbORnSL6SDOTgGVa00L5R/view?usp=sharing) 
from OMG dataset and put it to `demo_videos` directory for default run without arguments

Run `pip install -r requirements.txt`

Run `python3 demo.py` for running with default files or `python3 demo.py -h` for details
