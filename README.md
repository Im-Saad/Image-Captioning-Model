This project implements an image captioning model using a Transformer-based architecture and the InceptionV3 CNN encoder. 
It generates captions for images by extracting features using a pre-trained CNN and decoding them using a Transformer decoder. 
The dataset used for training is the COCO 2017 captions dataset.

Download the Dataset:
   - Download COCO 2017 dataset from [COCO website](https://cocodataset.org/).
   - Store images in the `train2017` folder and annotations in the `annotations` folder under a parent directory called `data`.

Preprocess and Train the Model:
   - Run `train.py` to preprocess the dataset, split it into training and validation sets, and train the image captioning model.
   - Trained model weights are saved as `image_captioning_model.h5` after training.
   # python train.py

Generate Captions for New Images:
   -After training, use `inference.py` to generate captions for new images.
   # python inference.py <path_to_image>

File Descriptions

dataset_setup.py:
   - Prepares the dataset by loading the COCO annotations, processing captions, and splitting data into training and validation sets.

model_setup.py:
   - Defines the model architecture:
     - CNN_Encoder: Extracts image features using InceptionV3 without the fully connected layers.
     - TransformerEncoderLayer: Self-attention and feed-forward layers for encoding the image features.
     - TransformerDecoderLayer: Multi-head attention, embedding, and dense layers for decoding captions.
     - ImageCaptioningModel: Combines the CNN and Transformer layers, with custom training and testing steps.

train.py:
   - Compiles and trains the image captioning model using the training set.

inference.py:
   - Loads the trained model and generates captions for a new image by feeding it through the CNN encoder and Transformer decoder.
   - 
