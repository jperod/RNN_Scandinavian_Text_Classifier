# RNN Scandinavian Text Classifier

Simple RNN short sentence classifier for scandinavian languages: Danish, Norwegian and Swedish:
* Trained on a translated subtitles dataset downloaded from: http://opus.nlpl.eu/OpenSubtitles.php
* Trained on sentences between 15 and 100 characters
* Best saved model achieves ~90% accuracy on validation data after ~5 training epochs.
* Implemented with low but not too low learning rate of 0.005 to prevent vanishing/exploding gradients, a common problem in the training of RNNs.
* Developed with PyTorch.
* Integrated on a REST API service.
* Dockerized using docker.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Extracting the compressed dataset

The data has to be downloaded and extracted from the following links: 
* OpenSubtitles.da-en.da (danish) - http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/da-en.txt.zip
* OpenSubtitles.en-no.no (norwegian) - http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-no.txt.zip
* OpenSubtitles.en-sv.sv (swedish)- http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-sv.txt.zip

Each .zip file contains multiple files, make sure to extract only the file ending in the respective language:
* From da-en.txt.zip extract only the file: OpenSubtitles.da-en.da
* From en-no.txt.zip extract only the file: OpenSubtitles.en-no.no
* From en-sv.txt.zip extract only the file: OpenSubtitles.en-sv.sv

The folder datasets/OpenSubs should contain these 3 files

### Installation

1. Clone this repository
```
git clone https://github.com/jperod/RNN_Scandinavian_Text_Classifier.git
```
2. install virtualenv 
```
pip install virtualenv
```
3. Create a python virtualenv
```
virtualenv venv
```
4.i (Windows) Activate virtual environment
```
cd venv\Scripts
activate
cd ..\..
```
4.ii (Linux / Mac) Activate virtual environment
```
source venv/bin/activate
```
5 Install required libraries
```
pip install -r requirements.txt
```

### Training the model

To train using the same configuration as mine (default values)
```
python train_rnn.py
```
To train on different configuration you can modify the following optional arguments:
* --nh (number of hidden units, default=256)
* --lr (learning rate, default=0.005)
* --e (number of epochs, default=5)
* --pe (print every n iterations, default=1000)
* --ds (dataset size of data to extract, default=100000 sentences)
* --dir (directory of OpenSubtitles data, default='datasets/OpenSubs')
* --ckp (bool, use if training from checkpoint, default=False)
* --ckp_dir (if --ckp, directory of saved model, default='save_hn_256_lr_0.005')

Example:
```
python train_rnn.py --nh 128 --lr 0.001 --e 2 --ckp --ckp_dir 'save_hn_128_lr_0.001'
```

(running train_rnn.py will automatically create a save checkpoint directory in /saves if --ckp is used then --ckp_dir will be used to load model. To train model from 0, it is recommended to backup the saved model and delete 'save_hn_XXX_lr_XXXX' file so that a new one can be generated.)

Example of the output during training:
```
XXXXX X% (35m 48s) | Står du der, jenta mi? / da ✗ (no)
Train Accuracy: 55.1% | Validation Accuracy: 54.9%


```
### Using trained model to generate predictions

To see an example of the model predicting multiple random sentences
```
python predict_rnn.py --example
```
To predict a custom sentence string
```
python predict_rnn.py --string 'Der er i øjeblikket ingen tekst på denne side.'
```

## REST API on Docker

### Dockerizing the Flask app service with Docker

Building the container
```
docker build -t stc .
```
After the build completes, run the container
```
docker run -d -p 5000:5000 stc
```

### To run the REST API on the built container


To serve a prediction using REST API do:
```
http://http://0.0.0.0:5000/predict/<sentence>
```
Example:
```
http://http://0.0.0.0:5000/predict/När katten är borta dansar råttorna på bordet
```
Output in json format:
```
{"language":"sv"}
```

## Authors

* **Pedro Rodrigues** (https://github.com/jperod)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This model is based from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
* Useful Flask + Docker tutorial https://runnable.com/docker/python/dockerize-your-flask-application

