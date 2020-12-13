# RNN Scandinavian Text Classifier

Simple RNN short sentence classifier for scandinavian languages: Danish, Norwegian and Swedish:
* Trained on a translated subtitles dataset downloaded from: http://opus.nlpl.eu/OpenSubtitles.php
* Trained on sentences between 15 and 100 characters
* Best model saved when model achieved 90.5% accuracy on validation data after 5 training epochs.
* Implemented with exponential learning rate decay to prevent vanishing/exploding gradients, a common problem in the training of RNNs.
* Developed with PyTorch.
* Integrated on a REST API service.
* Dockerized using docker.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Extracting the dataset

The data has to be downloaded and extracted from the following links: 
* OpenSubtitles.da-en.da (danish) - http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/da-en.txt.zip
* OpenSubtitles.en-no.no (norwegian) - http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-no.txt.zip
* OpenSubtitles.en-sv.sv (swedish)- http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-sv.txt.zip

Each .zip file contains multiple files, make sure to extract only the file ending in the respective language:
* From da-en.txt.zip extract only the file: OpenSubtitles.da-en.da
* From en-no.txt.zip extract only the file: OpenSubtitles.en-no.no
* From en-sv.txt.zip extract only the file: OpenSubtitles.en-sv.sv

These 3 files should be extracted to the folder: datasets/OpenSubs.

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
* --lr_d (learning rate decay percentage to decrease every 10k iterations, default=0.97)
* --e (number of epochs, default=5)
* --pe (print every n iterations, default=100)
* --ds (dataset size of data to extract, default=100000 sentences)
* --dir (directory of OpenSubtitles data, default='datasets/OpenSubs')
* --ckp (bool, use if training from checkpoint, default=False)
* --ckp_dir (if --ckp, directory of saved model, default='save_hn_256_lr_0.005')

Example:
```
python train_rnn.py --nh 128 --lr 0.001 --e 2 --ckp --ckp_dir 'save_hn_128_lr_0.001'
```

(running train_rnn.py will automatically create a save checkpoint directory in /saves if --ckp is used then --ckp_dir will be used to load model. To train model from 0, it is recommended to backup the saved model and delete 'save_hn_XXX_lr_XXXX' file so that a new one can be generated.)

Example of the output during training (using --pe=10 and --ckp):
```
860 0% (4m 20s) | Nej, vet du vad? / sv ✓
Train Accuracy: 87.6% | Validation Accuracy: 88.4%
870 0% (4m 24s) | Mr. Townsend, du kan da være mer rundhåndet enn det. / no ✓
Train Accuracy: 87.7% | Validation Accuracy: 88.3%
880 0% (4m 25s) | Kommer hun her ikke? / da ✓
Train Accuracy: 87.6% | Validation Accuracy: 88.3%
890 0% (4m 28s) | - Jeg så jer sammen. / da ✓
Train Accuracy: 87.5% | Validation Accuracy: 88.2%
900 0% (4m 30s) | Eller vad sa Runeberg, vår nationalskald? / sv ✓
Train Accuracy: 87.7% | Validation Accuracy: 88.2%
910 0% (4m 33s) | Han må være slangemenneske. / no ✗ (da)
Train Accuracy: 87.7% | Validation Accuracy: 88.1%
920 0% (4m 36s) | -Det fins ingen miss Froy. / no ✓
Train Accuracy: 87.7% | Validation Accuracy: 88.2%
```
### Using trained model to generate predictions

To see an example of the model predicting multiple random sentences.
```
python predict_rnn.py --example
```
output
```
Testing on dataset sentences:

> Hold nu op, hun har det skidt
The following sentence is: [da]

> Jeg har akkurat bakt en sukkerkake
The following sentence is: [no]

> Man känner igen den, den är bekväm.
The following sentence is: [sv]

Testing on random sentences from the internet:

> Hej, jeg hedder Pedro og jeg elsker at drikke øl!
The following sentence is: [da]

> Mit luftpudefartøj er fyldt med ål
The following sentence is: [da]

> Der er i øjeblikket ingen tekst på denne side. Du kan søge efter sidenavnet på andre sider, søge i relaterede logger eller oprette siden. 
The following sentence is: [da]

> Jeg forstår det veldig godt.
The following sentence is: [no]

> Floreanaspottefugl er ein sterkt truga art av spottefuglar. Han er naturleg endemisk til øya Floreana, ei av Galápagosøyane.
The following sentence is: [no]

> När katten är borta dansar råttorna på bordet
The following sentence is: [sv]

> Rosshavet (engelska: Ross Sea) är ett randhav av Antarktiska oceanen och ligger mellan Victoria Land och Marie Byrd Land
The following sentence is: [sv]
```
To predict a custom sentence string
```
python predict_rnn.py --string 'Der er i øjeblikket ingen tekst på denne side.'
```
output
```
> Der er i øjeblikket ingen tekst på denne side.
The following sentence is: [da]
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
# Future Work 

* extract more scandivanian datasets and train on more sentences.
* create a web app to integrate the REST API and host it (ex: heroku web app).
* build same model using transformers and compare

## Authors

* **Pedro Rodrigues** (https://github.com/jperod)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This model is based from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
* Useful Flask + Docker tutorial https://runnable.com/docker/python/dockerize-your-flask-application

