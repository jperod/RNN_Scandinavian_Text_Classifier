# RNN_Scandinavian_Text_Classifier

Simple RNN short sentence classifier for scandinavian languages: Danish, Norwegian and Swedish:
* Trained on a translated subtitles dataset downloaded from: http://opus.nlpl.eu/OpenSubtitles.php
* Achieves 90.2% accuracy on validation data after ~5 training epochs.
* Implemented exponential learning rate decay to prevent vanishing/exploding gradients during training.
* Developed with PyTorch.
* Integrated on a REST API service.
* Dockerized using docker.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Extracting the data

* The data has already been downloaded from http://opus.nlpl.eu/OpenSubtitles.php and compressed.
* Extract datasets/OpenSubs.rar
* A folder is extracted with the following files: os_da.txt (danish), os_no.txt (norwegian) and sv.txt (swedish).

### Installing the environment

* 1. Clone this repository
```
git clone https://github.com/jperod/RNN_Scandinavian_Text_Classifier.git
```
* 2. install virtualenv 
```
pip install virtualenv
```
* 3. Create a python virtualenv
```
virtualenv venv
```
* 4.i (Windows) Activate virtual environment
```
cd venv\Scripts
activate
cd ..\..
```
* 4.ii (Linux / Mac) Activate virtual environment
```
source venv/bin/activate
```
* 5 Install required libraries
```
pip install -r requirements.txt
```

### Training the model

* 
* (running train_rnn.py will automatically create a save checkpoint directory in /saves if --ckp is used then --ckp_dir will be used to load model. To train model from 0, it is recommended to backup the saved model and delete 'save_hn_XXX_lr_XXXX' file so that a new one can be generated.)

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## REST API

### REST API Response:
```
{"class_id": 1, "class_name": "Danish"}
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Pedro Rodrigues** (https://github.com/jperod)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

