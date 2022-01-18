# Real-time Facial Age and Emotion Recognition

This repository's goal is to demonstrate how to classify the age group
(Child, Young Adult, Adult, Senior) and the emotion (Neutral, Happy, Surprised,
Sad, Angry, Disgusted, Fearful) of a human, using convolutional neural networks (CNN).
## Datasets

Age Group Recognition:
- [facial age](https://www.kaggle.com/frabbisw/facial-age):
An image dataset consisting human faces with ages.

Emotion Recognition:
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data):
48 x 48 grayscale images of faces that are categorized in to one of the seven classes mentioned above.
- [labels](https://github.com/microsoft/FERPlus/blob/master/fer2013new.csv):
FER+ provides a set of new better quality ground truth labels for FER2013.
## Requirements

- Python >= 3.9
- See [requirements.txt](requirements.txt).

To install the requirements:
```
pip install -r requirements.txt
```

## Running the application
```
python src/app.py
```

### Docker
To build the image:
```
docker build -t name .
```

To run the container:
```
docker run -dp 3000:3000 name
```


## License

This repository is released under the
[MIT license](https://opensource.org/licenses/MIT).
In short, this means you are free to use this software in any personal, open-source or -commercial projects. Attribution is optional but appreciated.
