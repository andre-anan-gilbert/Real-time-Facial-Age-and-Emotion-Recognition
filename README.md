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
- [fer2013.csv](https://www.kaggle.com/deadskull7/fer2013) FER2013 with the images encoded in a CSV
- [labels](https://github.com/microsoft/FERPlus/blob/master/fer2013new.csv):
  FER+ provides a set of new better quality ground truth labels for FER2013.

## Requirements

- Python >= 3.9
- See [requirements.txt](requirements.txt).

To install the requirements:

```
pip install -r requirements.txt
```

These requirements are only used for the app as well as the notebooks [age_classifier.ipynb](/src/notebooks/age_classifier.ipynb) and [emotion_classifier.ipynb](/src/notebooks/emotion_classifier.ipynb). Since SHAP is not working with tensorflow >= 2.6, other requirements are used for [visualization.ipynb](/src/notebooks/visualization.ipynb).

We do not recommend running [visualization.ipynb](/src/notebooks/visualization.ipynb), but if you have to do it, use the following instructions:

```
pip install -r requirements_visualization.txt
```

The installation of aggdraw (which is a dependency of visualkeras) might fail, please refer to the [repository of visualkeras](https://github.com/paulgavrikov/visualkeras#installing-aggdraw-fails) for a solution on linux.

On windows, the problem can be solved by installing [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). From the Visual Studio installer, please install Visual Studio Buildtools and the Desktop Development with C++ workload.

On Mac it should work similarly to Windows, although this is untested because of budget reasons.

Since visualkeras is only relevant for the model architecture diagrams in the [visualization.ipynb](/src/notebooks/visualization.ipynb) another option would be to install all packages except aggdraw and visualkeras:

```
pip install $(grep -ivE "aggdraw|visualkeras" requirements_visualization.txt)
```

This allows running the [visualization.ipynb](/src/notebooks/visualization.ipynb)-Notebook (for example for SHAP) by removing the following lines of code:

```
import visualkeras

visualkeras.layered_view(emotion_classifier,
                         to_file='../../docs/emotion_classifier.png',
                         legend=True,
                         font=font,
                         color_map=color_map)

visualkeras.layered_view(age_classifier,
                         to_file='../../docs/age_classifier.png',
                         legend=True,
                         font=font,
                         color_map=color_map)
```

## Running the application

```
python src/app.py
```

## Notebooks

- [age_classifier.ipynb](/src/notebooks/age_classifier.ipynb) contains data preparation as well as implementation, training and performance evaluation of the Age Group Classifier
- [emotion_classifier.ipynb](/src/notebooks/emotion_classifier.ipynb) contains data preparation as well as implementation, training and performance evaluation of the Emotion Classifier
- [visualization.ipynb](/src/notebooks/visualization.ipynb) contains visualization of model architecture with visualkeras and explainable AI with SHAP for both models

## License

This repository is released under the
[MIT license](https://opensource.org/licenses/MIT).
In short, this means you are free to use this software in any personal, open-source or -commercial projects. Attribution is optional but appreciated.
