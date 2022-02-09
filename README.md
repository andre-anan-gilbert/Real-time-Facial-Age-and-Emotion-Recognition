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
- [fer2013.csv](https://www.kaggle.com/deadskull7/fer2013): FER2013 with the images encoded in a CSV.
- [labels](https://github.com/microsoft/FERPlus/blob/master/fer2013new.csv):
  FER+ provides a set of new better quality ground truth labels for FER2013.

## Requirements

- Python 3.9
- See [requirements](requirements.txt)

To install the requirements:

```
pip install -r requirements.txt
```

These requirements are used for the app, the following notebooks:
- [age_classifier.ipynb](/src/notebooks/age_classifier.ipynb)
- [emotion_classifier.ipynb](/src/notebooks/emotion_classifier.ipynb)

which run using tensorflow 2.7, since they use the Rescaling layer, which was introduced in tensorflow 2.6.

Since [SHAP](https://github.com/slundberg/shap) doesn't work with tensorflow >= 2.6, and [visualkeras](https://github.com/paulgavrikov/visualkeras) requires additional Software, other requirements are used for [visualization.ipynb](/src/notebooks/visualization.ipynb). The Visualization notebook is only included for documentation purposes.

For this reason, we do not recommend running [visualization.ipynb](/src/notebooks/visualization.ipynb), but if you want to, use the following instructions:

```
pip install -r requirements_visualization.txt
```

In case the installation of aggdraw (which is a dependency of visualkeras) fails, please refer to the [repository of visualkeras](https://github.com/paulgavrikov/visualkeras#installing-aggdraw-fails) for a solution on linux.

On windows, the problem can be solved by installing [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). From the Visual Studio installer, please install Visual Studio Buildtools and the Desktop Development with C++ workload.

On Mac it should work similarly to Windows (untested).

Since visualkeras is only relevant for the model architecture diagrams in the [visualization.ipynb](/src/notebooks/visualization.ipynb), another option would be to install all packages except aggdraw and visualkeras:

```
pip install $(grep -ivE "aggdraw|visualkeras" requirements_visualization.txt)
```

This allows running the [visualization.ipynb](/src/notebooks/visualization.ipynb)-Notebook (for example for SHAP) by removing the following lines of code:

```
# cell 1
import visualkeras

# cell 9
visualkeras.layered_view(emotion_classifier,
                         to_file='../../docs/model_architecture/emotion_classifier.png',
                         legend=True,
                         font=font,
                         color_map=color_map)

# cell 17
visualkeras.layered_view(age_classifier,
                         to_file='../../docs/model_architecture/age_classifier.png',
                         legend=True,
                         font=font,
                         color_map=color_map)
```

## Running the Application

To run the application:
```
python src/app.py
```

To stop the application:
```
press q
```

## Notebooks

- [age_classifier.ipynb](/src/notebooks/age_classifier.ipynb) contains the data preparation, implementation, training and performance evaluation of the Age Group Classifier
- [emotion_classifier.ipynb](/src/notebooks/emotion_classifier.ipynb) contains the data preparation, implementation, training and performance evaluation of the Emotion Classifier
- [visualization.ipynb](/src/notebooks/visualization.ipynb) contains the visualization of the model architecture with visualkeras and explainable AI with SHAP

## License

This repository is released under the
[MIT license](https://opensource.org/licenses/MIT).
In short, this means you are free to use this software in any personal, open-source or -commercial projects. Attribution is optional but appreciated.
