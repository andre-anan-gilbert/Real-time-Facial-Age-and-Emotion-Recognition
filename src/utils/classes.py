"""Classes which can be predicted by the deep learning algorithms."""


class AgeGroups:
    """Class for the age classifier.

    Attributes:
        TABLE: Dictionary containing the age groups.
    """
    TABLE = {
        0: 'Child',
        1: 'Young Adult',
        2: 'Adult',
        3: 'Senior',
    }


class Emotions:
    """Class for the emotion classifier.

    Attributes:
        LIST: List containing the emotions.
    """
    LIST = [
        'neutral',
        'happy',
        'surprised',
        'sad',
        'angry',
        'disgusted',
        'fearful',
    ]


class Products:
    """Class for the product recommendation.

    Attributes:
        TABLE: Dictionary containing the products.
    """
    TABLE = {
        '(Child, Happy)': '...',
        '(Child, Neutral)': '...',
    }
