# Copyright 2021 The Emotion Recognition Authors. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
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
        TABLE: Dictionary containing the emotions.
    """
    TABLE = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Surprised',
        3: 'Sad',
        4: 'Angry',
        5: 'Disgusted',
        6: 'Fearful',
    }


class Products:
    """Class for the product recommendation.

    Attributes:
        TABLE: Dictionary containing the products.
    """
    TABLE = {
        '(Child, Happy)': 'Pocket Money Account',
        '(Child, Neutral)': 'Youth Current Account',
        '(Child, Surprised)': 'Youth Current Account',
        '(Child, Sad)': 'Savings Account',
        '(Child, Angry)': 'Pocket Money Account',
        '(Child, Disgusted)': 'Savings Account',
        '(Child, Fearful)': 'Savings Account',
        '(Young Adult, Happy)': 'Depot',
        '(Young Adult, Neutral)': 'Current Account',
        '(Young Adult, Surprised)': 'Depot',
        '(Young Adult, Sad)': 'Savings Account',
        '(Young Adult, Angry)': 'Fixed Deposit Account',
        '(Young Adult, Disgusted)': 'Savings Account',
        '(Young Adult, Fearful)': 'Captial Accumulation Benefits',
        '(Adult, Happy)': 'Mortgage Loan',
        '(Adult, Neutral)': 'Current Account',
        '(Adult, Surprised)': 'Loan',
        '(Adult, Sad)': 'No recommendation.',
        '(Adult, Angry)': 'Other Insurance',
        '(Adult, Disgusted)': 'Current Account',
        '(Adult, Fearful)': 'Overnight Money Account',
        '(Senior, Happy)': 'Credit Card',
        '(Senior, Neutral)': 'Other Insurance',
        '(Senior, Surprised)': 'Credit Card',
        '(Senior, Sad)': 'No recommendation.',
        '(Senior, Angry)': 'No recommendation.',
        '(Senior, Disgusted)': 'No recommendation.',
        '(Senior, Fearful)': 'Life Insurance',
    }
