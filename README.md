# Intelligent Systems for Smart Health
Materials and tests to support the course at Düsseldorf University of Applied Sciences (HSD).


## Create new environment for this course (recommended)
It is recommended to create a new environment for this course with many Python libraries that we will use in the Live Coding sessions. You can simply download the `environment.yml` file in this repository, or clone the repository using:
```
git clone https://github.com/florian-huber/ai_for_smart_health.git
```
Then, in the folder with the `environment.yml` file simply run:
```
conda update -n base -c defaults conda  # optional (to make sure conda is up to date)
conda env create -f environment.yml
```
This should create a Python 3.10 environment with the packages listed in the yaml-file.

## Hardware & performance
This course will include a lot of machine learning, mostly also deep learning. For this, we will work with the Python libraries scikit-learn (classical machine learning) and Tensorflow (deep learning).

Scikit-learn can be accelerated using scikit-learn-intelex (https://intel.github.io/scikit-learn-intelex), although this will not be needed for the live coding sessions we do.

Tensorflow can make use of NVIDIA graphic cards, which can greatly enhance deep learning model training. While this allows us to use a lot of low-level optimization, it can be tricky and time consuming to get this up and running.
