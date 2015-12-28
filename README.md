<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/README_1200x800.gif">
</p>

# data-science-ipython-notebooks

This repo is a collection of IPython Notebooks I reference while working with data.  Although I developed and maintain many notebooks, other notebooks I reference were created by various authors, who are credited within their notebook(s) by providing their names and/or a link to their source.

For detailed instructions, scripts, and tools to more optimally set up your development environment for data analysis, check out the [dev-setup](https://github.com/donnemartin/dev-setup) repo.

<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/coversmall_alt.png">
  <br/>
</p>

## Index

* [spark](#spark)
* [mapreduce-python](#mapreduce-python)
* [kaggle-and-business-analyses](#kaggle-and-business-analyses)
* [deep-learning](#deep-learning)
* [scikit-learn](#scikit-learn)
* [statistical-inference-scipy](#statistical-inference-scipy)
* [pandas](#pandas)
* [matplotlib](#matplotlib)
* [numpy](#numpy)
* [python-data](#python-data)
* [amazon web services](#aws)
* [command lines](#commands)
* [misc](#misc)
* [notebook-installation](#notebook-installation)
* [credits](#credits)
* [contributing](#contributing)
* [contact-info](#contact-info)
* [license](#license)

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/spark.png">
</p>

## spark

IPython Notebook(s) demonstrating spark and HDFS functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [spark](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/spark/spark.ipynb) | In-memory cluster computing framework, up to 100 times faster for certain applications and is well suited for machine learning algorithms. |
| [hdfs](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/spark/hdfs.ipynb) | Reliably stores very large files across machines in a large cluster. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/mrjob.png">
</p>

## mapreduce-python

IPython Notebook(s) demonstrating Hadoop MapReduce with mrjob functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [mapreduce-python](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/mapreduce/mapreduce-python.ipynb) | Supports MapReduce jobs in Python with [mrjob](https://github.com/Yelp/mrjob), running them locally or on Hadoop clusters. Demonstrates mrjob code, unit test, and config file to analyze Amazon S3 bucket logs on Elastic MapReduce.  [Disco](https://github.com/discoproject/disco/) is another python-based alternative.|

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/kaggle.png">
</p>

## kaggle-and-business-analyses

IPython Notebook(s) used in [kaggle](https://www.kaggle.com/) competitions and business analyses.

| Notebook | Description |
|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [titanic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/kaggle/titanic.ipynb) | Predicts survival on the Titanic.  Demonstrates data cleaning, exploratory data analysis, and machine learning. |
| [churn-analysis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/analyses/churn.ipynb) | Predicts customer churn.  Exercises logistic regression, gradient boosting classifers, support vector machines, random forests, and k-nearest-neighbors.  Discussion of confusion matrices, ROC plots, feature importances, prediction probabilities, and calibration/descrimination.|

<br/>
<p align="center">
  <img src="http://i.imgur.com/ZhKXrKZ.png">
</p>

## deep-learning

IPython Notebook(s) demonstrating deep learning functionality.

<br/>
<p align="center">
  <img src="https://avatars0.githubusercontent.com/u/15658638?v=3&s=100">
</p>

### tensor-flow-tutorials

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [tsf-basics](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/1_intro/basic_operations.ipynb) | Learn basic operations in TensorFlow, a library for various kinds of perceptual and language understanding tasks from Google. |
| [tsf-linear](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/2_basic_classifiers/linear_regression.ipynb) | Implement linear regression in TensorFlow. |
| [tsf-logistic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/2_basic_classifiers/logistic_regression.ipynb) | Implement logistic regression in TensorFlow. |
| [tsf-nn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/2_basic_classifiers/nearest_neighbor.ipynb) | Implement nearest neighboars in TensorFlow. |
| [tsf-alex](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/3_neural_networks/alexnet.ipynb) | Implement AlexNet in TensorFlow. |
| [tsf-cnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/3_neural_networks/convolutional_network.ipynb) | Implement convolutional neural networks in TensorFlow. |
| [tsf-mlp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/3_neural_networks/multilayer_perceptron.ipynb) | Implement multilayer perceptrons in TensorFlow. |
| [tsf-rnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/3_neural_networks/recurrent_network.ipynb) | Implement recurrent neural networks in TensorFlow. |

### tensor-flow-exercises

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [tsf-not-mnist](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/1_notmnist.ipynb) | Learn simple data curation by creating a pickle with formatted datasets for training, development and testing in TensorFlow. |
| [tsf-fully-connected](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/2_fullyconnected.ipynb) | Progressively train deeper and more accurate models using logistic regression and neural networks in TensorFlow. |
| [tsf-regularization](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/3_regularization.ipynb) | Explore regularization techniques by training fully connected networks to classify notMNIST characters in TensorFlow. |
| [tsf-convolutions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/4_convolutions.ipynb) | Create convolutional neural networks in TensorFlow. |
| [tsf-word2vec](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/5_word2vec.ipynb) | Train a skip-gram model over Text8 data in TensorFlow. |
| [tsf-lstm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/6_lstm.ipynb) | Train a LSTM character model over Text8 data in TensorFlow. |

<br/>
<p align="center">
  <img src="http://www.deeplearning.net/software/theano/_static/theano_logo_allblue_200x46.png">
</p>

### theano-tutorials

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [theano-intro](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/intro_theano/intro_theano.ipynb) | Intro to Theano, which allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It can use GPUs and perform efficient symbolic differentiation. |
| [theano-scan](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/scan_tutorial/scan_tutorial.ipynb) | Learn scans, a mechanism to perform loops in a Theano graph. |
| [theano-logistic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/intro_theano/logistic_regression.ipynb) | Implement logistic regression in Theano. |
| [theano-rnn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/rnn_tutorial/simple_rnn.ipynb) | Implement recurrent neural networks in Theano. |
| [theano-mlp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/theano-tutorial/theano_mlp/theano_mlp.ipynb) | Implement multilayer perceptrons in Theano. |

### deep-learning-misc

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [deep-dream](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/deep-dream/dream.ipynb) | Caffe-based computer vision program which uses a convolutional neural network to find and enhance patterns in images. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scikitlearn.png">
</p>

## scikit-learn

IPython Notebook(s) demonstrating scikit-learn functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [intro](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb) | Intro notebook to scikit-learn.  Scikit-learn adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |
| [knn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb#K-Nearest-Neighbors-Classifier) | K-nearest neighbors. |
| [linear-reg](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-linear-reg.ipynb) | Linear regression. |
| [svm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-svm.ipynb) | Support vector machine classifier, with and without kernels. |
| [random-forest](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-random-forest.ipynb) | Random forest classifier and regressor. |
| [k-means](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-k-means.ipynb) | K-means clustering. |
| [pca](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-pca.ipynb) | Principal component analysis. |
| [gmm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-gmm.ipynb) | Gaussian mixture models. |
| [validation](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-validation.ipynb) | Validation and model selection. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scipy.png">
</p>

## statistical-inference-scipy

IPython Notebook(s) demonstrating statistical inference with SciPy functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scipy | SciPy is a collection of mathematical algorithms and convenience functions built on the Numpy extension of Python. It adds significant power to the interactive Python session by providing the user with high-level commands and classes for manipulating and visualizing data. |
| [effect-size](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/effect_size.ipynb) | Effect size. |
| [sampling](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/sampling.ipynb) | Random sampling. |
| [hypothesis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scipy/hypothesis.ipynb) | Hypothesis testing. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/pandas.png">
</p>

## pandas

IPython Notebook(s) demonstrating pandas functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [pandas](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/pandas.ipynb) | Software library written for data manipulation and analysis in Python. Offers data structures and operations for manipulating numerical tables and time series. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/matplotlib.png">
</p>

## matplotlib

IPython Notebook(s) demonstrating matplotlib functionality.

| Notebook | Description |
|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [matplotlib](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/matplotlib.ipynb) | Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. |
| [matplotlib-applied](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/matplotlib-applied.ipynb) | Matplotlib visualizations appied to Kaggle competitions for exploratory data analysis.  Examples of bar plots, histograms, subplot2grid, normalized plots, scatter plots, subplots, and kernel density estimation plots. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/numpy.png">
</p>

## numpy

IPython Notebook(s) demonstrating NumPy functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [numpy](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/numpy.ipynb) | Adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/python.png">
</p>

## python-data

IPython Notebook(s) demonstrating Python functionality geared towards data analysis.

| Notebook | Description |
|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [data structures](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/structs.ipynb) | Tuples, lists, dicts, sets. |
| [data structure utilities](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/structs_utils.ipynb) | Slice, range, xrange, bisect, sort, sorted, reversed, enumerate, zip, list comprehensions. |
| [functions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/functions.ipynb) | Functions as objects, lambda functions, closures, *args, **kwargs currying, generators, generator expressions, itertools. |
| [datetime](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/datetime.ipynb) | Datetime, strftime, strptime, timedelta. |
| [logging](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/logs.ipynb) | Logging with RotatingFileHandler and TimedRotatingFileHandler. |
| [pdb](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/pdb.ipynb) | Interactive source code debugger. |
| [unit tests](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/python-data/unit_tests.ipynb) | Nose unit tests. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/aws.png">
</p>

## aws

IPython Notebook(s) demonstrating Amazon Web Services (AWS) and AWS tools functionality.

Also check out:

* [SAWS](https://github.com/donnemartin/saws): A Supercharged AWS command line interface (CLI).
* [Awesome AWS](https://github.com/donnemartin/awesome-aws): A curated list of libraries, open source repos, guides, blogs, and other resources.

| Notebook | Description |
|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [boto](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#Boto) | Official AWS SDK for Python. |
| [s3cmd](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3cmd) | Interacts with S3 through the command line. |
| [s3distcp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3distcp) | Combines smaller files and aggregates them together by taking in a pattern and target file.  S3DistCp can also be used to transfer large volumes of data from S3 to your Hadoop cluster. |
| [s3-parallel-put](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3-parallel-put) | Uploads multiple files to S3 in parallel. |
| [redshift](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#redshift) | Acts as a fast data warehouse built on top of technology from massive parallel processing (MPP). |
| [kinesis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#kinesis) | Streams data in real time with the ability to process thousands of data streams per second. |
| [lambda](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#lambda) | Runs code in response to events, automatically managing compute resources. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/commands.png">
</p>

## commands

IPython Notebook(s) demonstrating various command lines for Linux, Git, etc.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [linux](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/linux.ipynb) | Unix-like and mostly POSIX-compliant computer operating system.  Disk usage, splitting files, grep, sed, curl, viewing running processes, terminal syntax highlighting, and Vim.|
| [anaconda](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#anaconda) | Distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. |
| [ipython notebook](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#ipython-notebook) | Web-based interactive computational environment where you can combine code execution, text, mathematics, plots and rich media into a single document. |
| [git](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#git) | Distributed revision control system with an emphasis on speed, data integrity, and support for distributed, non-linear workflows. |
| [ruby](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#ruby) | Used to interact with the AWS command line and for Jekyll, a blog framework that can be hosted on GitHub Pages. |
| [jekyll](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#jekyll) | Simple, blog-aware, static site generator for personal, project, or organization sites.  Renders Markdown or Textile and Liquid templates, and produces a complete, static website ready to be served by Apache HTTP Server, Nginx or another web server.  [Pelican](https://github.com/getpelican/pelican) is a python-based alternative. |
| [django](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#django) | High-level Python Web framework that encourages rapid development and clean, pragmatic design. It can be useful to share reports/analyses and for blogging. Lighter-weight alternatives include [Pyramid](https://github.com/Pylons/pyramid), [Flask](https://github.com/mitsuhiko/flask), [Tornado](https://github.com/tornadoweb/tornado), and [Bottle](https://github.com/bottlepy/bottle).

## misc

IPython Notebook(s) demonstrating miscellaneous functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [regex](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/misc/regex.ipynb) | Regular expression cheat sheet useful in data wrangling.|

## notebook-installation

### anaconda

Anaconda is a free distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing that aims to simplify package management and deployment.

Follow instructions to install [Anaconda](http://docs.continuum.io/anaconda/install.html) or the more lightweight [miniconda](http://conda.pydata.org/miniconda.html).

### pip-requirements

If you prefer to use a more lightweight installation procedure than Anaconda, first clone the repo then run the following pip command on the provided requirements.txt file:

    $ pip install -r requirements.txt

### running-notebooks

To view interactive content or to modify elements within the IPython notebooks, you must first clone or download the repository then run the ipython notebook.  More information on IPython Notebooks can be found [here.](http://ipython.org/notebook.html)

```
$ git clone https://github.com/donnemartin/data-science-ipython-notebooks.git
$ cd data-science-ipython-notebooks
$ ipython notebook
```

Notebooks tested with Python 2.7.x.

## credits

* [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793) by Wes McKinney
* [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas
* [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
* [Statistical Interference Using Computational Methods in Python](https://github.com/AllenDowney/CompStats) by Allen Downey
* [Yhat Blog](http://blog.yhathq.com/) by Yhat
* [Kaggle](https://www.kaggle.com/) by Kaggle
* [Spark Docs](https://spark.apache.org/docs/latest/) by Apache Spark
* [AWS Docs](http://aws.amazon.com/documentation/) by Amazon Web Services

## contributing

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/donnemartin/data-science-ipython-notebooks/issues).

## contact-info

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [donne.martin@gmail.com](mailto:donne.martin@gmail.com)
* Twitter: [@donne_martin](https://twitter.com/donne_martin)
* GitHub: [donnemartin](https://github.com/donnemartin)
* LinkedIn: [donnemartin](https://www.linkedin.com/in/donnemartin)
* Website: [donnemartin.com](http://donnemartin.com)

## license

This repository contains a variety of content; some developed by Donne Martin, and some from third-parties.  The third-party content is distributed under the license provided by those parties.

The content developed by Donne Martin is distributed under the following license:

    Copyright 2015 Donne Martin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
