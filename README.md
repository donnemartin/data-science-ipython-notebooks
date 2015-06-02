<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/coversmall.png">
  Graphs from <a href="https://github.com/jakevdp/sklearn_pycon2015">PyCon 2015 Scikit-learn Tutorial</a>
</p>

# data-science-ipython-notebooks
Continually updated Data Science IPython Notebooks: Spark, Hadoop MapReduce, HDFS, AWS, Kaggle, scikit-learn, matplotlib, pandas, NumPy, SciPy, Python, and various command lines.

This repo is a collection of IPython Notebooks I reference while working with data.

## Index

* [spark](#spark)
* [mapreduce-python](#mapreduce-python)
* [amazon web services](#aws)
* [kaggle](#kaggle)
* [scikit-learn](#scikit-learn)
* [pandas](#pandas)
* [matplotlib](#matplotlib)
* [numpy](#numpy)
* [scipy](#scipy)
* [python-data](#python-data)
* [command lines](#commands)
* [credits](#credits)
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
| [mapreduce-python](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/mapreduce/mapreduce-python.ipynb) | Supports MapReduce jobs in Python with [mrjob](https://github.com/Yelp/mrjob), running them locally or on Hadoop clusters. Demonstrates mrjob code, unit test, and config file to analyze Amazon S3 bucket logs on Elastic MapReduce.|

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/aws.png">
</p>

## aws

IPython Notebook(s) demonstrating Amazon Web Services (AWS) and AWS tools functionality.

| Notebook | Description |
|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [s3cmd](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3cmd) | Interacts with S3 through the command line. |
| [s3distcp](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3distcp) | Combines smaller files and aggregates them together by taking in a pattern and target file.  S3DistCp can also be used to transfer large volumes of data from S3 to your Hadoop cluster. |
| [s3-parallel-put](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#s3-parallel-put) | Uploads multiple files to S3 in parallel. |
| [redshift](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#redshift) | Acts as a fast data warehouse built on top of technology from massive parallel processing (MPP). |
| [kinesis](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#kinesis) | Streams data in real time with the ability to process thousands of data streams per second. |
| [lambda](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/aws/aws.ipynb#lambda) | Runs code in response to events, automatically managing compute resources. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/kaggle.png">
</p>

## kaggle

IPython Notebook(s) used in [kaggle](https://www.kaggle.com/) competitions.

| Notebook | Description |
|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [titanic](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/kaggle/titanic.ipynb) | Predicts survival on the Titanic.  Demonstrates data cleaning, exploratory data analysis, and machine learning. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scikitlearn.png">
</p>

## scikit-learn

IPython Notebook(s) demonstrating scikit-learn functionality.

Credits: Forked from [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [intro](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb) | Intro notebook to scikit-learn.  Scikit-learn adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |
| [knn](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-intro.ipynb#K-Nearest-Neighbors-Classifier) | K-nearest neighbors. |
| [linear-reg](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-linear-reg.ipynb) | Linear regression. |
| [svm](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-svm.ipynb) | Support vector machine classifier, with and without kernels. |
| [random-forest](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-random-forest.ipynb) | Random forest classifier and regressor. |
| [k-means](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-k-means.ipynb) | K-means clustering. |
| [pca](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-pca.ipynb) | Principal component analysis. |
| [validation](#scikit-learn) | Coming soon. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/pandas.png">
</p>

## pandas

IPython Notebook(s) demonstrating pandas functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [pandas](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/pandas.ipynb) | Software library written for data manipulation and analysis in Python. Offers data structures and operations for manipulating numerical tables and time series. |
| [pandas io](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/pandas_io.ipynb) | Input and output operations. |
| [pandas cleaning](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/pandas/pandas_clean.ipynb) | Data wrangling operations. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/matplotlib.png">
</p>

## matplotlib

IPython Notebook(s) demonstrating matplotlib functionality.

Credits: Some content forked from [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel

| Notebook | Description |
|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [matplotlib](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/matplotlib/matplotlib.ipynb) | Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/numpy.png">
</p>

## numpy

IPython Notebook(s) demonstrating NumPy functionality.

Credits: Forked from [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [numpy](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/numpy/numpy.ipynb) | Adds Python support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays. |

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/data-science-ipython-notebooks/master/images/scipy.png">
</p>

## scipy

[Coming Soon] IPython Notebook(s) demonstrating SciPy functionality.

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
| [jekyll](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/commands/misc.ipynb#jekyll) | Simple, blog-aware, static site generator for personal, project, or organization sites.  Renders Markdown or Textile and Liquid templates, and produces a complete, static website ready to be served by Apache HTTP Server, Nginx or another web server. |

## credits

* [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793) by Wes McKinney
* [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas
* [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
* [Yhat blog](http://blog.yhathq.com/) by Yhat
* [Kaggle](https://www.kaggle.com/) by Kaggle
* [Spark Docs](https://spark.apache.org/docs/latest/) by Apache Spark
* [AWS Docs](http://aws.amazon.com/documentation/) by Amazon Web Services

## license

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
