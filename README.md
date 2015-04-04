<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/coversmall.png">
</p>

# ipython-data-notebooks
Continually updated IPython Data Science Notebooks geared towards processing big data (AWS, Spark, Hadoop MapReduce, HDFS, Linux command line, Python, NumPy, pandas, matplotlib, SciPy, scikit-learn, Kaggle).

## kaggle

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/kaggle.png">
</p>

IPython Notebooks used in [kaggle](https://www.kaggle.com/) competitions.

| Notebook | Description |
|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [titanic](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/kaggle/titanic.ipynb) | Predicts survival on the Titanic.  Demonstrates data cleaning, exploratory data analysis, and machine learning. |

## spark

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/spark.png">
</p>

IPython Notebooks demonstrating spark and HDFS functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [spark](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/spark/spark.ipynb) | In-memory cluster computing framework, up to 100 times faster for certain applications and is well suited for machine learning algorithms. |
| [hdfs](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/spark/hdfs.ipynb) | Reliably stores very large files across machines in a large cluster. |

## aws

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/aws.png">
</p>

IPython Notebooks demonstrating Amazon Web Services functionality.

| Notebook | Description |
|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [s3cmd](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#s3cmd) | Interacts with S3 through the command line. |
| [s3-parallel-put](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#s3-parallel-put) | Uploads multiple files to S3 in parallel. |
| [s3distcp](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#s3distcp) | Combines smaller files and aggregates them together by taking in a pattern and target file.  S3DistCp can also be used to transfer large volumes of data from S3 to your Hadoop cluster. |
| [mrjob](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#mrjob) | Supports MapReduce jobs in Python 2.5+ and runs them locally or on Hadoop clusters. |
| [redshift](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#redshift) | Acts as a fast data warehouse built on top of technology from massive parallel processing (MPP). |
| [kinesis](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#kinesis) | Streams data in real time with the ability to process thousands of data streams per second. |
| [lambda](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/aws/aws.ipynb#lambda) | Runs code in response to events, automatically managing compute resources. |

## python-core

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/python.png">
</p>

IPython Notebooks demonstrating core Python functionality geared towards data analysis.

| Notebook | Description |
|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [data structures](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/python-core/structs.ipynb) | Tuples, lists, dicts, sets. |
| [data structure utilities](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/python-core/structs_utils.ipynb) | Slice, range, xrange, bisect, sort, sorted, reversed, enumerate, zip, list comprehensions. |
| [functions](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/python-core/functions.ipynb) | Functions as objects, lambda functions, closures, *args, **kwargs currying, generators, generator expressions, itertools. |
| [datetime](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/python-core/datetime.ipynb) | Datetime, strftime, strptime, timedelta. |
| [unit tests](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/python-core/unit_tests.ipynb) | Nose unit tests. |

## pandas

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/pandas.png">
</p>

IPython Notebooks demonstrating pandas functionality.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [pandas](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/pandas/pandas.ipynb) | Software library written for data manipulation and analysis in Python. Offers data structures and operations for manipulating numerical tables and time series. |
| [pandas io](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/pandas/pandas_io.ipynb) | Input and output operations. |
| [pandas cleaning](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/pandas/pandas_clean.ipynb) | Data wrangling operations. |

## commands

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/commands.png">
</p>

IPython Notebooks demonstrating various command lines for Linux, Git, etc.

| Notebook | Description |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [linux](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/linux.ipynb) | Unix-like and mostly POSIX-compliant computer operating system. |
| [anaconda](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/misc.ipynb#anaconda) | Distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment. |
| [ipython notebook](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/misc.ipynb#ipython-notebook) | Web-based interactive computational environment where you can combine code execution, text, mathematics, plots and rich media into a single document. |
| [git](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/misc.ipynb#git) | Distributed revision control system with an emphasis on speed, data integrity, and support for distributed, non-linear workflows. |
| [ruby](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/misc.ipynb#ruby) | Used to interact with the AWS command line and for Jekyll, a blog framework that can be hosted on GitHub Pages. |
| [jekyll](http://nbviewer.ipython.org/github/donnemartin/ipython-data-notebooks/blob/master/commands/misc.ipynb#jekyll) | Simple, blog-aware, static site generator for personal, project, or organization sites.  Renders Markdown or Textile and Liquid templates, and produces a complete, static website ready to be served by Apache HTTP Server, Nginx or another web server. |

## matplotlib

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/matplotlib.png">
</p>

[Coming Soon] IPython Notebooks demonstrating matplotlib functionality.

## scikit-learn

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/scikitlearn.png">
</p>

[Coming Soon] IPython Notebooks demonstrating scikit-learn functionality.

## scipy

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/scipy.png">
</p>

[Coming Soon] IPython Notebooks demonstrating SciPy functionality.

## numpy

<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/donnemartin/ipython-data-notebooks/master/images/numpy.png">
</p>

[Coming Soon] IPython Notebooks demonstrating NumPy functionality.

## License

    Copyright 2014 Donne Martin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
