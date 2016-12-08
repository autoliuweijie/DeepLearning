# Deep Learning with Python

Note: This is my learning project according to Jason Brownlee's "Deep Learning with Python"


## 0. Config Environment

### 0.1 Create Virtualenv

cd into the directory of deep_learning_with_python

```
$ cd PATH/deep_learning_with_python/
```

Create virtualenv

```
$ virtualenv .env
```

### 0.2 Installing Libraries

Activate virtualenv and install some necessary libraries by pip

```
$ source .env/bin/activate
$ pip install -r requirements.txt
```

Wait about 20min for libraries installing, and you can deactivate the virtualenv if you like.

```
$ deactivate
```

### 0.3 Download dataset

cd to dlwp/data_set/, and run the download scrpt, such as:

```
$ cd dlwp/data_set/
$ ./get_pima_indians_diabetes_data.sh
``` 

the dataset will be downloaded to this directory

### 0.4 Start

Now, you can open the jupyter notebook by

```
$ ./start_ipython_notebook.sh 
```

or 

```
$ ipython notebook
```
