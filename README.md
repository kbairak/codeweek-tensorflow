# Material for [Tensorflow][Tensorflow] presentation for Codeweek 2018

## Prerequisites

### Suitable python version

We need to find which python version you have intalled in your computer. For
Tensorflow to run you need Python version 2.7 or any Python 3 version between
3.0 and 3.6. ie 2.6 or 3.7 will not do. Open a console and try this:

```sh
$ python -V
Python 3.7.0
```

If you get 3.7, there is still a chance you have more than one versions
intalled. Write `python` and press Tab twice to enable auto completion:

```sh
$ python
python                   python3.4-config         python3.7
python2                  python3.4m               python3.7-config
python2.7                python3.4m-config        python3.7m
python2.7-config         python3.6                python3.7m-config
python2-config           python3.6-config         python3-config
python3                  python3.6m               python-config
python3.4                python3.6m-config        PYTHONDONTWRITEBYTECODE
```

If you don't have an suitable version or don't have Python at all, follow your
operating system's or distribution's intructions to get it or visit
[Python-downloads][Python-downloads]. After that, do the checks again.


### Virtual Environment

A virtual environment allows you to isolate the dependencies of a Python
project inside a, well, virtual environment. Python version 3.3 and above comes
with the `venv` module by default. Otherwise, you have to download the
[virtualenv][virtualenv] package for your operating system/distribution.


#### Check if you can create virtual environments

Try this:

```sh
$ python -m venv
```

If you get

```
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip] [--prompt PROMPT]
                        ENV_DIR [ENV_DIR ...]
venv: error: the following arguments are required: ENV_DIR
```

it means you are good to go. If you get

```
/usr/bin/python3.2: No module named venv
```

it means you need to install virtualenv. In that case, try

```sh
$ virtualenv
```

and see if the command is found or not.


#### Specifying a Python version

When inside the virtual environment, the `python` executable points to the
Python version use assigned to the virtual environment. So, if you have
multiple versions intalled and you don't want to use the latest, you have to
specify the one you want by hand.

Lets assume we have 3.6 and 3.7 installed and we want to create a virtual
environment with 3.6. With the `venv` module, we have to do:

```sh
$ python3.6 -m venv codeweek_venv
```

With 'virtualenv':

```sh
$ virtualenv -p python3.2 codeweek_venv
```


#### Create and activate the virtual environment

With venv:

```sh
$ python3.6 -m venv codeweek_venv
$ source codeweek_venv/bin/activate
```

With virtualenv:

```sh
$ virtualenv -p python3.6 codeweek_venv
$ source codeweek_venv/bin/activate
```

Check that you are using the virtual environment's python version with:

```sh
$ which python
<path-to-current-folder>/codeweek_venv/bin/python

$ python -V
Python 3.6.6
```

To exit the virtual environment, run:

```sh
$ deactivate
```

When inside a virtual environment, anything you install with `easy_install` or
`pip install` will not affect the rest of the sytem.


#### Install dependencies

This repo has a `requirements.txt` file. If you get it in your current working
directory, you can install all the dependencies listed there with:

```sh
$ pip install -r requirements.txt
```

## Topics

- [01. Basic tensorflow usage](/01-basic)


[Tensorflow]: https://www.tensorflow.org/
[Python-downloads]: https://www.python.org/downloads/
[virtualenv]: https://virtualenv.pypa.io/en/stable/
