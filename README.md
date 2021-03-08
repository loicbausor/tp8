# customMLLib
Very simple template for packaging ML project

## The template
You will see that this package already has a setup.py, just provide your custom information, package requirements !

You already have an empty package folder with an ```__init__.py``` file inside ;-)

Just add some modules to it (.py files with functions and classes).

## Install
```pip install path/to/package/folder```

## Usage
```python
from customML.mymodule import myfunction
```

## What I expect
Implement one of your TP under this template.

It will not be easy to test packages that use tensorflow without GPU, but have a try.

I will judge how your package is easy to install with ```pip```, how it works, but not necessarily if it produces correct results.

I just check deployment and usability of your sources.

Next to the package, I expect a script (just like ```myscript.py```to be usable and have **logs**, **custom errors** and **argument parsing**) that uses some functions of your package.

I'm not expecting doc, but you have to replace this README.md to describe what your package do, how to install and requirements.

I'm not expecting unit tests.

## Resources
Another simple package structure (with unit tests also!) can be found here: https://github.com/navdeep-G/samplemod

I parameterized your logger, but you will find a tutorial on logging here: https://docs.python.org/3/howto/logging.html

I started a parser in your script, but you will find the doc of argument parsing here: https://docs.python.org/3/library/argparse.html

Other resources can be found in the last slides of the course.

Enjoy, ask all the questions you have.
