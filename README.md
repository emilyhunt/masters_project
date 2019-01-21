# Photometric redshift inference with a mixture density network

*An adventure in redshift inference using a PDF-outputting neural network code*

## Getting started

### Pre-requisites

Version numbers are for versions of packages used on the development machine.

#### Required: 

(Everything that actually does something important in the core code)

* Python (3.6.6) (including base packages os, sys, time, gc and typing)
* TensorFlow (1.12.0)
* TensorBoard (1.12.0)
* numpy (1.15.2)
* scipy (1.1.0)
* pandas (0.23.4)
* matplotlib (2.2.3)
* pymc3 (3.5)


#### Optional:

* astropy (3.0.5) (only used to read certain data types)
* LaTeX on your local machine to make plots but with fancy fonts (only worth it for report/paper writing)


### Running the code

If everything above is installed, then you should be able to run the code with default data and settings (using `run_3dhst.py`). It can also be ran on fake data (`run_fake_data.py`) and can experiment with using small training data sets (`tac_3dhst.py`).

Extra files are included to generate plots from final data, and begin with `plotter_`.

*What to expect while it's going:*

* The code will run in your terminal, giving you updates on how it's doing and an ETA of when it's going to finish.

* The code will generate logs in `/logs/` each time it prints to terminal, that can be viewed by starting a tensorboard server. In a terminal with python active, try `tensorboard --logdir=<my log dir>`. You should then be able to access tensorboard in an internet browser, at `<computer name/address>:6006`.

* At the end of running, `run_3dhst.py` will generate plots at `/plots/<run super name>/<run name>/`. It dumps quite a lot of plots out by default.


## Code outline

All core code exists in the *scripts* folder, an outline of which is explained below in alphabetical order:

* `file_handling.py`: functions for reading save and fits files.

* `galaxy_pairs.py`: code that implements the Quadri *et. al* (2010) photometric redshift error estimation with galaxy pairs.

* `loss_funcs.py`: loss (aka cost) functions for networks to use. The most developed one is `NormalPDFLoss` which implements the original version of a mixture density function's loss function. The `<something>CDFLoss` functions were experimentatal and are included for completeness only.

* `mdn.py`: the full definition of a mixture density network class, including lots of helpful member functions for training and evaluating the code.

* `preprocessing.py`: a set of pre-processing functions and a preprocessor class that handles scaling.

* `util.py`: bits and bobs, including definitions of distributions and a checker for redshifts that are valid.

* `z_plot.py`: plotting functions that make pretty plots for evaluating the code.


## Extra files that are on the Bath X-Drive only (not on GitHub):

* A ton of plots at `/plots/`

* All of my final data and plots

* Data used in the running, stored locally so that it's easier to access

* Some random bits of example code I used during development in `/examples/`.


## Digging deeper

Most of the code is quite well commented, although some comments are a bit out of date due to a bit of a rush to finish the project at the end but without prioritising writing perfectly documented software. I hope nothing is too bad, but in spots you might find e.g. `# todo: emily please comment your code better` in places where I never got around to updating some longer comments (in particular some function strings.)

Do contact me at the below details if you're trying to use the code but run into any frustrating issues.


## Author

This code was written by Emily Hunt (eh594@bath.ac.uk, or emily.hunt.physics@gmail.com after she graduates, or @emilydoesastro on Twitter) under the supervision of Dr Stijn Wuyts.


## Versioning

Versionin numbering was not used. Refer to commit notes on GitHub to check out how the code changed over time.


## License

MIT License. See LICENSE.md for more.


## Acknowledgements

This code was inspired by [a blog post by cbonnett](http://cbonnett.github.io/MDN.html).

Thank-you to Dr Stijn Wuyts for his supervision, everyone in our office for their moral support and Caroline Bertemes for her helpful comments. 
