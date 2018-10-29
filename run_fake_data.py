"""Trains a network on fake data from the blog example."""

import numpy as np
import pandas as pd


# Import the data

data_train = pd.read_csv('data/galaxy_redshift_sims_train.csv')
data_validation = pd.read_csv('data/galaxy_redshift_sims_valid.csv')

#


# Create a network
