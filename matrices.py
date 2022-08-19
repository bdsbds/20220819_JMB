#  https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functools import partial
import scipy.optimize as optimize
import pygmo as pg
import sys


def getInteractionMatrix(myCity, alpha=1, gamma=2, dist_c=50):
	_distanceMatrix = myCity.distanceMatrix
	neighborhood = myCity.neighborhoodMatrix
	_A = alpha*np.power(1+_distanceMatrix/dist_c, -gamma)
	# _A = np.eye(4) pour annuler la diffusion
	return _A*neighborhood


def appendIfNotNan(idList, tab, n):
	for i in range(n):
		if ~np.isnan(tab[i]):
			idList += [(i, np.int(tab[i]))]
	return idList
