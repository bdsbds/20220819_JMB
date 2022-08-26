import numpy as np
from scipy.integrate import odeint
from functools import partial
import scipy.optimize as optimize
import pygmo as pg
import os


class modelParameters:
	def __init__(self, y0, beta0, beta1, mu0, sigma, gamma, simTime):
		self.y0 = y0
		self.beta0 = beta0
		self.beta1 = beta1
		self.mu0 = mu0
		self.sigma = sigma
		self.gamma = gamma
		self.simTime = simTime
