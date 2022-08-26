import math
import datetime

import numpy as np
import yaml
import scipy.sparse as sp
from fractions import Fraction
from models import *
from scipy import special
import os

os.chdir('C:\Code\GEOBIOMICS\Joshua_JMB')

class setParamsV3:
	def __init__(self, modelParams, simulParams, model):
		self.beta0 = modelParams['beta0']
		self.beta1 = modelParams['beta1']
		self.n = int(simulParams['n'])
		self.K = int(simulParams['K'])
		self.model = model
		if model == 'ode0':
			self.n = 2
		elif (model == 'odeK_old') or (model == 'odeK'):
			self.n = self.K + 2# used for visualization purposes
		self.sigma = modelParams['sigma']
		self.gamma = modelParams['gamma']
		self.mu0 = modelParams['mu0']
		self.popTotale = 1.
		#
		self.Tfin = simulParams['endTimeDays']
		self.deltaT = simulParams['deltaT']
		self.I_init = simulParams['I_init']
		self.lambda_init = simulParams['lambda_init']
		self.S_init = self.popTotale - self.I_init
		self.aL = 5# magic number, just in case mu0 = 0
		if self.mu0 > 0:
			self.aL = np.min([self.sigma / np.sqrt(2. * self.mu0), 5])
		self.deltaA = self.aL / np.sqrt(self.n)
		self.timeStep = self.deltaT
		# in case of large mu0 value, lower time step required
		if self.mu0 > 0:
			self.timeStep = np.min([self.timeStep, np.log(1 + 1. / self.n) / self.mu0])
		# in case of large sigma value
		if self.sigma > 0:
			self.timeStep = np.min([self.timeStep, 2 * self.deltaA**2 / self.sigma**2])
		print("time step = ", self.timeStep)
		self.timeVec = np.arange(0, self.Tfin + self.deltaT, self.deltaT)
		# useless
		self.datetimes = np.array([datetime.datetime(2020, 5, 12) + datetime.timedelta(seconds = jj * 24 * 3600) for jj in self.timeVec])


def setInitModelV3(myParams):
	n = myParams.n
	K = myParams.K
	I_init = myParams.I_init
	S_init = myParams.S_init
	deltaA = myParams.deltaA
	lambda_init = myParams.lambda_init
	if (myParams.model == 'pde_old') or (myParams.model == 'pde'):
		X = np.zeros(n+1)
		X[0:n] = np.array([sinitFuncP((k+0.5) * deltaA, myParams) for k in np.arange(n)])
		alpha = np.sum(X[0:n])*deltaA / S_init
		X = X / alpha
		X[n] = I_init
	elif myParams.model == 'ode0':
		X = np.zeros(3)
		X[0] = S_init
		X[1] = myParams.beta0 + lambda_init**2 * myParams.beta1
		X[2] = I_init
	elif (myParams.model == 'odeK_old') or (myParams.model == 'odeK'):
		X = np.zeros(K+3)
		X[0] = S_init
		if myParams.K > 0:
			X[1] = S_init/4
		X[K+1] = lambda_init
		X[K+2] = I_init
		X[0:K+1] = initS_odek(myParams)
	else:
		print('Model does not exist, please double check yaml file')
	return X


def coef_alpha_2k(k):
	return np.sqrt(np.sqrt(np.pi) * math.factorial(2*k) * np.power(2, 2*k) / 2)


def integral_psi2k_square(k):
	return  np.sqrt( math.factorial(2*k) / np.power(2, 2*k) / math.factorial(k)**2 * np.sqrt(np.pi) )


def psi2k_function(k, a):
	return Hermite2k(k, a) / coef_alpha_2k(k) * np.exp(-a**2 / 2)


def Hermite2k( k, a):
	#if k == 0:
	#	coefs = np.array([1])
	#elif k == 1:
	#	coefs = np.array([-2, 4])
	#elif k == 2:
	#	coefs = np.array([12, -48, 16])
	#elif k == 3:
	#	coefs = np.array([-120, 720, -480, 64])
	#elif k == 4:
	#	coefs = np.array([1680, -13440, 13440, -3584, 256])
	#elif k == 5:
	#	coefs = np.array([-30240, 302400, -403200, 161280, -23040, 1024])
	coefs = np.array([ special.hermite(2 * k)[2 * i] for i in range(k+1)])
	return np.dot(np.array([np.power(a, 2*m) for m in range(k+1)]), coefs)


def initS_odek(myParams):
	K = myParams.K
	n = 1000
	deltaA = 0.01
	SinitK = np.zeros(K+1)
	inputa = (0.5 + np.arange(n)) * deltaA
	for k in range(K+1):
		psi2k_vector = np.array([psi2k_function(k, x) for x in inputa])
		SinitK[k] = deltaA * np.dot(sinitFuncP(inputa, myParams), psi2k_vector)
	renorm = np.array([ integral_psi2k_square(k) * SinitK[k] for k in np.arange(K+1)]).sum()
	SinitK = SinitK / renorm * (1. - myParams.I_init)
	return SinitK


def step_transport(S, myParams):
	mu0 = myParams.mu0
	n = myParams.n
	timeStep = myParams.timeStep
	Snew = np.zeros(n)
	kappa = np.exp(mu0 * timeStep)
	for i in range(n):
		if i == n-1:
			Snew[i] = S[i] * kappa
		else:
			Snew[i] = S[i] * ((i+1) - i * kappa) + S[i+1] * ( (i+1) * (kappa - 1))
	return Snew


def step_diffusion(S, myParams):
	'''
	Explicit scheme in time for diffusion
	'''
	Cmat = getDiffusionMatrix(myParams)
	timeStep = myParams.timeStep
	return S + timeStep * Cmat.dot(S)


def sir_raw(X, tt, myParams):
	'''
	SIR model without drift nor diffusion, still with vector valued susceptible S
	X = vector of size=n+1 corresponding to the concatenation of
	S (size n), I (size 1)
	'''
	n = myParams.n
	Amat = getDiagBeta(myParams)
	onesVec = np.ones(n)
	dX = np.zeros(n+1)
	logS = X[0:n]
	logI = X[n]
	S = np.exp(logS)
	I = np.exp(logI)
	dX[0:n] = - Amat.dot(onesVec) * I / myParams.popTotale
	dX[n] = Amat.dot(S).sum() * myParams.deltaA / myParams.popTotale - myParams.gamma
	return dX


def sir_diffX(X, tt, myParams):
	'''
	X = vector of size=n+1 corresponding to the concatenation of
	S (size n), I (size 1)
	'''
	n = myParams.n
	Cmat = getDiffusionMatrix(myParams)
	Bmat = getDriftMatrix(myParams)
	Amat = getDiagBeta(myParams)
	dX = np.zeros(n+1)
	S = X[0:n]
	I = X[n]
	newlyInfected = Amat.dot(S) * I / myParams.popTotale
	dX[0:n] = - newlyInfected + Bmat.dot(S) + Cmat.dot(S)
	dX[n] = newlyInfected.sum() * myParams.deltaA - I * myParams.gamma
	dX[X+dX < 0.] = 0.
	return dX


def sir_diffX_ode0(X, tt, myParams):
	'''
	ODE model in terms of (S, barBeta, I, R)
	X = vector of size=3 corresponding to the concatenation of
	S (size 1), Beta (size 1) and I (size 1)
	'''
	dX = np.zeros(3)
	dX[0] = - X[1] * X[0] * X[2] / myParams.popTotale
	dX[1] = - 2 * X[2] * (X[1] - myParams.beta0)**2 \
			- 2 * myParams.mu0 * (X[1] - myParams.beta0) \
			+ myParams.sigma**2 * myParams.beta1
	dX[2] = X[1] * X[0] * X[2] / myParams.popTotale - myParams.gamma * X[2]
	dX[X + dX < 0.] = 0. # bad idea
	return dX


def sir_diffX_odeK_old(X, tt, myParams):
	'''
	deprecated: clipping required for negative values -> use odeK instead
	X = vector of size=K+3 corresponding to the concatenation of
	S_k (size K+1), lambda (size 1) and I (size 1)
	'''
	K = myParams.K
	dX = np.zeros(myParams.K+3)
	tildeAlphaCoefs = np.array([integral_psi2k_square(k) for k in range(K+1)])
	barBeta = (1. + 4. * np.sum(X[0:K+1] * np.arange(K+1) * tildeAlphaCoefs) \
			   / np.sum(X[0:K+1] * tildeAlphaCoefs)) * myParams.beta1 * X[K+1]**2 + myParams.beta0
	GG0 = sp.diags( np.arange(K+1) )
	G0 = sp.csr_matrix(GG0)
	GG1 = sp.diags([np.zeros(K+1), np.array([np.sqrt( 2 * (k+1) * (2*k+1)) for k in range(K)])], [0, 1])
	G1 = sp.csr_matrix(GG1)
	IDD = sp.diags(np.ones(K+1))
	ID = sp.csr_matrix(IDD)
	dX[0:K+1] = - (myParams.beta0 + X[K+1]**2 * myParams.beta1) * X[K+2] * ID.dot(X[0:K+1]) \
			  + (myParams.sigma**2 / 2 / X[K+1]**2 - myParams.beta1 * X[K+2] * X[K+1]**2) * G1.dot(X[0:K+1]) \
			  - (2 * myParams.beta1 * X[K+2] * X[K+1]**2 + myParams.sigma**2 / X[K+1]**2) * G0.dot(X[0:K+1])
	dX[K+1] = -  myParams.beta1 * X[K+2] * X[K+1]**3 - myParams.mu0 * X[K+1] + myParams.sigma**2 / 2 / X[K+1]
	dX[K+2] = barBeta * X[K+2] * np.sum( np.dot(X[0:K+1], tildeAlphaCoefs))  \
			  / myParams.popTotale - myParams.gamma * X[K+2]
	dX[X + dX < 0.] = 0. # clipping, bad idea
	return dX


def sir_diffX_odeK(X, tt, myParams):
	'''
	First K modes of Schrödinger operator, solved in LogS, logLambda and LogI
	X = vector of size=K+3 corresponding to the concatenation of
	S_k (size K+1), lambda (size 1) and I (size 1)
	'''
	K = myParams.K
	beta0 = myParams.beta0
	beta1 = myParams.beta1
	sigma = myParams.sigma
	mu0 = myParams.mu0
	dX = np.zeros(myParams.K+3)
	tildeAlphaCoefs = np.array([integral_psi2k_square(k) for k in range(K+1)])
	barBeta = (1. + 4. * np.sum(np.exp(X[0:K+1]) * np.arange(K+1) * tildeAlphaCoefs) \
			   / np.sum(np.exp(X[0:K+1]) * tildeAlphaCoefs)) * beta1 * np.exp(2 * X[K+1]) + beta0
	GG0 = sp.diags(np.arange(K+1))
	G0 = sp.csr_matrix(GG0)
	GG1 = sp.diags([np.zeros(K+1), np.array([ np.sqrt(2 * (k+1) * (2*k+1)) for k in range(K)])], [0, 1])
	G1 = sp.csr_matrix(GG1)
	dX[0:K+1] = - (beta0 + np.exp(2 * X[K+1]) * beta1) * np.exp(X[K+2])  \
			  + (sigma**2 / 2 / np.exp(2 * X[K+1]) - beta1 * np.exp(X[K+2]) * np.exp(2 * X[K+1])) \
				* G1.dot(np.exp(X[0:K+1])) / np.exp(X[0:K+1]) \
			  - (2 * beta1 * np.exp(X[K+2]) * np.exp(2 * X[K+1]) + sigma**2 / np.exp(2 * X[K+1])) \
				* G0.dot(np.exp(X[0:K+1])) / np.exp(X[0:K+1])
	dX[K+1] = -  beta1 * np.exp(X[K+2]) * np.exp(2 * X[K+1]) - mu0 + sigma**2 / 2 / np.exp(2 * X[K+1])
	dX[K+2] = barBeta * np.sum(np.dot(np.exp(X[0:K+1]), tildeAlphaCoefs)) / myParams.popTotale - myParams.gamma
	return dX


def sir_runX_pde(X, myParams):
	'''
	PDE solver with n discrete points over the half line
	X = vector of size=n+1 corresponding to the concatenation of
	S (size n), and I (size 1)
	'''
	nts = myParams.timeVec.size
	np.seterr(divide = 'ignore')
	n = myParams.n
	result = np.zeros([myParams.timeVec.size, myParams.n+1])
	result[0, :] = X
	for i in range(nts-1):
		#print("i = ", i, " ; t = ", (i+1) * timeStep)
		#result[i, :][ result[i, :] == 0. ] = 1e-15
		Xold = np.log(result[i, :])
		Xnew = sir_runX_step(Xold, myParams)[-1, :]
		Xnew = np.exp(Xnew)
		Xnew[0:n] = step_transport(Xnew[0:n], myParams)
		Xnew[0:n] = step_diffusion(Xnew[0:n], myParams)
		result[i + 1, :] = Xnew
	return result


def sir_runX(X, myParams):
	return odeint(sir_diffX, X, myParams.timeVec, args=(myParams, ))


def sir_runX_step(X, myParams):
	return odeint(sir_raw, X, np.array([0, myParams.timeStep]), args=(myParams, ))


def sir_runX_ode0(X, myParams):
	return odeint(sir_diffX_ode0, X, myParams.timeVec, args=(myParams, ))


def sir_runX_odeK_old(X, myParams):
	return odeint(sir_diffX_odeK_old, X, myParams.timeVec, args=(myParams, ))


def sir_runX_odeK(X, myParams):
	logX = np.log(X)
	LogResult = odeint(sir_diffX_odeK, logX, myParams.timeVec, args=(myParams, ))
	return np.exp(LogResult)


def computeModel(X, myParams):
	if myParams.model == "pde_old":
		result = sir_runX(X, myParams)
	elif myParams.model == "pde":
		result = sir_runX_pde(X, myParams)
	elif myParams.model == "ode0":
		result = sir_runX_ode0(X, myParams)
	elif myParams.model == "odeK_old":
		result = sir_runX_odeK_old(X, myParams)
	elif myParams.model == "odeK":
		result = sir_runX_odeK(X, myParams)
	else:
		print('MODEL ERROR')
	return result


def getParamsV3(inputFile = './paramsV3.yaml'):
	with open(inputFile) as file:
		paramSet = yaml.load(file, Loader=yaml.FullLoader)
	paramSet['modelparams'] = fracToFloat(paramSet['modelparams'])
	paramSet['simulparams'] = fracToFloat(paramSet['simulparams'])
	return paramSet


def getMatrixDiffusionOld(n):
	'''
	deprecated
	'''
	M = sp.diags([-2*np.ones(n), np.ones(n-1), np.ones(n-1)], [0, 1, -1])
	H = sp.csr_matrix(M)
	H[0, 0] = -1
	H[n-1, n-1] = -1
	return H


def getDiffusionMatrix(myParams):
	'''
	Build diffusion matrix with zero flux at the boundary a=0 and a = C sqrt(n)
	'''
	n = myParams.n
	deltaA = myParams.deltaA
	sigma = myParams.sigma
	M = sp.diags([-2 * np.ones(n), np.ones(n-1), np.ones(n-1)], [0, 1, -1])
	H = sp.csr_matrix(M)
	H[0, 0] = -1
	H[n-1, n-1] = -1
	return H * sigma**2 / 2. / deltaA**2


def getDriftMatrix(myParams):
	'''
	Drift matrix with centered scheme
	'''
	n = myParams.n
	deltaA = myParams.deltaA # computed
	mu0 = myParams.mu0
	centered = False
	if centered:
		M = deltaA * sp.diags([np.ones(n), np.arange(n-1), -np.arange(n-1)], [0, 1, -1])
		H = sp.csr_matrix(M)
		H[n-1, n-1] = -(n-1) * deltaA
	else:
		M = deltaA * sp.diags([- (1 + 2*np.arange(n)), 1+2*(1+np.arange(n-1)), ], [0, 1])
		H = sp.csr_matrix(M)
		H[n-1, n-1] = 0
	return H * mu0 / 2. / deltaA


def sinitFunc(a, myParams):
	aL = myParams.aL
	S_init = myParams.S_init
	aL = 1.
	return S_init / aL * np.sqrt(2. / np.pi) * np.exp( -a**2 / 2. / aL**2)


def sinitFuncP(a, myParams):
	aL = myParams.aL
	S_init = myParams.S_init
	aL = 1.
	p = 1
	return  S_init / aL * (a / aL)**p * np.exp( -a / aL) / math.factorial(p)

# plt.plot([ (k+0.5)*myParams.deltaA for k in np.arange(1000)], [sinitFunc((k+0.5)*myParams.deltaA, myModelParams) for k in np.arange(100)])

def betaFunc(a, myParams):
	beta0 = myParams.beta0
	beta1 = myParams.beta1
	return beta0 + beta1 * a**2

# plt.plot([ (k+0.5)/1000 for k in np.arange(1000)], [betaFunc((k+0.5)/1000, myModelParams) for k in np.arange(1000)])

def getDiagBeta(myParams):
	return sp.diags(np.array([betaFunc((k + 0.5) * myParams.deltaA, myParams) for k in np.arange(myParams.n)]))


def fracToFloat(y):
	for item in y:
		y[str(item)] = float(Fraction(y[str(item)]))
	return y


def strToFloat(y):
	if y is not None:
		return np.array([float(z) for z in y])
	else:
		return np.array([])


def plotI(result, myParams, yscaleopt = "log", xvar = "index"):
	minInfectedRate = 1.e-6
	timeVec = myParams.timeVec
	datetimes = myParams.datetimes
	plt.yscale(yscaleopt)
	n = int(result.shape[1]) - 1
	tmp = 1.1
	ptitle = 'Fraction of infected individuals'
	if yscaleopt == "log":
		tmp = 3
		ptitle += ' (log scale)'
	plt.ylim( (minInfectedRate, tmp * np.max(result[:, n]) ) )
	plt.xlabel('Number of days')
	plt.ylabel('Fraction of infected individuals')
	plt.gca().set_ylim(bottom=minInfectedRate)
	if (xvar == "index"):
		xvariable = timeVec
	else:
		xvariable = datetimes
		plt.xticks(rotation = 45)
	plt.plot(xvariable, result[:, n], color='black', linewidth=2)
	plt.title(ptitle)
	plt.show()
	pd.DataFrame(result[:, n]).to_csv(myParams.model + ".csv")



def plotS(result, myParams, yscaleopt="log", xvar="index"):
	plt.yscale(yscaleopt)
	minSusceptibleRate = 1.e-5
	n = int(result.shape[1]) - 1
	tmp = 1.1
	ptitle = 'Fraction of susceptible individuals, ' + str(n) + ' groups '
	if yscaleopt == "log":
		tmp = 3
		ptitle += ' (log scale)'
	plt.ylim((minSusceptibleRate, tmp * np.max(result[:, 0:n])))
	if xvar == "index":
		xvariable = myParams.timeVec
	else:
		xvariable = myParams.datetimes
		plt.xticks(rotation=45)
	for i in np.arange(n):
		plt.plot(xvariable, result[:, i] )
	plt.plot(xvariable, result[:, 0:n].sum(axis=1) / myParams.n, color='black', linewidth=2)
	plt.title(ptitle)
	plt.show()


def plotSa(result, myParams, yscaleopt="log", sampleTimes = np.arange(0., 500., 20.)):
	plt.yscale(yscaleopt)
	minSusceptibleRate = 1.e-5
	n = int(result.shape[1]) - 1
	sampleTimes = sampleTimes[sampleTimes <= myParams.Tfin]
	tmp = 1.1
	a_variable = np.array([(k+0.5) * myParams.deltaA for k in np.arange(n)])
	ptitle = 'Fraction of susceptible individuals, ' + str(n) + ' groups '
	if yscaleopt == "log":
		tmp = 3
		ptitle += ' (log scale)'
	plt.ylim((minSusceptibleRate, tmp*np.max(result[:, 0:n]  )))
	for i in np.array(sampleTimes/myParams.deltaT).astype(int):
		plt.plot(a_variable, result[i, 0:n] )
	plt.title(ptitle)
	plt.legend(sampleTimes)
	plt.show()


def plotSK(result, myParams, yscaleopt="log", xvar="index"):
	plt.yscale(yscaleopt)
	minSusceptibleRate = 1.e-5
	n = int(result.shape[1]) - 1
	K = myParams.K
	tildeAlphaCoefs = np.array([integral_psi2k_square(k) for k in range(K+1)])
	resultK = np.dot(result[:, 0:K+1], tildeAlphaCoefs[0:K+1])
	#tmp=1.1
	ptitle = 'Fraction of susceptible individuals, ' + str(K) + ' modes '
	if yscaleopt == "log":
		ptitle += ' (log scale)'
	plt.ylim((minSusceptibleRate, 1.3))
	if (xvar == "index"):
		xvariable = myParams.timeVec
	else:
		xvariable = myParams.datetimes
		plt.xticks(rotation = 45)
	plt.plot(xvariable, resultK)
	plt.title(ptitle)
	plt.show()


def main():
	##################
	# INITIALIZATION #
	##################
	paramSet = getParamsV3('./paramsV3.yaml')
	print(paramSet)
	model = paramSet['model']
	modelParams = paramSet['modelparams']
	simulParams = paramSet['simulparams']
	print(" ")
	print('# INITIALIZATION ', model)
	print(" ")
	myParams = setParamsV3(modelParams, simulParams, model)
	X0 = setInitModelV3(myParams)
	if (myParams.model == 'pde_old') or (myParams.model == 'pde'):
		myParams.popTotale = X0[0:myParams.n].sum() * myParams.deltaA + X0[myParams.n]
	else:
		myParams.popTotale = 1.
	print('Total population = ', myParams.popTotale)
	###########
	# results #
	###########
	print(" ")
	print("# Simulation results")
	print(" ")
	result = computeModel(X0, myParams)
	plotI(result / myParams.popTotale, myParams, yscaleopt='log')
	if (myParams.model == 'pde_old') or (myParams.model == "pde"):
		plotS(result / myParams.popTotale, myParams, yscaleopt='log')
		plotSa(result / myParams.popTotale, myParams, yscaleopt='log')
		df = pd.DataFrame(result[:, myParams.n])
		df.to_csv(myParams.model+"_n"+str(myParams.n)+".csv")
	elif (myParams.model == 'odeK_old') or (myParams.model == 'odeK'):
		plotSK(result / myParams.popTotale, myParams, yscaleopt='log')
		df = pd.DataFrame(result[:, myParams.n])
		df.to_csv(myParams.model + "_K" + str(myParams.K) + ".csv")


if __name__ == "__main__":
	main()
