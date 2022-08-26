import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 18})


def main():
	x0 = pd.read_csv('odeK_K0.csv')
	x0.columns = ['Day', 'I']
	x1 = pd.read_csv('odeK_K1.csv')
	x1.columns = ['Day', 'I']
	x2 = pd.read_csv('odeK_K2.csv')
	x2.columns = ['Day', 'I']
	x3 = pd.read_csv('odeK_K3.csv')
	x3.columns = ['Day', 'I']
	x4 = pd.read_csv('odeK_K4.csv')
	x4.columns = ['Day', 'I']
	x7 = pd.read_csv('odeK_K7.csv')
	x7.columns = ['Day', 'I']
	xpde200 = pd.read_csv('pde2_n200.csv')
	xpde200.columns = ['Day', 'I']
	plt.yscale('log')
	plt.xlim((0, 200))
	plt.title('PDE solution vs multimode ODE approximations')
	plt.plot(x0['Day']/100., x0['I'])
	plt.plot(x1['Day']/100., x1['I'])
	plt.plot(x2['Day']/100., x2['I'])
	plt.plot(x3['Day']/100., x3['I'])
	plt.plot(x4['Day']/100., x4['I'])
	plt.plot(x7['Day']/100., x7['I'])
	plt.plot(xpde200['Day']/100., xpde200['I'], color = 'black')
	plt.legend(['K=0', 'K=1', 'K=2', 'K=3', 'K=4', 'K=7', 'PDE 200 points'])
	plt.xlabel('Days, t')
	plt.ylabel('Fraction of infected, I(t)')

if __name__ == "__main__":
	main()
