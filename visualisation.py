#  https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from models import *


def plotMesures(myCity, simTime, timeMesure, I_mesure, ptitle='Mesures de fraction de population infectée'):
    nb_zones = myCity.nb_zones
    nb_zones_x = myCity.nb_zones_x
    nb_zones_y = myCity.nb_zones_y
    timeVec = simTime.timeVec
    fig, axs = plt.subplots(nb_zones_y, nb_zones_x)
    fig.suptitle(ptitle)
    ymax = 1.1 * np.nanmax(I_mesure)
    if I_mesure.shape[0] > 0:
        for i in range(nb_zones):
            if nb_zones == 1:
                axs.set_xlim(0, np.nanmax(timeVec))
                axs.plot(timeMesure, I_mesure[:, i], marker='o', linestyle='', color='b')
                axs.legend(['I' + str(i + 1) + ' measures'])
                axs.set_ylim(0, ymax)
            elif nb_zones_x == 1 or nb_zones_y == 1:
                axs[i].set_xlim(0, np.nanmax(timeVec))
                axs[i].plot(timeMesure, I_mesure[:, i], marker='o', linestyle='', color='b')
                axs[i].legend(['I' + str(i + 1) + ' measures'])
                axs[i].set_ylim(0, ymax)
            else:
                axs[i // nb_zones_x, i % nb_zones_y].set_xlim(0, np.max(timeVec))
                axs[i // nb_zones_x, i % nb_zones_y].plot(timeMesure, I_mesure[:, i], marker='o', linestyle='',
                                                          color='b')
                axs[i // nb_zones_x, i % nb_zones_y].legend(['I' + str(i + 1) + ' measures'])
                axs[i // nb_zones_x, i % nb_zones_y].set_ylim(0, ymax)
    else:
        print("Aucune mesure : pas d'affichage des graphes")
    plt.show()
    return 0


def aggregateResult(paramSet, myCity, zoneList, result, variable='I'):
    model = paramSet['model']
    localData = myCity.localData
    nn = localData.shape[0]
    coef = getModelCoef(model)
    if variable=='R':
        coef += 1
    tmp1 = result[:, coef * nn:(coef + 1) * nn] * np.array(localData['Population_hab'])
    return np.sum(tmp1[:, np.array(zoneList)], axis=1) / localData['Population_hab'][np.array(zoneList)].sum()
