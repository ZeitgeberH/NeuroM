from neurom.io import neurolucida
from neurom.io.datawrapper import DataWrapper
from neurom.core.dataformat import COLS
# from functools import partial
from neurom.morphmath import convex_hull
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from neurom.features.utilities import getSomaStats

class MultiSoma(object):
    def __init__(self, fname) -> None:
        ''' load .asc file with multiple somas and contour
        '''
        print('Multi-soma is set True! Only .asc file is supported!')
        try:
            morphor = neurolucida.read(fname, data_wrapper=DataWrapper)
        except:
            raise('neurolucida reading error')
        self.pop_neurons = []
        self.custom_data = []
        self.soma_hulls = []
        self.centers = []
        for idx, rdw_ in enumerate(morphor):
            if rdw_.data_block[0][COLS.TYPE] == 1:  ## somas
                self.pop_neurons.append(rdw_.soma_points()) ## ndarray of soma points (n x 7)
                points = rdw_.soma_points()[:,:2]
                maxDia, soma_center, soma_radius, soma_avgRadius = getSomaStats(rdw_.soma_points()[:,:3])
                self.centers.append(soma_center)
                self.soma_hulls.append(convex_hull(points))
            else: ## ignore this part for now
                self.custom_data.append(rdw_)
    @classmethod
    def getDistMatrix(self, centers):
        ''' measure pair-wise inter-soma distance
        centers: dict with keys: X, Y, Z
        '''
        distanceList = []
        if len(centers["X"]) > 1:## calculte distance between cells 
            x0, y0, z0 = centers["X"][-1], centers["Y"][-1], centers["Z"][-1]
            for kkk, _ in enumerate(centers["X"][:-1]):
                d = np.round(
                    np.sqrt(
                        (centers["X"][kkk] - x0) ** 2
                        + (centers["Y"][kkk] - y0) ** 2
                        + (centers["Z"][kkk] - z0) ** 2
                    ),
                    1,
                )
                distanceList.append(d)
        return distanceList

    def drawHulls(self, ax=None, contour_color='k',alpha=0.2, fill=True,\
        contour_on=True, faceColor='g', contour_linewidth=2, labels=None,realistic_diameters=False):
        if ax is None:
            fig, ax = plt.subplots()
        if labels is None:
            labels = list(np.arange(len(self.pop_neurons)+1))
        if fill:
            for points, hull in zip(self.pop_neurons, self.soma_hulls):        
                ax.add_patch(Polygon(points[hull.vertices,:2], fill=True, color=faceColor, alpha=alpha))
        c = 0
        for n1, h1 in zip(self.pop_neurons, self.soma_hulls):
            ax.plot(n1[h1.vertices,0], n1[h1.vertices,1], color=contour_color,\
                 linestyle='--', lw=contour_linewidth, label=labels[c])
            ax.axes.text(
            self.centers[c][0],
            self.centers[c][1],
            labels[c],
            horizontalalignment="center",
            verticalalignment="center",
            color="r",
            fontsize=11,
        )
            c +=1




