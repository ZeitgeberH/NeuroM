from neurom.io import neurolucida
from neurom.io.datawrapper import DataWrapper
from neurom.core.dataformat import COLS
# from functools import partial
from neurom.morphmath import convex_hull
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

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
        for idx, rdw_ in enumerate(morphor):
            if rdw_.data_block[0][COLS.TYPE] == 1:  ## somas
                self.pop_neurons.append(rdw_.soma_points()) ## ndarray of soma points (n x 7)
                points = rdw_.soma_points()[:,:2]
                self.soma_hulls.append(convex_hull(points))
            else: ## ignore this part for now
                self.custom_data.append(rdw_)

    def getDistMatrix(self):
        ''' measure inter-soma distance
        '''
        raise NotImplementedError

    def drawHulls(self, ax=None, contourColor='k',alpha=0.2, fill=True,faceColor='g', linewidth=2):
        if ax is None:
            fig, ax = plt.subplots()
        if fill:
            for points, hull in zip(self.pop_neurons, self.soma_hulls):        
                ax.add_patch(Polygon(points[hull.vertices,:2], fill=True, color=faceColor, alpha=alpha))
        for n1, h1 in zip(self.pop_neurons, self.soma_hulls):
            ax.plot(n1[h1.vertices,0], n1[h1.vertices,1], color=contourColor, linestyle='--', lw=linewidth)


