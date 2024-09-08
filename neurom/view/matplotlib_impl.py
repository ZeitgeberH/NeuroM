# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 501ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Morphology draw functions using matplotlib."""

from functools import wraps
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.transforms as mtransforms
import matplotlib.offsetbox as offsetbox
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from neurom import NeuriteType, geom
from neurom.core.morphology import iter_neurites, iter_sections, iter_segments
from neurom.core.soma import SomaCylinders
from neurom.core.dataformat import COLS
from neurom.core.types import tree_type_checker
from neurom.morphmath import segment_radius
from neurom.view.dendrogram import Dendrogram, get_size, layout_dendrogram, move_positions
from neurom.view import matplotlib_utils
from scipy.spatial import ConvexHull
import matplotlib.patheffects as path_effects
_LINEWIDTH = 1.0
_ALPHA = 0.8
_DIAMETER_SCALE = 1.0
TREE_COLOR = {NeuriteType.basal_dendrite: 'blue',
              NeuriteType.apical_dendrite: 'purple',
              NeuriteType.axon: 'red',
              NeuriteType.soma: 'black',
              NeuriteType.undefined: 'green',
              NeuriteType.custom5: 'orange',
              NeuriteType.custom6: 'orange',
              NeuriteType.custom7: 'orange',
              NeuriteType.custom8: 'orange',
              NeuriteType.custom9: 'orange',
              NeuriteType.custom10: 'orange'}

class AnchoredHScaleBar(offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = offsetbox.TextArea(label)
        self.vpac = offsetbox.VPacker(children=[size_bar,txt],  
                                 align="center", pad=ppad, sep=sep) 
        offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)

def _implicit_ax(plot_func, params=None):
    """Sets ``ax`` arg for plot functions if ``ax`` is not set originally."""

    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        fig = None
        ax = kwargs.get('ax', None)
        if ax is None and len(args) == 1:
            fig, ax = matplotlib_utils.get_figure(params=params)
            kwargs['ax'] = ax
        res = plot_func(*args, **kwargs)
        if fig:
            matplotlib_utils.plot_style(fig=fig, ax=ax)
        return res

    return wrapper


def _implicit_ax3d(plot_func):
    return _implicit_ax(plot_func, {'projection': '3d'})


def _plane2col(plane):
    """Take a string like 'xy', and return the indices from COLS.*."""
    planes = ('xy', 'yx', 'xz', 'zx', 'yz', 'zy')
    assert plane in planes, 'No such plane found! Please select one of: ' + str(planes)
    return (getattr(COLS, plane[0].capitalize()),
            getattr(COLS, plane[1].capitalize()), )


def _get_linewidth(tree, linewidth, diameter_scale):
    """Calculate the desired linewidth based on tree contents.

    If diameter_scale exists, it is used to scale the diameter of each of the segments
    in the tree
    If diameter_scale is None, the linewidth is used.
    """
    if diameter_scale is not None and tree:
        linewidth = [segment_radius(s) * diameter_scale
                     for s in iter_segments(tree)] ## neuroludica v3 use diameter instead of radius
    return linewidth


def _get_color(treecolor, tree_type):
    """If treecolor set, it's returned, otherwise tree_type is used to return set colors."""
    if treecolor is not None:
        return treecolor
    return TREE_COLOR.get(tree_type, 'green')


@_implicit_ax
def plot_tree(tree, ax=None, plane='xy',
              diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
              color=None, alpha=_ALPHA, realistic_diameters=False, enhanceBifurcationPlotting=True,
              DiameterScaling=2.0):
    """Plots a 2d figure of the tree's segments.

    Args:
        tree(neurom.core.Section or neurom.core.Neurite): plotted tree
        ax(matplotlib axes): on what to plot
        plane(str): Any pair of 'xyz'
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
        realistic_diameters(bool): scale linewidths with axis data coordinates

    Note:
        If the tree contains one single point the plot will be empty
        since no segments can be constructed.
    """
    plane0, plane1 = _plane2col(plane)

    section_segment_list = [(section, segment, len(section.children)>=2)
                            for section in iter_sections(tree)
                            for segment in iter_segments(section)]
    colors = [_get_color(color, section.type) for section, _, _ in section_segment_list]
    segs = [((seg[0][plane0], seg[0][plane1]),
                (seg[1][plane0], seg[1][plane1]))
            for _, seg, _ in section_segment_list]

    linewidth = _get_linewidth(
        tree,
        diameter_scale=DiameterScaling,
        linewidth=linewidth)
    collection = LineCollection(segs, colors=colors, linewidth=linewidth, alpha=alpha,
     path_effects=[path_effects.Stroke(capstyle="round",joinstyle='round')])  
    # trans = mtransforms.Affine2D().scale(1.0) + ax.transData
    # collection.set_transform(trans)  
    ax.add_collection(collection)
    if enhanceBifurcationPlotting:
        bifurcation_transition(section_segment_list, colors, ax) 


def bifurcation_transition(section_segment_list, colors, ax):
    """Enhance the plotting of bifurcations by plotting a polygon around the bifurcation point"""
    for (bif_point, _, isbif), color in zip(section_segment_list, colors):
        if isbif:
            parent_points = np.array(bif_point.points[-1,:])
            for child in bif_point.children:
                child_points = np.array(child.points[1,:])
                plot_polygon(parent_points, child_points, color, ax)

@_implicit_ax                
def plot_polygon(p1, p2, color, ax=None):
    """Plots a polygon"""
    if np.abs(p1[COLS.R]-p2[COLS.R]) < 0.05:
        return
    p1a, p1b = orthognal_points(p1[:2],p2[:2], p1[COLS.R])
    p1c, p1d = orthognal_points(p2[:2],p1[:2], p2[COLS.R])
    ## matplotlib draw a polygon
    ax.fill([p1a[0], p1b[0], p1c[0], p1d[0]], [p1a[1], p1b[1], p1c[1], p1d[1]],
    linewidth=0.1,color=color, alpha=1.0,antialiased=True)

def orthognal_points(p1,p2,x):
    """Returns the coordinates of two points orthogonal to the line formed by points p1 and p2 with distance x
    scale is used to scale the distance between the end points"""
    v = p2-p1
    v = v/np.sqrt(np.sum(v**2))
    v = np.array([-v[1],v[0]])
    p3 = p1 + x*v
    p4 = p1 - x*v
    return p3, p4
    
@_implicit_ax
def plot_soma(soma, ax=None, plane='xy',
              soma_outline=True,
              linewidth=_LINEWIDTH,
              color=None, alpha=_ALPHA, DiameterScaling=2.0):
    """Generates a 2d figure of the soma.

    Args:
        soma(neurom.core.Soma): plotted soma
        ax(matplotlib axes): on what to plot
        plane(str): Any pair of 'xyz'
        soma_outline(bool): should the soma be drawn as an outline
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    """
    plane0, plane1 = _plane2col(plane)
    color = _get_color(color, tree_type=NeuriteType.soma)
    if isinstance(soma, SomaCylinders):
        plane0, plane1 = _plane2col(plane)
        points = np.vstack([soma.points[:,plane0].ravel(),
                            soma.points[:,plane1].ravel()])
        
        try:
            points = points.T
            hull = ConvexHull(points)
            ax.add_patch(Polygon(points[hull.vertices], fill=True, color=color, alpha=alpha,\
            zorder=-1))
        except Exception as e:
            # print('ConvexHull failed for soma with points: {}'.format(points), e)
            points = np.column_stack([points[:,0], points[:,1]])
            if DiameterScaling >= 1.0:
                lw = soma.radius
            else:
                lw = soma.radius/2
            collection = LineCollection([points], colors=color, linewidth=lw, alpha=alpha,
            path_effects=[path_effects.Stroke(capstyle="round",joinstyle='round')])  
            # trans = mtransforms.Affine2D().scale(1.0) + ax.transData
            # collection.set_transform(trans)  
            ax.add_collection(collection)

    else:
        if soma_outline:
            ax.add_artist(Circle(soma.center[[plane0, plane1]], soma.radius,
                                 color=color, alpha=alpha))
        else:
            points = [[p[plane0], p[plane1]] for p in soma.iter()]
            if points:
                points.append(points[0])  # close the loop
                x, y = tuple(np.array(points).T)
                ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

    bounding_box = geom.bounding_box(soma)
    ax.dataLim.update_from_data_xy(np.vstack(([bounding_box[0][plane0], bounding_box[0][plane1]],
                                              [bounding_box[1][plane0], bounding_box[1][plane1]])),
                                   ignore=False)

def rotate_contour(contour, angle):
    """Rotate contour by angle in degrees."""
    angle = np.deg2rad(-angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(contour, rotation_matrix)

# pylint: disable=too-many-arguments
@_implicit_ax
def plot_morph(morph, ax=None,
               neurite_type=NeuriteType.all,
               plane='xy',
               soma_outline=True,
               diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
               color=None, alpha=_ALPHA, realistic_diameters=False,scale_bar=True, contour_on=False,
               contour_linewidth=0.1,contour_color='k', rotationContour=0,enhanceBifurcationPlotting=True,
               neutriteColors=None, DiameterScaling=2.0):
    """Plots a 2D figure of the morphology, that contains a soma and the neurites.

    Args:
        neurite_type(NeuriteType|tuple): an optional filter on the neurite type
        ax(matplotlib axes): on what to plot
        morph(Morphology): morphology to be plotted
        soma_outline(bool): should the soma be drawn as an outline
        plane(str): Any pair of 'xyz'
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
        realistic_diameters(bool): scale linewidths with axis data coordinates
        scale_bar(bool): draw a scalebar. Default:True
        contour_on(bool): draw contour of slice if any. Default:False.
        contour_linewidth(float): default 0.1
        contour_color(str): default 'k'
    """
    if neutriteColors is not None: # if neutriteColors is not None, update TREE_COLOR
        for k in neutriteColors: 
            TREE_COLOR[k] = neutriteColors[k]
    if contour_on: # draw contour if any
        for idx, x in enumerate(morph.markers):
            ps = np.array(x.points)
            if rotationContour != 0:
                ps = rotate_contour(ps[:,:2], rotationContour)
            ax.plot(ps[:,0],ps[:,1], linewidth=contour_linewidth, linestyle='--',color=contour_color,
                label='contour_'+str(idx))    
    if len(morph.soma.points) >0:
        try:
            plot_soma(morph.soma, ax, plane=plane, soma_outline=soma_outline, linewidth=linewidth,
                    color=color, alpha=alpha, DiameterScaling=DiameterScaling)
        except Exception as e:
            print('Soma failed to plot: ', e)
    else:
        print('No soma found')

    for neurite in iter_neurites(morph, filt=tree_type_checker(neurite_type)):
        plot_tree(neurite, ax, plane=plane,
                  diameter_scale=diameter_scale, linewidth=linewidth,
                  color=color, alpha=alpha, realistic_diameters=realistic_diameters,
                  enhanceBifurcationPlotting=enhanceBifurcationPlotting, DiameterScaling=DiameterScaling)
    if scale_bar:
        ob = AnchoredHScaleBar(size=50, extent = 0.01, label="50 um", loc=4, frameon=False,
                        pad=0.6,sep=4, linekw=dict(color="black"),ax=ax) 
        ax.add_artist(ob)
    ax.set_title(morph.name)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])


def _update_3d_datalim(ax, obj):
    """Unlike w/ 2d Axes, the dataLim isn't set by collections, so it has to be updated manually."""
    min_bounding_box, max_bounding_box = geom.bounding_box(obj)
    xy_bounds = np.vstack((min_bounding_box[:COLS.Z],
                           max_bounding_box[:COLS.Z]))
    ax.xy_dataLim.update_from_data_xy(xy_bounds, ignore=False)

    z_bounds = np.vstack(((min_bounding_box[COLS.Z], min_bounding_box[COLS.Z]),
                          (max_bounding_box[COLS.Z], max_bounding_box[COLS.Z])))
    ax.zz_dataLim.update_from_data_xy(z_bounds, ignore=False)


@_implicit_ax3d
def plot_tree3d(tree, ax=None,
                diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
                color=None, alpha=_ALPHA):
    """Generates a figure of the tree in 3d.

    If the tree contains one single point the plot will be empty \
    since no segments can be constructed.

    Args:
        tree(neurom.core.Section or neurom.core.Neurite): plotted tree
        ax(matplotlib axes): on what to plot
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    """
    section_segment_list = [(section, segment)
                            for section in iter_sections(tree)
                            for segment in iter_segments(section)]
    segs = [(seg[0][COLS.XYZ], seg[1][COLS.XYZ]) for _, seg in section_segment_list]
    colors = [_get_color(color, section.type) for section, _ in section_segment_list]

    linewidth = _get_linewidth(tree, diameter_scale=diameter_scale, linewidth=linewidth)

    collection = Line3DCollection(segs, colors=colors, linewidth=linewidth, alpha=alpha)
    ax.add_collection3d(collection)

    _update_3d_datalim(ax, tree)


@_implicit_ax3d
def plot_soma3d(soma, ax=None, color=None, alpha=_ALPHA):
    """Generates a 3d figure of the soma.

    Args:
        soma(neurom.core.Soma): plotted soma
        ax(matplotlib axes): on what to plot
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    """
    color = _get_color(color, tree_type=NeuriteType.soma)

    if isinstance(soma, SomaCylinders):
        for start, end in zip(soma.points, soma.points[1:]):
            matplotlib_utils.plot_cylinder(ax,
                                           start=start[COLS.XYZ], end=end[COLS.XYZ],
                                           start_radius=start[COLS.R], end_radius=end[COLS.R],
                                           color=color, alpha=alpha)
    else:
        matplotlib_utils.plot_sphere(ax, center=soma.center[COLS.XYZ], radius=soma.radius,
                                     color=color, alpha=alpha)

    # unlike w/ 2d Axes, the dataLim isn't set by collections, so it has to be updated manually
    _update_3d_datalim(ax, soma)


@_implicit_ax3d
def plot_morph3d(morph, ax=None, neurite_type=NeuriteType.all,
                 diameter_scale=_DIAMETER_SCALE, linewidth=_LINEWIDTH,
                 color=None, alpha=_ALPHA):
    """Generates a figure of the morphology, that contains a soma and a list of trees.

    Args:
        morph(Morphology): morphology to be plotted
        ax(matplotlib axes): on what to plot
        neurite_type(NeuriteType): an optional filter on the neurite type
        diameter_scale(float): Scale factor multiplied with segment diameters before plotting
        linewidth(float): all segments are plotted with this width, but only if diameter_scale=None
        color(str or None): Color of plotted values, None corresponds to default choice
        alpha(float): Transparency of plotted values
    """
    plot_soma3d(morph.soma, ax, color=color, alpha=alpha)

    for neurite in iter_neurites(morph, filt=tree_type_checker(neurite_type)):
        plot_tree3d(neurite, ax,
                    diameter_scale=diameter_scale, linewidth=linewidth,
                    color=color, alpha=alpha)

    ax.set_title(morph.name)


def _get_dendrogram_legend(dendrogram):
    """Generates labels legend for dendrogram.

    Because dendrogram is rendered as patches, we need to manually label it.
    Args:
        dendrogram (Dendrogram): dendrogram

    Returns:
        List of legend handles.
    """
    def neurite_legend(neurite_type):
        return Line2D([0], [0], color=TREE_COLOR[neurite_type], lw=2, label=neurite_type.name)

    if dendrogram.neurite_type == NeuriteType.soma:
        handles = {d.neurite_type: neurite_legend(d.neurite_type)
                   for d in [dendrogram] + dendrogram.children}
        return handles.values()
    return [neurite_legend(dendrogram.neurite_type)]


def _as_dendrogram_polygon(coords, color):
    return Polygon(coords, color=color, fill=True)


def _as_dendrogram_line(start, end, color):
    return FancyArrowPatch(start, end, arrowstyle='-', color=color, lw=2, shrinkA=0, shrinkB=0)


def _get_dendrogram_shapes(dendrogram, positions, show_diameters):
    """Generates drawable patches for dendrogram.

    Args:
        dendrogram (Dendrogram): dendrogram
        positions (dict of Dendrogram: np.array): positions xy coordinates of dendrograms
        show_diameter (bool): whether to draw shapes with diameter or as plain lines

    Returns:
        List of matplotlib.patches.
    """
    color = TREE_COLOR[dendrogram.neurite_type]
    start_point = positions[dendrogram]
    end_point = start_point + [0, dendrogram.height]
    if show_diameters:
        shapes = [_as_dendrogram_polygon(dendrogram.coords + start_point, color)]
    else:
        shapes = [_as_dendrogram_line(start_point, end_point, color)]
    for child in dendrogram.children:
        shapes.append(_as_dendrogram_line(end_point, positions[child], color))
        shapes += _get_dendrogram_shapes(child, positions, show_diameters)
    return shapes


@_implicit_ax
def plot_dendrogram(obj, ax=None, show_diameters=True):
    """Plots Dendrogram of `obj`.

    Args:
        obj (neurom.Morphology, neurom.Section): morphology or section
        ax: matplotlib axes
        show_diameters (bool): whether to show node diameters or not
    """
    dendrogram = Dendrogram(obj)
    positions = layout_dendrogram(dendrogram, np.array([0, 0]))
    w, h = get_size(positions)
    positions = move_positions(positions, np.array([.5 * w, 0]))
    ax.set_xlim([-.05 * w, 1.05 * w])
    ax.set_ylim([-.05 * h, 1.05 * h])
    ax.set_title('Morphology Dendrogram')
    ax.set_xlabel('micrometers (um)')
    ax.set_ylabel('micrometers (um)')
    shapes = _get_dendrogram_shapes(dendrogram, positions, show_diameters)
    ax.add_collection(PatchCollection(shapes, match_original=True))

    ax.set_aspect('auto')
    ax.legend(handles=_get_dendrogram_legend(dendrogram))
