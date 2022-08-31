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
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Tools to visualize neuron morphological objects.

Examples:
    >>> from neurom import viewer
    >>> m = ... # load a neuron
    >>> viewer.draw(m)                    # 2d plot
    >>> viewer.draw(m, mode='3d')         # 3d plot
    >>> viewer.draw(m.neurites[0])        # 2d plot of neurite tree
    >>> viewer.draw(m, mode='dendrogram') # dendrogram plot
"""

from neurom.view.matplotlib_impl import (plot_morph, plot_morph3d,
                                         plot_tree, plot_tree3d,
                                         plot_soma, plot_soma3d,
                                         plot_dendrogram)
from neurom.view import matplotlib_utils
from neurom.core.morphology import Section, Neurite, Morphology
from neurom.core.soma import Soma
from neurom.utils import deprecated_module
from neurom.io.multiSomas import MultiSoma

deprecated_module('Module `viewer` is deprecated. See the documentation\'s migration page.')

MODES = ('2d', '3d', 'dendrogram')

_VIEWERS = {
    'neuron_3d': plot_morph3d,
    'neuron_2d': plot_morph,
    'neuron_dendrogram': plot_dendrogram,
    'tree_3d': plot_tree3d,
    'tree_2d': plot_tree,
    'tree_dendrogram': plot_dendrogram,
    'soma_3d': plot_soma3d,
    'soma_2d': plot_soma,
    'multi-somas_2d': MultiSoma.drawHulls
}


class ViewerError(Exception):
    """Base class for viewer exceptions."""


class InvalidDrawModeError(ViewerError):
    """Exception class to indicate invalid draw mode."""


class NotDrawableError(Exception):
    """Exception class for things that aren't drawable."""


def draw(obj, mode='2d', **kwargs):
    """Draw a morphology object.

    Arguments:
        obj: morphology object to be drawn (neuron, tree, soma).
        mode (Optional[str]): drawing mode ('2d', '3d', 'dendrogram'). Defaults to '2d'.
        **kwargs: keyword arguments for underlying neurom.view.view functions.

    Raises:
        InvalidDrawModeError if mode is not valid
        NotDrawableError if obj is not drawable
        NotDrawableError if obj type and mode combination is not drawable

    Examples:
        >>> from neurom import viewer, load_morphology
        >>> m = load_morphology('/path/to/morphology') # load a neuron
        >>> fig, _ = viewer.draw(m)             # 2d plot
        >>> fig.show()
        >>> fig3d, _ = viewer.draw(m, mode='3d') # 3d plot
        >>> fig3d.show()
        >>> fig, _ = viewer.draw(m.neurites[0]) # 2d plot of neurite tree
        >>> dend, _ = viewer.draw(m, mode='dendrogram')
    """
    if mode not in MODES:
        raise InvalidDrawModeError('Invalid drawing mode %s' % mode)

    if 'realistic_diameters' in kwargs and mode == '3d':
        if kwargs['realistic_diameters']:
            raise NotImplementedError('Option realistic_diameter not implemented for 3D plots')
        del kwargs['realistic_diameters']

    if 'ax' not in kwargs:
        if mode in ('2d', 'dendrogram'):
            fig, ax = matplotlib_utils.get_figure()
        else:
            fig, ax = matplotlib_utils.get_figure(params={'projection': '3d'})
    else:
        fig = kwargs.pop('fig', None)
        ax =  kwargs.pop('ax', None)   

    if isinstance(obj, Morphology):
        tag = 'neuron'
    elif isinstance(obj, (Section, Neurite)):
        tag = 'tree'
    elif isinstance(obj, Soma):
        tag = 'soma'
    elif isinstance(obj, MultiSoma):
        tag = 'multi-somas'
    else:
        raise NotDrawableError('draw not implemented for %s' % obj.__class__)

    viewer = '%s_%s' % (tag, mode)
    try:
        plotter = _VIEWERS[viewer]
    except KeyError as e:
        raise NotDrawableError('No drawer for class %s, mode=%s' % (obj.__class__, mode)) from e

    output_path = kwargs.pop('output_path', None)
    plotter(obj, ax, **kwargs)

    if mode != 'dendrogram':
        matplotlib_utils.plot_style(fig=fig, ax=ax, **kwargs)

    if output_path:
        matplotlib_utils.save_plot(fig=fig, output_path=output_path, **kwargs)

    return fig, ax
