import numpy as np
import pandas as pd
import neurom as morphor_nm
from neurom import features
from neurom import morphmath as mm
from neurom.core.types import NeuriteType
from neurom.core.morphology import Section

def getSomaStats(ps):
    ''' Basic statistics for soma
    ps: pointset from a neuron object. Type: 2-D numpy array
    '''
    maxDia = minmaxDist(ps)
    ps_center = np.nanmean(ps, axis=0)
    ps_radius = np.sqrt(np.nansum((ps - ps_center) ** 2, axis=1))
    ps_avgRadius = np.nanmean(ps_radius)
    return maxDia, ps_center, ps_radius, ps_avgRadius

def minmaxDist(ps):
    """calculate minimal and maximal diameter
    ps: pointset from a neuron object. Type: 2-D numpy array
    """
    nPoints = len(ps)
    # minDia = 1000
    maxDia = 0
    for j in range(nPoints - 1):
        for k in range(j + 1, nPoints):
            diameter = np.sqrt(np.sum((ps[j] - ps[k]) ** 2))
            # if diameter < minDia:
            #     minDia = diameter
            if diameter > maxDia:
                maxDia = diameter
    return maxDia

def sec_len(n,sec):
    """Return the length of a section."""
    return mm.section_length(sec.points)

def total_neurite_length(n):
    '''Total neurite length (sections)'''
    return sum(sec_len(n, s) for s in morphor_nm.iter_sections(n))

def total_neurite_volume(n):
    '''Total neurite volume'''
    return sum(mm.segment_volume(s) for s in morphor_nm.iter_segments(n))

def total_neurite_area(n):
    '''Total neurite area'''
    return sum(mm.segment_area(s) for s in morphor_nm.iter_segments(n))

def total_bifurcation_points(n):
    '''Number of bifurcation points'''
    return sum(1 for _ in morphor_nm.iter_sections(n,
                                          iterator_type=Section.ibifurcation_point))
def max_branch_order(n):
    '''Maximum branch order'''
    return  max(features.section.branch_order(s) for s in morphor_nm.iter_sections(n))

def sholl_analysis(n, step_size=10):
    freq = features.morphology.sholl_frequency(n, neurite_type=NeuriteType.all, step_size=step_size, bins=None)
    bins =  list(n.soma.radius + np.arange(len(freq))*step_size)
    return freq, bins

def extractMorhporFeatures(n, df_summary=None):
    ''' return a dataframe (formated as record) contains useful features
    n: NeuroM Neuron object
    '''
    if df_summary is None:
        df_summary = {}
    maxDia, soma_center, soma_radius, soma_avgRadius = getSomaStats(n.soma.points)
    df_summary["Neuron id"] = [n.name]
    df_summary["center X"] = [soma_center[0]]
    df_summary["center Y"] = [soma_center[1]]
    df_summary["center Z"] = [soma_center[2]]
    df_summary["average radius"] = [soma_avgRadius]
    df_summary["maximal radius"] = [np.max(soma_radius)]
    df_summary["minimal radius"] = [np.min(soma_radius)]
    df_summary["maximal diameter"] = [maxDia]
    df_summary['max_radial_distance'] = [features.morphology.max_radial_distance(n)]
    for neurite in n.neurites:
        nsections = features.morphology.number_of_sections_per_neurite(n, neurite.type)
        if len(nsections) > 1 :
            for i in range(len(nsections)):
                df_summary[str(neurite.type).split('.')[-1]+' segment'+str(i+1)] = [nsections[i]]
        else:
            df_summary[str(neurite.type)] = [nsections]
    # neurite features
    neurite_funcs = [total_neurite_length,total_neurite_volume,total_neurite_area,\
        total_neurite_area,total_bifurcation_points,
        max_branch_order]
    for f in neurite_funcs:
        df_summary[f.__doc__] = [f(n)]
    return df_summary