import numpy as np
import pandas as pd
import neurom as morphor_nm
from neurom import features
from neurom import morphmath as mm
from neurom.core.types import NeuriteType
from neurom.core.morphology import Section
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

def getSomaStats(ps):
    ''' Basic statistics for soma
    ps: pointset from a neuron object. Type: 2-D numpy array
    '''
    maxDia = minmaxDist(ps)
    ps_center = np.nanmean(ps, axis=0)
    ps_radius = np.sqrt(np.nansum((ps - ps_center) ** 2, axis=1))
    ps_avgRadius = np.nanmean(ps_radius)
    return maxDia, ps_center, ps_radius, ps_avgRadius

def getAreaFromHull(p):
    '''Shoelace algorithm for area of simple polygon
    Input: point set as 2D numpy array. Must be in order like Convex Hull
    Output: area for 
    '''     
    x = np.asanyarray(p[:,0])
    y = np.asanyarray(p[:,1])
    n = len(x)
    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)    
    return (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0

def getPerimeter_Area(ps):
    ps2  = ps.copy()
    ps2 = ps2[:,:2] ## make sure we get 2D array
    try:
        c_hull = ConvexHull(ps2)
        area, perimeter =  c_hull.volume,c_hull.area
    except:
        print('Hull alg. failed! return None for area and perimeter for the soma')
        area, perimeter =  np.nan, np.nan
    return area, perimeter

def getSomaCircularityIndex(area, perimeter):
    ''' Calculate roundness of a soma contour 
    return a float value between (0,1.0). 1 being a perfect circle
    '''
    if area is not np.nan and perimeter is not np.nan:
        return  np.round((4 * np.pi * area) / (perimeter * perimeter), 2)
    else:
        return np.nan

def getShapeFactors(ps_):
    ''' 
    Giving soma points,return a bunch of shape factors in one go
    circularity, max_diameter, shape_factor as defined in neurom.morphmath
    '''
    ps = ps_[:,:2].copy()
    try:
        hull = ConvexHull(ps)
        cirIndex = 4.0 * np.pi * hull.volume / hull.area**2 #circularity
        hull_points = ps[hull.vertices]
        max_pairwise_distance = np.max(cdist(hull_points, hull_points))
        shapefactor= hull.volume / max_pairwise_distance**2
    except:
        cirIndex, max_pairwise_distance, shapefactor  = np.nan, np.nan, np.nan
    try:
        aspRatio = mm.aspect_ratio(ps)
    except:
        aspRatio = np.nan
    return cirIndex, max_pairwise_distance, shapefactor,aspRatio

def poly_centroid_poly(poly):
    poly2 = poly[:,:2].copy()  ## only works for 2D
    T = Delaunay(poly2).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0
    for m in range(n):
        sp = poly2[T[m, :], :]
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)
    C = C / np.sum(W)
    cx, cy = C[0], C[1]
    poly[:,0] -= cx
    poly[:,1] -= cy
    return poly, cx, cy

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

def min_max_trunk_angle(n, neurteType = NeuriteType.basal_dendrite):
    angles = morphor_nm.features.morphology.trunk_angles(n, neurite_type=neurteType, consecutive_only=False)
    if len(angles)>1:
        # print(n.name, angles)
        nmax = []
        nmin = []
        for x in angles:
            if 0 in x:
                x.remove(0)
            nmax.append(max(x))
            nmin.append(min(x))
        try:
            ad_min = min(nmin)
            ad_max = max(nmax)
        except:
            ad_min = np.nan
            ad_max = np.nan    
    else:
        ad_min = np.nan
        ad_max = np.nan    
    return ad_min, ad_max, angles

def dendrites_dispersion_index(ad_min, ad_max, angles):
    '''dendrites_dispersion_index'''
    ## calculate dendrite directions dispersion
    ## ad_max: maximum angle between any two dendrites, in (0, pi)
    ## ad_min: minimum angle between any two dendrites, in (0, pi)
    ## dispersion index: (ad_max-ad_min) / (ad_max+ad_min) (0, 1). 0 being total overlap.
    if ad_max!=np.nan and ad_min!=np.nan:
        if ad_max==ad_min:
            ad_max = np.pi ## two dendrites only
            return 1-(ad_max-ad_min) / (ad_max+ad_min)
    else:
        return np.nan

def sholl_analysis(n, step_size=10):
    ''' sholl analysis
    n: NeuroM neuron object with neurites
    step_size: step size for analysis
    output: frequency and bins (in um)
    '''
    freq = features.morphology.sholl_frequency(n, neurite_type=NeuriteType.all, step_size=step_size, bins=None)
    bins =  list(n.soma.radius + np.arange(len(freq))*step_size)
    return freq, bins

def extractMorhporFeatures(n, df_summary=None):
    ''' return a dictionary contains useful morphor features
    n: NeuroM Neuron object
    '''
    if df_summary is None:
        df_summary = {}
    maxDia, soma_center, soma_radius, soma_avgRadius = getSomaStats(n.soma.points)
    df_summary["Neuron id"] = n.name
    # df_summary["center X"] = soma_center[0]
    # df_summary["center Y"] = soma_center[1]
    # df_summary["center Z"] = soma_center[2]
    cellArea, cellPerimeter = getPerimeter_Area(n.soma.points)
    cirIndex, max_pairwise_distance, shapefactor, asratio = getShapeFactors(n.soma.points)
    df_summary["soma average radius"] = soma_avgRadius
    df_summary["soma maximal radius"] = np.max(soma_radius)
    df_summary["soma minimal radius"] = np.min(soma_radius)
    df_summary['soma max_pairwise_dist'] = max_pairwise_distance
    df_summary["soma perimeter"] = cellPerimeter
    df_summary["soma area"] = cellArea
    df_summary['soma circularity index'] = cirIndex
    df_summary['soma shape factor'] = shapefactor
    df_summary['soma aspect_ratio'] = asratio
    df_summary['cell max_radial_dist'] = features.morphology.max_radial_distance(n)
    df_summary['total number of neurites'] = len(n.neurites)
    if len(n.neurites) > 0:
        for neurite in n.neurites:
            nsections = features.morphology.number_of_sections_per_neurite(n, neurite.type)
            neuriteTrunkLength = morphor_nm.features.morphology.trunk_section_lengths(n, neurite_type=neurite.type)
            neutriteName = str(neurite.type).split('.')[-1]
            df_summary[neutriteName+ ' Nseg'] = len(nsections)
            if len(nsections) > 1 :
                for i in range(len(nsections)):
                    df_summary[neutriteName+ ' seg'+str(i+1)+' Nsec'] = nsections[i]
                    df_summary[neutriteName+ ' seg'+str(i+1)+' trunkLen'] = neuriteTrunkLength[i] # stem length
                
            else:
                df_summary[neutriteName+' Nsec'] = nsections[0]
                df_summary[neutriteName+' trunkLen'] = neuriteTrunkLength[0] # stem length
        # neurite features
        neurite_funcs = [total_neurite_length,total_neurite_volume,total_neurite_area,\
            total_neurite_area,total_bifurcation_points,
            max_branch_order]
        for f in neurite_funcs:
            df_summary[f.__doc__] = f(n)

    admin, admax, angles = min_max_trunk_angle(n)
    df_summary['trunk angle min'] = admin
    df_summary['trunk angle max'] = admax
    df_summary['trunk angle dispersion index'] = dendrites_dispersion_index(admin, admax, angles)
    return df_summary