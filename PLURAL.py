# -*- coding: utf-8 -*-
"""
Created 2021 - 2023
@author: Johannes H. Uhl, University of Colorado Boulder, USA
"""

######## P L U R A L : Place-level urban-rural remoteness indices ########

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gp
from scipy.spatial import Voronoi, Delaunay, distance_matrix
import shapely
from pysal.lib import weights
import scipy.sparse
from sklearn.metrics import auc
from shapely.ops import polygonize, cascaded_union
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt

#########################################################################
### input data specifications ###########################################

input_csv = 'placedata_co.csv' # Small sample dataset. Population counts and locations from NHGIS (http://doi.org/10.18128/D050.V16.0)
universe_polygon = './indata_spatial/co.shp' ## if set to '',then a convex hull will be used for clipping the Thiessen polygons

crs_in = '+proj=longlat +datum=WGS84 +no_defs +type=crs' #CRS of xcol,ycol.
crs_out = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs' #CRS in which distance calculations are made. Must be planar / metric..
xcol = 'lon' #column in input CSV file with x coordinates in crs_in
ycol = 'lat' #column in input CSV file with y coordinates in crs_in
placeid = 'placeid' #column in input CSV file with unique place ID
placename = 'placename' #column in input CSV file with place name
years = [1930,2010] # years for which place-pop data is available
pop_columns = ['placepop1930', 'placepop2010'] # columns with place population for eacn year

if universe_polygon=='':
    alpha=50000000 # alpha parameter for concave hull creation, to be used to clip the Thiessen polygons to the study area. 
    #Needs to be adjusted to the data.

#########################################################################
### Control variables, to be executed in indicated order#################

create_shapefiles = True ### preparation. Converts input data into annual shapefiles.
plural1_create_components = True  ### plural-1. Calculates distance measures and focal population density
create_thiessens = True ### plural-1. Creates Voronoi diagrams for the place points, for visualization and for the spatial network
plural1_compute_index= True ### plural-1. Calculates the PLURAL-1 indices
plural2_generate_contiguity_and_dist_matrices = True ### plural-2. Uses the Voronoi diagrams to calculat spatial weights.
plural2_compute_network_metrics = True ### plural-2. Derives the network based remoteness indicators.
plural2_compute_index = True ### plural-2. Calculates PLURAL-2 indices.

#########################################################################
### PLURAL-1 parameters #################################################

# define distance bands
popcats_lower = [1, 10000, 20000, 50000, 100000, 250000]
popcats_upper = [1000000000, 20000, 50000, 100000, 250000, 1000000000]
focpopdens_radius = 10000  # radius for focal population density in m
# define weights (optional). by default, equal weights are used.
# keep weights_plural1 = [] if no weighting schemes should be used.
weights_plural1 = []
weights_plural1.append([0.25, 0.25, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])#place_centric
weights_plural1.append([0.25, 0.25, 0.03333333, 0.06666667, 0.1, 0.13333333, 0.16666667])#place_and_metro_centric
weights_plural1.append([0.1, 0.1, 0.053333328, 0.106666672, 0.16, 0.213333328, 0.266666672])#metro_centric
# names for each weighting scheme, used in output files.
weights_plural1_scheme_names = ['plural1_place_centric','plural1_place_and_metro_centric', 'plural1_metro_centric']
# highest possible values of place population and focal pop. density for scaling:
globmaxpop = 10000000 # persons
globmaxpopdens = 15000 # persons / sqkm

##########################################################################
### PLURAL-2 parameters  #################################################
# define weights (optional). by default, equal weights are used.
# keep weights_plural2 = [] if no weighting schemes should be used.
weights_plural2 = []
weights_plural2.append([0.125, 0.125, 0.125, 0.125, 1/14.0, 1/14.0,1/14.0, 1/14.0, 1/14.0, 1/14.0, 1/14.0])  # pop focus
weights_plural2.append([1/14.0, 1/14.0, 1/14.0, 1/14.0, 0.125, 0.125,0.125, 0.125, 1/14.0, 1/14.0, 1/14.0])  # DNPI focus
weights_plural2.append([1/14.0, 1/14.0, 1/14.0, 1/14.0, 1/14.0, 1 /14.0, 1/14.0, 1/14.0, 1/6.0, 1/6.0, 1/6.0])  # significance
# names for each weighting scheme, used in output files.
weights_plural2_scheme_names = ['plural2_pop_focus','plural2_DNPI_focus', 'plural2_significance_focus']
maxorder = 3 ## order of cardinalities for contiguity matrix.

##########################################################################

### some helper functions:
def closest_node_dist(node, nodes):
    return int(np.sqrt(np.min(np.sum((nodes - node)**2, axis=1))))

def closest_node_dist_not_self(node, nodes):
    return int(np.sqrt(sorted(np.sum((nodes - node)**2, axis=1))[1]))

def get_focal_popcount(node, nodes, focpopdens_radius, popvals):
    df=pd.DataFrame({'dist_sq': np.sum((nodes - node)**2, axis=1),
                     'popvals': popvals})
    return (df[df.dist_sq<focpopdens_radius*focpopdens_radius].popvals.sum())

def concave_hull(points_gdf,max_circum=50000000):
    ### credit: HTenkanen (https://gist.github.com/HTenkanen/49528990d1ab4bcb5562ba01ba6262ef)
    if len(points_gdf) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return points_gdf.unary_union.convex_hull
    x = points_gdf.geometry.x.values
    y = points_gdf.geometry.y.values
    coords = np.vstack((x, y)).T    
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    #filtered = triangles[circums < (1.0 / alpha)]
    filtered = triangles[circums < max_circum]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return gp.GeoDataFrame({"geometry": [cascaded_union(triangles).buffer(10000)]}, index=[0], crs=points_gdf.crs)    

def getNdegreNeighbors(degree, currcol, placeids):
    neighb_idx_orig = np.where(
        np.logical_and(currcol > 0, currcol <= degree))
    neighb_places = placeids[neighb_idx_orig]
    return neighb_places.copy()

def getRank(value, distribution):
    a = list(distribution)+[value]
    idxs = np.argsort(np.array(a))
    percentile = np.divide(np.argwhere(idxs == distribution.shape[0])[
                           0][0], float(distribution.shape[0]))
    return(percentile)

def comp_auc_2crit(DIST_CRIT, POP_CRIT):
    try:
        plotvectorx = nn_dists_sq_incl_self[nn_dists_sq_incl_self < DIST_CRIT]
        plotvectory = nn_pops_sq_incl_self_cum[nn_dists_sq_incl_self < DIST_CRIT]
        # normalize x vector to 0,1
        plotvectorx = plotvectorx-np.nanmin(plotvectorx)
        plotvectorx = np.divide(plotvectorx, np.nanmax(plotvectorx))
        # normalize x vector to placepop,1
        plotvectory = np.divide(plotvectory, float(POP_CRIT))
        plotvectory[plotvectory > 1.0] = 1.0
        # compute AUC:
        auc_ = auc(plotvectorx, plotvectory)
    except:
        print('%s %s' % (year, placeid), file=open('errors.txt', 'a'))
        auc_ = np.nan
    return auc_    

###############################################################################

##create directories 
dir_shp_harm = './SHP'
dir_voronoi = './VORONOI' ## will hold output data as spatial data
index_files_dir = './DATA'
dir_matrices = './MATRICES'
csvdir = './CSV' ## will hold output data as csv files
plotdir = './PLOT' ## will hold output data as csv files

if not os.path.exists(dir_shp_harm):
    os.mkdir(dir_shp_harm)
if not os.path.exists(dir_voronoi):
    os.mkdir(dir_voronoi)
if not os.path.exists(index_files_dir):
    os.mkdir(index_files_dir)
if not os.path.exists(dir_matrices):
    os.mkdir(dir_matrices)
if not os.path.exists(csvdir):
    os.mkdir(csvdir)
if not os.path.exists(plotdir):
    os.mkdir(plotdir)
    
if create_shapefiles: #reads input csv and create a shp for each year

    allpopdf = pd.read_csv(input_csv, encoding='ISO-8859-1')
    allpopdf = allpopdf[[placeid, placename, xcol, ycol]+pop_columns]
    gdf = gp.GeoDataFrame(allpopdf, geometry=gp.points_from_xy(
        allpopdf[xcol], allpopdf[ycol]))
    gdf.crs = crs_in
    if crs_in!=crs_out:
        gdf.geometry = gdf.geometry.to_crs(crs_out)
    for year in years:
        relpopcol = pop_columns[years.index(year)]
        gdf_curr = gdf[['geometry', placeid, placename, relpopcol]]
        gdf_curr.columns = ['geometry', 'placeid', 'placename', 'totalpop']
        outshp = dir_shp_harm+os.sep+'placepop_harm_%s.shp' % year
        gdf_curr=gdf_curr[gdf_curr.totalpop>0]
        gdf_curr.to_file(filename=outshp)
        print('exported shp', year)

if plural1_create_components: #calculates components for PLURAL-1
    
    focwinarea=  np.pi*focpopdens_radius*focpopdens_radius  
    arearatio = 1000000/focwinarea

    for year in years:
        inshp = dir_shp_harm+os.sep+'placepop_harm_%s.shp' % year
        alldf = gp.read_file(filename=inshp)
        alldf['x']=alldf.geometry.x.map(int)
        alldf['y']=alldf.geometry.y.map(int)

        place_coo_groups=[]
        for popcat_low in popcats_lower:
            popcat_high = popcats_upper[popcats_lower.index(popcat_low)]
            currcoos=alldf[np.logical_and(alldf.totalpop>=popcat_low,alldf.totalpop<popcat_high)][['x','y']]
            place_coo_groups.append(currcoos)

        dist_meas_all_places = []
        allplaces = len(alldf)
        for i, row in alldf.iterrows():
            placeid = row.placeid
            placepop = row.totalpop
            curryear = year

            centr_x = row.x
            centr_y = row.y

            distance_to_closest = np.zeros((len(popcats_lower)))#.astype(np.int64)

            for popcat_low in popcats_lower:
                curridx=popcats_lower.index(popcat_low)
                popcat_high = popcats_upper[curridx]
                currcoos = place_coo_groups[curridx] 
                currnode=np.array([centr_x,centr_y])
                if popcat_low==1:
                    if len(currcoos)==0:
                        mindist=np.nan # dummy distance value if pop category does not exist.
                    else:
                        mindist = closest_node_dist_not_self(currnode,currcoos)
                else:
                    if len(currcoos)==0:
                        mindist=np.nan # dummy distance value if pop category does not exist.
                    else:                    
                        mindist = closest_node_dist(currnode,currcoos)
                distance_to_closest[curridx]=mindist
                #print(year,i,popcat_low,popcat_high,mindist)
                
            currcoos = place_coo_groups[0]
            curr_focpopcount = get_focal_popcount(currnode, currcoos, focpopdens_radius, alldf.totalpop.values)
            focpopdens = np.round(arearatio*curr_focpopcount,3)

            dist_meas_all_current_place = []
            dist_meas_all_current_place.append(placeid)
            dist_meas_all_current_place.append(placepop)
            dist_meas_all_current_place.append(curryear)
            dist_meas_all_current_place.append(focpopdens)
            for xxx in distance_to_closest:
                dist_meas_all_current_place.append(xxx)
            dist_meas_all_places.append(dist_meas_all_current_place)
            print('Calculating PLURAL-1 components',curryear, i, allplaces)

        outdf_plural1 = pd.DataFrame(dist_meas_all_places)
        dfcolnames = []
        dfcolnames.append('placeid')
        dfcolnames.append('placepop')
        dfcolnames.append('year')
        dfcolnames.append('focpopdens')

        for popcat_lower in popcats_lower:
            popcat_upper = popcats_upper[popcats_lower.index(popcat_lower)]
            popcat_lower_str = str(int(popcat_lower/1000.0))+'k'
            popcat_upper_str = str(int(popcat_upper/1000.0))+'k'
            if popcat_lower_str == '0k':
                popcat_lower_str = '0'
            if popcat_upper_str == '1000000k':
                popcat_upper_str = 'any'
            dfcolnames.append('d%s%s' % (popcat_lower_str, popcat_upper_str))

        outdf_plural1.columns = dfcolnames
        alldf = alldf.merge(outdf_plural1, on='placeid', how='left')
        alldf.to_file(filename=inshp.replace('_harm_', '_harm_w_distances_'))
        print(year)

if create_thiessens:

    for year in years:
        inshp = dir_shp_harm+os.sep+'placepop_harm_w_distances_%s.shp' % year
        outshp = dir_voronoi+os.sep+'placepop_harm_w_distances_voronoi_%s.shp' % year

        points_gdf = gp.read_file(inshp)

        ## code snippet and advice from https://gis.stackexchange.com/questions/337561/making-polygon-for-every-point-in-set-using-voronoi-diagram
        x = points_gdf.geometry.x.values
        y = points_gdf.geometry.y.values
        coords = np.vstack((x, y)).T
        
        if universe_polygon=='':
            hulldf = concave_hull(points_gdf,max_circum=alpha)
            hulldf.crs = crs_out
        else:
            hulldf = gp.read_file(universe_polygon)
            if not hulldf.crs == crs_out:
                hulldf=hulldf.to_crs(crs_out)
        
        #hullx,hully = hulldf.geometry[0].exterior.coords.xy
        #hull_coords = np.dstack((hullx,hully))[0]               
        #coords = np.vstack((coords,hull_coords))
        
        vor = Voronoi(coords)        
        lines = [shapely.geometry.LineString(vor.vertices[line]) for line in 
            vor.ridge_vertices if -1 not in line]
        polys = shapely.ops.polygonize(lines)
        voronois = gp.GeoDataFrame(geometry=gp.GeoSeries(polys))
        voronois.crs = crs_out
        voronois = gp.clip(voronois,hulldf,keep_geom_type=True)                
        voronois.plot()
        #voronoi_joineddf=gp.sjoin(voronois,points_gdf,how='left')
        voronoi_joineddf=gp.sjoin(voronois,points_gdf,how='left')
        voronoi_joineddf.plot(column='d10k20k')

        
        if voronoi_joineddf['placeid'].duplicated().any():
            voronoi_joineddf['area']=voronoi_joineddf.geometry.area
            voronoi_joineddf=voronoi_joineddf.sort_values(by='area').reset_index()
            voronoi_joineddf=voronoi_joineddf.drop_duplicates(subset='placeid', keep='last', ignore_index=False)

        voronoi_joineddf.to_file(outshp)
        voronoi_joineddf.plot()
        
        print('voronoi polygons created', year)

if plural1_compute_index:

    incols_orig = ['totalpop', 'focpopdens']
    
    for popcat_lower in popcats_lower[1:]:
        popcat_upper = popcats_upper[popcats_lower.index(popcat_lower)]
        popcat_lower_str = str(int(popcat_lower/1000.0))+'k'
        popcat_upper_str = str(int(popcat_upper/1000.0))+'k'
        if popcat_lower_str == '0k':
            popcat_lower_str = '0'
        if popcat_upper_str == '1000000k':
            popcat_upper_str = 'any'
        colname = 'd%s%s' % (popcat_lower_str, popcat_upper_str)
        incols_orig.append(colname)    

    incols = [x+'_norm' for x in incols_orig]

    gdf = gp.GeoDataFrame()
    for year in years:  # :
        inshp = dir_voronoi+os.sep+'placepop_harm_w_distances_voronoi_%s.shp' % year
        currgdf = gp.read_file(filename=inshp)
        currgdf['year'] = year
        gdf = gdf.append(currgdf)
        print('read', year)

    gdf[incols_orig] = gdf[incols_orig].replace(-9999, np.nan)
    gdf[incols_orig] = gdf[incols_orig].fillna(0)

    columns_init = gdf.columns

    for col in gdf.columns:

        if col[0] == 'd':
            # set 0 to a low number (1) to avoid data gaps and inf values when inverting
            gdf.loc[gdf[col] == 0, col] = 1
            gdf[col+'_norm'] = np.log(gdf[col].values+1)

            # also adjust the distances: if a category is non-zero, all higher categories must be non-zero too
            # (e.g., a place x kilometers from a place in cat 10k-20k, can not have <x in dist to place in cat 50k100k)
            dist_colnames = []
            for popcat_lower in popcats_lower[1:]:
                popcat_upper = popcats_upper[popcats_lower.index(popcat_lower)]
                popcat_lower_str = str(int(popcat_lower/1000.0))+'k'
                popcat_upper_str = str(int(popcat_upper/1000.0))+'k'
                if popcat_lower_str == '0k':
                    popcat_lower_str = '0'
                if popcat_upper_str == '1000000k':
                    popcat_upper_str = 'any'
                colname = 'd%s%s' % (popcat_lower_str, popcat_upper_str)
                dist_colnames.append(colname)

            for distcol in dist_colnames[1:]:
                distcol_prev = dist_colnames[dist_colnames.index(distcol)-1]
                relrows = gdf[distcol]-gdf[distcol_prev] < 0
                gdf.loc[relrows][distcol_prev] = gdf.loc[relrows][distcol]
                print('adjusted  distance', year, distcol)

        if 'totalpop' in col:
            # set 0 to a low number (1) to avoid data gaps and inf values when inverting
            gdf.loc[gdf[col] == 0, col] = 1
            gdf[col+'_norm'] = np.log10(globmaxpop)-np.log10(gdf[col].values+1)

        if 'focpopdens' in col:
            # set 0 to a low number (1) to avoid data gaps and inf values when inverting
            gdf.loc[gdf[col] == 0, col] = 1
            gdf[col+'_norm'] = np.log10(globmaxpopdens) - \
                np.log10(gdf[col].values+1)  # pop inversion

    gdf['plural1_eq_weights'] = np.nanmean(gdf[incols].values, axis=1)
    weights_plural1_scheme_names = ['plural1_eq_weights']+weights_plural1_scheme_names
    weigh_avg_cols = []
    for weight_set in weights_plural1:
        weighted_sum = np.zeros((len(gdf)))
        for i in np.arange(0, len(incols)):
            weighted_sum += weight_set[i]*gdf[incols[i]].values
        #weight_set_str = '_'.join([str(x) for x in weight_set])
        weight_set_str = weights_plural1_scheme_names[weights_plural1.index(weight_set)+1]

        #wei_avg_col = 'plural1_weighted_%s' % weight_set_str
        wei_avg_col = weight_set_str
        gdf[wei_avg_col] = weighted_sum
        weigh_avg_cols.append(wei_avg_col)

    outcols_unscaled = ['plural1_eq_weights']+weigh_avg_cols
    outcols_scaled = ['%s_rescaled' % x for x in outcols_unscaled]

    for outcol in outcols_scaled:
        incol = outcols_unscaled[outcols_scaled.index(outcol)]
        currmin = np.nanmin(gdf[incol].values)
        currmax = np.nanmax(gdf[incol].values)
        gdf[outcol] = np.divide(gdf[incol].values-currmin, currmax-currmin)
        print('allyrs', np.nanmin(gdf[outcol]), np.nanmax(gdf[outcol]))

    for year, yeardf in gdf.groupby('year'):
        inshp = dir_voronoi+os.sep+'placepop_harm_w_distances_voronoi_%s.shp' % year
        outshp = inshp.replace(
            '.shp', '_w_plural1_scaled_across_years.gpkg')
        expdf = yeardf[list(columns_init)+outcols_scaled]
        expdf.columns = list(columns_init)+weights_plural1_scheme_names
        expdf.crs=crs_out
        expdf.to_file(filename=outshp, driver='GPKG')
        for plotcol in weights_plural1_scheme_names:
            fig,ax=plt.subplots()
            expdf.plot(column=plotcol,cmap='turbo_r',ax=ax, vmin=0,vmax=1)
            plt.title('PLURAL-1, %s %s, scaled across years' %(year,plotcol),fontsize=8)
            plt.show() 
            fig.savefig(plotdir+os.sep+'PLURAL1_%s_scaled_across_years_%s.png' %(plotcol,year),dpi=300)
        outcsv = csvdir+os.sep+'plural1_scaled_across_years_%s.csv' % year
        expdf.drop(labels=['geometry'], axis=1).to_csv(outcsv)

        # rescale per year and export:
        for outcol in outcols_scaled:
            incol = outcols_unscaled[outcols_scaled.index(outcol)]
            currmin = np.nanmin(yeardf[incol].values)
            currmax = np.nanmax(yeardf[incol].values)
            yeardf[outcol] = np.divide(
                yeardf[incol].values-currmin, currmax-currmin)
            print(year, np.nanmin(yeardf[outcol]), np.nanmax(yeardf[outcol]))

        outshp = inshp.replace(
            '.shp', '_w_plural1_scaled_per_year.gpkg')
        expdf = yeardf[list(columns_init)+outcols_scaled]
        expdf.columns = list(columns_init)+weights_plural1_scheme_names
        expdf.crs=crs_out
        expdf.to_file(filename=outshp, driver='GPKG')
        for plotcol in weights_plural1_scheme_names:
            fig,ax=plt.subplots()
            expdf.plot(column=plotcol,cmap='turbo_r',ax=ax, vmin=0,vmax=1)
            plt.title('PLURAL-1, %s %s, scaled per year' %(year,plotcol),fontsize=8)
            plt.show() 
            fig.savefig(plotdir+os.sep+'PLURAL1_%s_scaled_per_year_%s.png' %(plotcol,year),dpi=300)
        outcsv = csvdir+os.sep+'plural1_scaled_per_year_%s.csv' % year
        expdf.drop(labels=['geometry'], axis=1).to_csv(outcsv)
        print(year)


#########################################################################

if plural2_generate_contiguity_and_dist_matrices:

    generate_contmat = True
    generate_distmat = True

    for year in years:

        if generate_contmat:
            # read voronoi polygons
            voro_shp = dir_voronoi+os.sep + \
                'placepop_harm_w_distances_voronoi_%s_w_plural1%s.gpkg' % (
                    year, '_scaled_per_year')
            # create contiguency matrix
            plcentr_df = gp.read_file(voro_shp)
            
            plcentr_df.placeid=plcentr_df.placeid.map(str)
            
            w = weights.contiguity.Queen.from_dataframe(
                plcentr_df, idVariable='placeid')
            idordered = np.array(w.id_order)
            # idordered = np.array([int(str(x)[1:]) for x in list(idordered) if 'G' in str(x)]) ## remove the G

            np.savez(dir_matrices+os.sep+'contiguity_matrix_ids_ordered_%s_%s.npz' %
                     (year, maxorder), idordered)

            #wknn5.transform = 'r'
            w_total = np.zeros((len(w.cardinalities), len(w.cardinalities)))
            for order in range(1, maxorder+1):
                wcurr = weights.higher_order_sp(w, order)
                w_total = w_total + order*wcurr.full()[0]
                print(year, order)
            cont_matr = w_total.astype(np.uint8)  # w25.full()[0]
            cont_matr_sparse = scipy.sparse.csr_matrix(cont_matr)
            scipy.sparse.save_npz(
                dir_matrices+os.sep+'contiguity_matrix_sparse_%s_%s.npz' % (year, maxorder), cont_matr_sparse)
            print('generated contiguity matrix %s %s' % (year, maxorder))

        ######################################################################################################
        if generate_distmat:

            # read voronoi polygons
            voro_shp = dir_voronoi+os.sep + \
                'placepop_harm_w_distances_voronoi_%s_w_plural1%s.gpkg' % (
                    year, '_scaled_per_year')
            plcentr_df = gp.read_file(voro_shp)
            # xs=plcentr_df.x.values
            # ys=plcentr_df.y.values
            #stackedcoo = np.array(list(zip(list(xs),list(ys))))
            #distmat = distance_matrix(stackedcoo,stackedcoo)
            plcentr_df.placeid=plcentr_df.placeid.map(str)

            distmat = weights.distance.distance_matrix(
                plcentr_df[['x', 'y']].values, plcentr_df[['x', 'y']].values)
            print('generating distance matrix %s' % (year))

            distmat = distmat.astype(np.int64)
            distmat_sparse = scipy.sparse.csr_matrix(distmat)
            scipy.sparse.save_npz(
                dir_matrices+os.sep+'distance_matrix_sparse_%s.npz' % (year), distmat_sparse)
            print('generated distance matrix %s' % (year))


if plural2_compute_network_metrics:

    for year in years:

        # load data:
        voro_shp = dir_voronoi+os.sep + \
            'placepop_harm_w_distances_voronoi_%s_w_plural1%s.gpkg' % (
                year, '_scaled_per_year')
        plpoint_df = gp.read_file(voro_shp)
        distmat_file = dir_matrices+os.sep + \
            'distance_matrix_sparse_%s.npz' % (year)
        contmat_file = dir_matrices+os.sep + \
            'contiguity_matrix_sparse_%s_%s.npz' % (year, maxorder)
        cont_matr_sparse = scipy.sparse.load_npz(contmat_file)
        cont_matr = cont_matr_sparse.toarray()
        dist_matr_sparse = scipy.sparse.load_npz(distmat_file)
        dist_matr_all = dist_matr_sparse.toarray()

        plpoint_df['pop_percentile_global'] = plpoint_df.totalpop.rank(
            pct=True)
        plpoint_df['focpopdens_percentile_global'] = plpoint_df.focpopdens.rank(
            pct=True)

        ### loop places ###########
        # to link to distances.
        plpoint_df['index_orig'] = np.arange(0, len(plpoint_df))
        plpoint_df = plpoint_df.sort_values(by='placeid').reset_index()
        placeids = plpoint_df.placeid.values
        pops = plpoint_df.totalpop.values
        xs = plpoint_df.x.values
        ys = plpoint_df.y.values
        origidx = np.arange(0, cont_matr.shape[0])

        # we need to resort the distance matrix to make it consistent to the contiguity matrix:
        dist_matr_all = dist_matr_all[plpoint_df['index_orig'].values, :]
        dist_matr_all = dist_matr_all[:, plpoint_df['index_orig'].values]

        allplaces = len(plpoint_df)
        counter = 0
        errcounter = 0

        maxpop = np.nanmax(pops)
        maxdist = np.nanmax(dist_matr_all)

        OUTDATADF = []
        for i, row in plpoint_df.iterrows():
            counter += 1
            # print(counter,allplaces)

            # get attributes
            placeid = row.placeid
            placepop = row.totalpop

            # get nearest neighbors in topological space
            currcol = cont_matr[i, :]
            neighb_idx_orig = np.where(currcol > 0)
            neighb_degrees = currcol[currcol > 0]
            neighb_idx = origidx[neighb_idx_orig]
            neighb_places = placeids[neighb_idx]

            # get euc dists of neighbors and all places
            currplaceids = placeids
            currplacedists = dist_matr_all[i, :]
            neighb_dists = currplacedists[neighb_idx]

            # find out what is the max distance to a Nth degree neighbor.
            euc_dist_col = dist_matr_all[i, :]
            rel_eucdists = euc_dist_col[currcol > 0]
            try:
                max_rel_eucdist = np.nanmax(rel_eucdists)
            except:
                print('%s %s' % (year, placeid), file=open('errors.txt', 'a'))
                continue

            # get KNNs
            # nn_idx = currplacedists.argsort()[1:num_nns+1] #KNN
            # get all NNs except the place itself
            nn_idx = currplacedists.argsort()[1:]
            # get all NNs including the place itself
            nn_idx_incl_place = currplacedists.argsort()

            currpops = pops
            nn_pops_sq = currpops[nn_idx]
            nn_dists_sq = currplacedists[nn_idx]

            # percentiles, global and regional
            #
            P_GLOB = row.pop_percentile_global
            #
            ######## percentiles, local
            neigh_ids_deg1 = getNdegreNeighbors(1, currcol, placeids)
            neigh_ids_deg2 = getNdegreNeighbors(2, currcol, placeids)
            neigh_ids_deg3 = getNdegreNeighbors(3, currcol, placeids)

            pops_ndeg1 = pops[np.isin(placeids, neigh_ids_deg1)]
            pops_ndeg2 = pops[np.isin(placeids, neigh_ids_deg2)]
            pops_ndeg3 = pops[np.isin(placeids, neigh_ids_deg3)]

            pops_ndeg1_df = pd.DataFrame()
            pops_ndeg1_df['pop'] = pops_ndeg1
            pops_ndeg1_df['placeid'] = neigh_ids_deg1

            pops_ndeg2_df = pd.DataFrame()
            pops_ndeg2_df['pop'] = pops_ndeg2
            pops_ndeg2_df['placeid'] = neigh_ids_deg2

            pops_ndeg3_df = pd.DataFrame()
            pops_ndeg3_df['pop'] = pops_ndeg3
            pops_ndeg3_df['placeid'] = neigh_ids_deg3

            P_LOC1 = getRank(placepop, pops_ndeg1)
            P_LOC2 = getRank(placepop, pops_ndeg2)
            P_LOC3 = getRank(placepop, pops_ndeg3)

            # pop sums, local ABS
            TOTPOP_LOC1 = np.sum(pops_ndeg1)+placepop
            TOTPOP_LOC2 = np.sum(pops_ndeg2)+placepop
            TOTPOP_LOC3 = np.sum(pops_ndeg3)+placepop
            #
            allpops_deg1 = np.append(pops_ndeg1, placepop)
            allpops_deg2 = np.append(pops_ndeg2, placepop)
            allpops_deg3 = np.append(pops_ndeg3, placepop)

            # pop means, local ABS
            #
            AVG_POP_LOC1 = np.nanmean(allpops_deg1)
            AVG_POP_LOC2 = np.nanmean(allpops_deg2)
            AVG_POP_LOC3 = np.nanmean(allpops_deg3)
            #
            # pop medians, local ABS
            #
            MED_POP_LOC1 = np.nanmedian(allpops_deg1)
            MED_POP_LOC2 = np.nanmedian(allpops_deg2)
            MED_POP_LOC3 = np.nanmedian(allpops_deg3)

            # nndist mean, local ABS
            #
            currplacedists_1order_neighb = currplacedists[np.isin(
                currplaceids, neigh_ids_deg1)]
            currplacedists_2order_neighb = currplacedists[np.isin(
                currplaceids, neigh_ids_deg2)]
            currplacedists_3order_neighb = currplacedists[np.isin(
                currplaceids, neigh_ids_deg3)]

            tempdf1 = pd.DataFrame()
            tempdf1['placeid'] = currplaceids[np.isin(
                currplaceids, neigh_ids_deg1)]
            tempdf1['dist'] = currplacedists[np.isin(
                currplaceids, neigh_ids_deg1)]

            tempdf2 = pd.DataFrame()
            tempdf2['placeid'] = currplaceids[np.isin(
                currplaceids, neigh_ids_deg2)]
            tempdf2['dist'] = currplacedists[np.isin(
                currplaceids, neigh_ids_deg2)]

            tempdf3 = pd.DataFrame()
            tempdf3['placeid'] = currplaceids[np.isin(
                currplaceids, neigh_ids_deg3)]
            tempdf3['dist'] = currplacedists[np.isin(
                currplaceids, neigh_ids_deg3)]

            pops_ndeg1_df = pops_ndeg1_df.merge(
                tempdf1, on='placeid', how='left')
            pops_ndeg2_df = pops_ndeg2_df.merge(
                tempdf2, on='placeid', how='left')
            pops_ndeg3_df = pops_ndeg3_df.merge(
                tempdf3, on='placeid', how='left')

            AVG_NN_DIST_LOC1 = np.nanmean(currplacedists_1order_neighb)
            AVG_NN_DIST_LOC2 = np.nanmean(currplacedists_2order_neighb)
            AVG_NN_DIST_LOC3 = np.nanmean(currplacedists_3order_neighb)
            #
            # nndist median, local ABS
            #
            MED_NN_DIST_LOC1 = np.nanmedian(currplacedists_1order_neighb)
            MED_NN_DIST_LOC2 = np.nanmedian(currplacedists_2order_neighb)
            MED_NN_DIST_LOC3 = np.nanmedian(currplacedists_3order_neighb)
            #
            # nndist max, local ABS
            #
            MAX_NN_DIST_LOC1 = np.nanmax(currplacedists_1order_neighb)
            MAX_NN_DIST_LOC2 = np.nanmax(currplacedists_2order_neighb)
            MAX_NN_DIST_LOC3 = np.nanmax(currplacedists_3order_neighb)

            # Esch et al. 2012, local significance
            all_edges_locsig1 = np.divide(np.multiply(
                placepop, pops_ndeg1_df['pop'].values), np.square(pops_ndeg1_df.dist.values))
            all_edges_locsig2 = np.divide(np.multiply(
                placepop, pops_ndeg2_df['pop'].values), np.square(pops_ndeg2_df.dist.values))
            all_edges_locsig3 = np.divide(np.multiply(
                placepop, pops_ndeg3_df['pop'].values), np.square(pops_ndeg3_df.dist.values))

            sum_edges_locsig1 = np.nansum(
                all_edges_locsig1)  # esch - edge strength
            mean_edges_locsig1 = np.nanmean(
                all_edges_locsig1)  # esch - edge robustness
            median_edges_locsig1 = np.nanmedian(all_edges_locsig1)
            min_edges_locsig1 = np.nanmin(all_edges_locsig1)
            max_edges_locsig1 = np.nanmax(all_edges_locsig1)

            sum_edges_locsig2 = np.nansum(
                all_edges_locsig2)  # esch - edge strength
            mean_edges_locsig2 = np.nanmean(
                all_edges_locsig2)  # esch - edge robustness
            median_edges_locsig2 = np.nanmedian(all_edges_locsig2)
            min_edges_locsig2 = np.nanmin(all_edges_locsig2)
            max_edges_locsig2 = np.nanmax(all_edges_locsig2)

            sum_edges_locsig3 = np.nansum(
                all_edges_locsig3)  # esch - edge strength
            mean_edges_locsig3 = np.nanmean(
                all_edges_locsig3)  # esch - edge robustness
            median_edges_locsig3 = np.nanmedian(all_edges_locsig3)
            min_edges_locsig3 = np.nanmin(all_edges_locsig3)
            max_edges_locsig3 = np.nanmax(all_edges_locsig3)

            df_nn3 = plpoint_df[plpoint_df.placeid.isin(neigh_ids_deg3)]
            nn3xs = df_nn3.x.values
            nn3ys = df_nn3.y.values
            nn3ids = df_nn3.placeid.values
            stackedcoo_nn3 = np.array(list(zip(list(nn3xs), list(nn3ys))))
            distmat_nn3 = distance_matrix(stackedcoo_nn3, stackedcoo_nn3)
            distmat_nn3[distmat_nn3 == 0] = np.nanmax(distmat_nn3)+1
            avg_nndist_nn3 = np.nanmean(np.nanmin(distmat_nn3, axis=0))
            med_nndist_nn3 = np.nanmedian(np.nanmin(distmat_nn3, axis=0))

            # AUC based measures and distance to X measures

            # get ordered, cumulative pop vector, within sampling square
            nn_pops_sq_incl_self = currpops[nn_idx_incl_place]
            nn_pops_sq_incl_self_cum = np.cumsum(nn_pops_sq_incl_self)

            # get ordered distance vector, within sampling square
            nn_dists_sq_incl_self = currplacedists[nn_idx_incl_place]

            # different pop and distance thresholds up to which we evaluate

            # within neighborhood,up to neighborhood max pops.
            DIST_CRIT = MAX_NN_DIST_LOC3
            POP_CRIT = TOTPOP_LOC3
            auc_MAX_NN_DIST_LOC3__TOTPOP_LOC3 = comp_auc_2crit(
                DIST_CRIT, POP_CRIT)
            dist2pop_idx = nn_pops_sq_incl_self_cum[nn_pops_sq_incl_self_cum <=
                                                    POP_CRIT].shape[0]-1
            dist2pop = nn_dists_sq_incl_self[dist2pop_idx]
            DIST_2_TOTPOP_LOC3 = dist2pop

            # fixed values
            DIST_CRIT = 250000
            POP_CRIT = 500000
            auc_dist250000_pop500000 = comp_auc_2crit(DIST_CRIT, POP_CRIT)
            dist2pop_idx = nn_pops_sq_incl_self_cum[nn_pops_sq_incl_self_cum <=
                                                    POP_CRIT].shape[0]-1
            dist2pop = nn_dists_sq_incl_self[dist2pop_idx]
            DIST_2_500000 = dist2pop

            DIST_CRIT = 500000
            POP_CRIT = 1000000
            auc_dist500000_pop1000000 = comp_auc_2crit(DIST_CRIT, POP_CRIT)
            dist2pop_idx = nn_pops_sq_incl_self_cum[nn_pops_sq_incl_self_cum <=
                                                    POP_CRIT].shape[0]-1
            dist2pop = nn_dists_sq_incl_self[dist2pop_idx]
            DIST_2_1000000 = dist2pop

            # max values
            DIST_CRIT = maxdist
            POP_CRIT = maxpop
            auc_distmaxdist_popmaxpop = comp_auc_2crit(DIST_CRIT, POP_CRIT)
            dist2pop_idx = nn_pops_sq_incl_self_cum[nn_pops_sq_incl_self_cum <=
                                                    POP_CRIT].shape[0]-1
            dist2pop = nn_dists_sq_incl_self[dist2pop_idx]
            DIST_2_maxpop = dist2pop

            OUTDATA = []
            OUTDATA.append(placeid)
            OUTDATA.append(placepop)
            OUTDATA.append(P_GLOB)
            OUTDATA.append(P_LOC1)
            OUTDATA.append(P_LOC2)
            OUTDATA.append(P_LOC3)
            OUTDATA.append(TOTPOP_LOC1)
            OUTDATA.append(TOTPOP_LOC2)
            OUTDATA.append(TOTPOP_LOC3)
            OUTDATA.append(AVG_POP_LOC1)
            OUTDATA.append(AVG_POP_LOC2)
            OUTDATA.append(AVG_POP_LOC3)
            OUTDATA.append(MED_POP_LOC1)
            OUTDATA.append(MED_POP_LOC2)
            OUTDATA.append(MED_POP_LOC3)
            OUTDATA.append(AVG_NN_DIST_LOC1)
            OUTDATA.append(AVG_NN_DIST_LOC2)
            OUTDATA.append(AVG_NN_DIST_LOC3)
            OUTDATA.append(MED_NN_DIST_LOC1)
            OUTDATA.append(MED_NN_DIST_LOC2)
            OUTDATA.append(MED_NN_DIST_LOC3)
            OUTDATA.append(MAX_NN_DIST_LOC1)
            OUTDATA.append(MAX_NN_DIST_LOC2)
            OUTDATA.append(MAX_NN_DIST_LOC3)
            OUTDATA.append(auc_MAX_NN_DIST_LOC3__TOTPOP_LOC3)
            OUTDATA.append(DIST_2_TOTPOP_LOC3)
            OUTDATA.append(auc_dist250000_pop500000)
            OUTDATA.append(DIST_2_500000)
            OUTDATA.append(auc_dist500000_pop1000000)
            OUTDATA.append(DIST_2_1000000)
            OUTDATA.append(auc_distmaxdist_popmaxpop)
            OUTDATA.append(DIST_2_maxpop)
            OUTDATA.append(sum_edges_locsig1)
            OUTDATA.append(mean_edges_locsig1)
            OUTDATA.append(median_edges_locsig1)
            OUTDATA.append(min_edges_locsig1)
            OUTDATA.append(max_edges_locsig1)
            OUTDATA.append(sum_edges_locsig2)
            OUTDATA.append(mean_edges_locsig2)
            OUTDATA.append(median_edges_locsig2)
            OUTDATA.append(min_edges_locsig2)
            OUTDATA.append(max_edges_locsig2)
            OUTDATA.append(sum_edges_locsig3)
            OUTDATA.append(mean_edges_locsig3)
            OUTDATA.append(median_edges_locsig3)
            OUTDATA.append(min_edges_locsig3)
            OUTDATA.append(max_edges_locsig3)
            OUTDATADF.append(OUTDATA)

            print('Calculating spatial network metrics', year, counter, allplaces)

        OUTDF = pd.DataFrame(OUTDATADF)
        OUTDATA_COLUMNS = []
        OUTDATA_COLUMNS.append('placeid')
        OUTDATA_COLUMNS.append('placepop')
        OUTDATA_COLUMNS.append('P_GLOB')
        OUTDATA_COLUMNS.append('P_LOC1')
        OUTDATA_COLUMNS.append('P_LOC2')
        OUTDATA_COLUMNS.append('P_LOC3')
        OUTDATA_COLUMNS.append('TOTPOP_LOC1')
        OUTDATA_COLUMNS.append('TOTPOP_LOC2')
        OUTDATA_COLUMNS.append('TOTPOP_LOC3')
        OUTDATA_COLUMNS.append('AVG_POP_LOC1')
        OUTDATA_COLUMNS.append('AVG_POP_LOC2')
        OUTDATA_COLUMNS.append('AVG_POP_LOC3')
        OUTDATA_COLUMNS.append('MED_POP_LOC1')
        OUTDATA_COLUMNS.append('MED_POP_LOC2')
        OUTDATA_COLUMNS.append('MED_POP_LOC3')
        OUTDATA_COLUMNS.append('AVG_NN_DIST_LOC1')
        OUTDATA_COLUMNS.append('AVG_NN_DIST_LOC2')
        OUTDATA_COLUMNS.append('AVG_NN_DIST_LOC3')
        OUTDATA_COLUMNS.append('MED_NN_DIST_LOC1')
        OUTDATA_COLUMNS.append('MED_NN_DIST_LOC2')
        OUTDATA_COLUMNS.append('MED_NN_DIST_LOC3')
        OUTDATA_COLUMNS.append('MAX_NN_DIST_LOC1')
        OUTDATA_COLUMNS.append('MAX_NN_DIST_LOC2')
        OUTDATA_COLUMNS.append('MAX_NN_DIST_LOC3')
        OUTDATA_COLUMNS.append('auc_MAX_NN_DIST_LOC3__TOTPOP_LOC3')
        OUTDATA_COLUMNS.append('DIST_2_TOTPOP_LOC3')
        OUTDATA_COLUMNS.append('auc_dist250000_pop500000')
        OUTDATA_COLUMNS.append('DIST_2_500000')
        OUTDATA_COLUMNS.append('auc_dist500000_pop1000000')
        OUTDATA_COLUMNS.append('DIST_2_1000000')
        OUTDATA_COLUMNS.append('auc_distmaxdist_popmaxpop')
        OUTDATA_COLUMNS.append('DIST_2_maxpop')
        OUTDATA_COLUMNS.append('sum_edges_locsig1')
        OUTDATA_COLUMNS.append('mean_edges_locsig1')
        OUTDATA_COLUMNS.append('median_edges_locsig1')
        OUTDATA_COLUMNS.append('min_edges_locsig1')
        OUTDATA_COLUMNS.append('max_edges_locsig1')
        OUTDATA_COLUMNS.append('sum_edges_locsig2')
        OUTDATA_COLUMNS.append('mean_edges_locsig2')
        OUTDATA_COLUMNS.append('median_edges_locsig2')
        OUTDATA_COLUMNS.append('min_edges_locsig2')
        OUTDATA_COLUMNS.append('max_edges_locsig2')
        OUTDATA_COLUMNS.append('sum_edges_locsig3')
        OUTDATA_COLUMNS.append('mean_edges_locsig3')
        OUTDATA_COLUMNS.append('median_edges_locsig3')
        OUTDATA_COLUMNS.append('min_edges_locsig3')
        OUTDATA_COLUMNS.append('max_edges_locsig3')
        OUTDF.columns = OUTDATA_COLUMNS
        OUTDF.to_csv(index_files_dir+os.sep + 'plural2_components_%s.csv' % year, index=False)


if plural2_compute_index:

    # the boolean attribute: if true, variable is inverted
    adv_index_columns = []
    adv_index_columns.append(['totalpop', True])
    adv_index_columns.append(['TOTPOP_LOC1_dens', True])
    adv_index_columns.append(['TOTPOP_LOC2_dens', True])
    adv_index_columns.append(['TOTPOP_LOC3_dens', True])
    adv_index_columns.append(['auc_MAX_NN_DIST_LOC3__TOTPOP_LOC3', True])
    adv_index_columns.append(['auc_dist250000_pop500000', True])
    adv_index_columns.append(['auc_dist500000_pop1000000', True])
    adv_index_columns.append(['auc_distmaxdist_popmaxpop', True])
    adv_index_columns.append(['median_edges_locsig1', True])
    adv_index_columns.append(['median_edges_locsig2', True])
    adv_index_columns.append(['median_edges_locsig3', True])

    incols_orig = [x[0] for x in adv_index_columns]

    incols = [x+'_norm' for x in incols_orig]

    gdf = gp.GeoDataFrame()
    for year in years:  # :
        inshp = dir_voronoi+os.sep+'placepop_harm_w_distances_voronoi_%s.shp' % year
        currgdf = gp.read_file(filename=inshp)
        currgdf['year'] = year
        try:
            currgdf.placeid = currgdf.placeid.map(int)
        except:
            pass
        currcsv = index_files_dir+os.sep+'plural2_components_%s.csv' % year
        curr_adv_idx_df = pd.read_csv(currcsv)
        currgdf = currgdf.merge(curr_adv_idx_df, on='placeid', how='left')
        gdf = gdf.append(currgdf)
        print('read', year)

    gdf['TOTPOP_LOC1_dens'] = np.divide(
        gdf.TOTPOP_LOC1.values, (gdf.MAX_NN_DIST_LOC1.values**2).astype(np.float))
    gdf['TOTPOP_LOC2_dens'] = np.divide(
        gdf.TOTPOP_LOC2.values, (gdf.MAX_NN_DIST_LOC2.values**2).astype(np.float))
    gdf['TOTPOP_LOC3_dens'] = np.divide(
        gdf.TOTPOP_LOC3.values, (gdf.MAX_NN_DIST_LOC3.values**2).astype(np.float))

    gdf[incols_orig] = gdf[incols_orig].replace(-9999, np.nan)
    gdf[incols_orig] = gdf[incols_orig].fillna(0)

    columns_init = gdf.columns

    gdf[incols_orig] = gdf[incols_orig].replace(
        np.inf, np.nan).replace(-np.inf, np.nan).fillna(0)

    # convert to ranks, and invert if necessary:
    for col in adv_index_columns:
        colname = col[0]
        invert = col[1]
        gdf[colname +
            '_norm'] = gdf[colname].rank(pct=False, ascending=not invert)

    gdf['plural2_eq_weights'] = np.nanmean(gdf[incols].values, axis=1)
    weights_plural2_scheme_names = ['plural2_eq_weights']+weights_plural2_scheme_names
    weigh_avg_cols = []
    for weight_set in weights_plural2:
        weighted_sum = np.zeros((len(gdf)))
        for i in np.arange(0, len(incols)):
            weighted_sum += weight_set[i]*gdf[incols[i]].values
        #weight_set_str = '_'.join([str(x) for x in weight_set])
        weight_set_str = weights_plural2_scheme_names[weights_plural2.index(weight_set)+1]

        #wei_avg_col = 'plural2_weighted_%s' % weight_set_str
        wei_avg_col = weight_set_str
        gdf[wei_avg_col] = weighted_sum
        weigh_avg_cols.append(wei_avg_col)

    outcols_unscaled = ['plural2_eq_weights']+weigh_avg_cols
    outcols_scaled = ['%s_rescaled' % x for x in outcols_unscaled]

    for outcol in outcols_scaled:
        incol = outcols_unscaled[outcols_scaled.index(outcol)]
        currmin = np.nanmin(gdf[incol].values)
        currmax = np.nanmax(gdf[incol].values)
        gdf[outcol] = np.divide(gdf[incol].values-currmin, currmax-currmin)
        print('allyrs', np.nanmin(gdf[outcol]), np.nanmax(gdf[outcol]))

    for year, yeardf in gdf.groupby('year'):
        inshp = dir_voronoi+os.sep+'placepop_harm_w_distances_voronoi_%s.shp' % year
        outshp = inshp.replace(
            '.shp', '_w_plural2_scaled_across_years.gpkg')
        expdf = yeardf[list(columns_init)+outcols_scaled]
        expdf.columns = list(columns_init)+weights_plural2_scheme_names
        expdf.crs=crs_out
        expdf.to_file(filename=outshp, driver='GPKG')
        for plotcol in weights_plural2_scheme_names:
            fig,ax=plt.subplots()
            expdf.plot(column=plotcol,cmap='turbo_r',ax=ax, vmin=0,vmax=1)
            plt.title('PLURAL-2, %s %s, scaled across years' %(year,plotcol),fontsize=8)
            plt.show()     
            fig.savefig(plotdir+os.sep+'PLURAL2_%s_scaled_across_years_%s.png' %(plotcol,year),dpi=300)
        outcsv = csvdir+os.sep+'plural2_scaled_across_years_%s.csv' % year
        expdf.drop(labels=['geometry'], axis=1).to_csv(outcsv)

        # rescale per year and export:
        for outcol in outcols_scaled:
            incol = outcols_unscaled[outcols_scaled.index(outcol)]
            currmin = np.nanmin(yeardf[incol].values)
            currmax = np.nanmax(yeardf[incol].values)
            yeardf[outcol] = np.divide(
                yeardf[incol].values-currmin, currmax-currmin)
            print(year, np.nanmin(yeardf[outcol]), np.nanmax(yeardf[outcol]))

        outshp = inshp.replace(
            '.shp', '_w_plural2_scaled_per_year.gpkg')
        expdf = yeardf[list(columns_init)+outcols_scaled]
        expdf.columns = list(columns_init)+weights_plural2_scheme_names
        expdf.crs=crs_out
        expdf.to_file(filename=outshp, driver='GPKG')
        for plotcol in weights_plural2_scheme_names:
            fig,ax=plt.subplots()
            expdf.plot(column=plotcol,cmap='turbo_r',ax=ax, vmin=0,vmax=1)
            plt.title('PLURAL-2, %s %s, scaled per year' %(year,plotcol),fontsize=8)
            plt.show()
            fig.savefig(plotdir+os.sep+'PLURAL2_%s_scaled_per_year_%s.png' %(plotcol,year),dpi=300)
        outcsv = csvdir+os.sep+'plural2_scaled_per_year_%s.csv' % year
        expdf.drop(labels=['geometry'], axis=1).to_csv(outcsv)
        print(year)


    
    
    