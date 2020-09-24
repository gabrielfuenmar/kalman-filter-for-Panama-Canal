# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:07:57 2020

@author: snf52211
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from distance_bearing import haversine, bearing
from pykalman import KalmanFilter


def kalman_smooth(vessel):
    """DF input with minimum columns timestamp_position, lon, lat, course, speed, draught
    Kalman filter on existinf positions and interpolation of missing positions in 10 minutes interval
    Output: Df with columns "timestamp_position","lon","lat","speed", "draught """
    if vessel["name"].iloc[0]=="Atlantic Anchorage" or \
        vessel["name"].iloc[0]=="Pacific Anchorage" or vessel["name"].iloc[0]=="Gatun Anchorage":
            freq="10T"
    elif vessel["name"].iloc[0]=="Atlantic Access" or vessel["name"].iloc[0]=="Pacific Access":
            freq="1T"

    if vessel.shape[0]>=6 and (vessel["name"].iloc[0]=="Atlantic Anchorage" or \
        vessel["name"].iloc[0]=="Pacific Anchorage" or vessel["name"].iloc[0]=="Gatun Anchorage" or \
            vessel["name"].iloc[0]=="Atlantic Access" or vessel["name"].iloc[0]=="Pacific Access"):
    
        vessel["timestamp_position"]=pd.to_datetime(vessel["timestamp_position"])
        
        ##Generate columns of calculated values to find variances and mean errors
        vessel=vessel.assign(distance=np.vectorize(haversine)(vessel.lon,vessel.lat,vessel.lon.shift(),vessel.lat.shift()))
    
        vessel=vessel.assign(speed_calc=vessel.distance/((vessel.timestamp_position-vessel.timestamp_position.shift())/np.timedelta64(1,"h")))
        vessel=vessel.assign(distance_cal=abs(vessel.speed*((vessel.timestamp_position-vessel.timestamp_position.shift())/np.timedelta64(1,"h"))),
                             speed_error=vessel.speed-vessel.speed_calc)
        vessel=vessel.assign(distance_error=vessel.distance-vessel.distance_cal)
        
        # ###Estimation Errors
        error_est_x=(vessel.distance_error[(vessel.distance_error>=vessel.distance_error.quantile(0.05))&(vessel.distance_error<=vessel.distance_error.quantile(0.95))].var())/60
        error_est_y=error_est_x
        # ##Speed Error when compared with estimation calculation
        error_est_v=vessel.speed_error[(vessel.speed_error>=vessel.speed_error.quantile(0.05))&(vessel.speed_error<=vessel.speed_error.quantile(0.95))].var()
        
        # ##Observation Error
        error_obs_x=6.423e-6 ##As per NCOSBP User Range Error is of 0.713 m or equivalently 6.423e-6 in decimal degrees-
        error_obs_y=6.423e-6
        error_obs_v=9.8 ##0.18 km h as Al-Gaadi 2005.
        
        ns1min=60*1000000000
        vessel["timestamp_position"]=pd.to_datetime((vessel["timestamp_position"].astype(np.int64)// ns1min + 1 ) * ns1min)
        vessel.drop_duplicates(subset=["timestamp_position"],inplace=True)
        
        vessel.sort_values(by=["timestamp_position"],inplace=True)
        vessel.set_index("timestamp_position",inplace=True)
        
        ##Initial values
        x=vessel.lon.loc[vessel.lon.first_valid_index()]
        y=vessel.lat.loc[vessel.lat.first_valid_index()]
        v=vessel.speed.loc[vessel.speed.first_valid_index()]
        dt=0.1 ##Time difference between positions. Speed on NM/hours(knots)
        
        vessel=vessel.resample("1T").asfreq()
        
        observations=np.ma.masked_invalid(vessel[["lon","lat","speed"]].values)
        
        #Initial Estimation Cov matrix
        P=np.diag([error_est_x,0,error_est_y,0,error_est_v,0])
        A=np.array([[1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]])
        
        ##Observation Cov Matrix
        R=np.diag([error_obs_x,error_obs_y,error_obs_v])
        observation_matrix = [[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]]
        ##Initial State
        X=np.array([x,
            0,
            y,
            0,
            v,
            0])
        
        kf1 = KalmanFilter(transition_matrices = A,
                            transition_covariance=P,
                            observation_matrices = observation_matrix,
                            observation_covariance=R,
                            initial_state_mean = X)
        
        kf1 = kf1.em(observations, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(observations)
        
        smooth_pos=pd.DataFrame(smoothed_state_means,columns=["lon_s","lon_cov","lat_s","lat_cov","speed_s","speed_cov"]).drop(columns=["lon_cov","lat_cov","speed_cov"])
        
        vessel.reset_index(inplace=True)
        
        vessel=pd.concat([vessel,smooth_pos],axis=1)
        
        ##Recovering information of rows that might be removed for not being part of frequency
        try:
            imo=vessel.imo.loc[vessel.imo.first_valid_index()]
            full_transit=vessel.full_transit.loc[vessel.full_transit.first_valid_index()]
            transit_number=vessel.transit_number.loc[vessel.transit_number.first_valid_index()]
            subgroup=vessel.subgroup.loc[vessel.subgroup.first_valid_index()]
            name=vessel.name.loc[vessel.name.first_valid_index()]
        
            vessel["imo"]=imo
            vessel["lon"]=vessel.lon_s
            vessel["lat"]=vessel.lat_s
            vessel["full_transit"]=full_transit
            vessel["transit_number"]=transit_number
            vessel["subgroup"]=subgroup
            vessel["name"]=name

              
            ##Protect last indexed position at anchorage and the first in access
            if vessel["name"].iloc[0]=="Atlantic Anchorage" or \
            vessel["name"].iloc[0]=="Pacific Anchorage":
                protected_position=pd.DataFrame([vessel.loc[vessel["index"].last_valid_index()]])
                protected_position_in=pd.DataFrame([vessel.loc[vessel["index"].first_valid_index()]])
            elif vessel["name"].iloc[0]=="Atlantic Access" or \
            vessel["name"].iloc[0]=="Pacific Access":
                protected_position=pd.DataFrame([vessel.loc[vessel["index"].first_valid_index()]])
                protected_position_in=None
            else:
                protected_position=None
                protected_position_in=None
        
            ##Resample to reduce sample
            vessel.set_index("timestamp_position",inplace=True)   
        
            draught=vessel.draught.loc[vessel.draught.first_valid_index()]
            ##Assign 10 minutes
            vessel=vessel.resample(freq).asfreq().reset_index()
            ##Add back last indexed position
            vessel=pd.concat([vessel,protected_position,protected_position_in]).sort_values(by="timestamp_position")
        
            vessel=vessel[vessel.lon.notnull()].reset_index()
        
            vessel=vessel.assign(distance_to=np.vectorize(haversine)(vessel.lon,vessel.lat,vessel.lon.shift(-1),vessel.lat.shift(-1)))
            vessel["speed"]=vessel.speed_s   
        
            vessel["draught"]=draught
            vessel=vessel[["index","timestamp_position","lon","lat","speed","name","subgroup","draught","imo","full_transit","transit_number"]]  
        except KeyError:
            vessel=vessel[["index","timestamp_position","lon","lat","speed","name","subgroup","draught","imo","full_transit","transit_number"]]
    else:
        
        vessel=vessel[["index","timestamp_position","lon","lat","speed","name","subgroup","draught","imo","full_transit","transit_number"]]  
        
    return vessel
