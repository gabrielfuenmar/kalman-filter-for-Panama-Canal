# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 23:14:22 2020

@author: gabri
"""

import math
import numpy as np

def haversine(lon1,lat1,lon2,lat2):
    """Nautical miles calculation between positions"""
    r=6378137 ##WGS84 earth radius in km
     
    phi1,phi2=math.radians (lat1), math.radians(lat2)
    dphi=math.radians (lat2-lat1)
    dlambda=math.radians(lon2-lon1)
    
    a=math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
     
    return (r*(2*math.atan2(math.sqrt(a),math.sqrt(1-a))))/1852 ##Transformed to NM

def bearing(lon1,lat1,lon2,lat2):
    dL=math.radians(lon2-lon1)
    
    lat1=math.radians(lat1)
    lat2=math.radians(lat2)
    
    x=math.sin(dL)*math.cos(lat2)
    y=math.cos(lat1)*math.sin(lat2)-(math.sin(lat1)*math.cos(lat2)*math.cos(dL))

    initial_bearing=math.degrees(math.atan2(x,y))
    
    compass_bearing = (initial_bearing + 360) % 360
    
    return compass_bearing

def euclidean(lon1,lat1,lon2,lat2):
    point_a=np.array((lon1,lat1))
    point_b=np.array((lon2,lat2))
    
    distance = np.linalg.norm(point_a - point_b)
    
    return distance

