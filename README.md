# Kalman Filter for Panama Canal AIS positions interpolation

Kalman filter used as interpolation of missing positions at the Panama Canal, and frequency adjustments every 10 mins. 
Kalman filter done in 1 min frequency and adjusted after interpolation to 10 mins.

Dependencies:

    numpy 1.17.2
    shapely 1.6.4
    pandas  0.25.1
    pykalman 0.9.2
    distance_bearing supplied along with kalman filter code

Parameters:
    
    vessel: iterator with AIS positions from prefiltered positions belonging to Panama Canal shapeBox
    iteration:  5 iterations. No convergence measurement nor converged to optimal. Pending
    

Returns:
  
    dataframe with interpolated AIS positions with a frequency of 10 minutes from each other.
