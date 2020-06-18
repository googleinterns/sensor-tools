import pickle
import numpy as np
from skimage.util.shape import view_as_windows
import statistics

# make it usable for all rows and make one function you call first for phase 1 and 2 then a second main function for phase 3

def create_windows(matrix, rows, window_size, stride, mod_num):
  x = matrix['x']
  y = matrix['y']
  z = matrix['z']
  nanos = matrix['nanos'] 
  np_data = np.array([x,y,z,nanos], 'd')

  windowed_data = view_as_windows(np_data, (rows, window_size), step=stride)
  return windowed_data

def find_initial_lift_times(lift_windows, rows):
  start = lift_windows[0][rows-1][0]
  end = lift_windows[-1][-1][-1]
  return start, end

def find_lift_windows(windowed_data):
  variance=[]
  lift_windows =[]
  windows = windowed_data[0]
  for i in range(len(windows)):
    x = windows[i][0]
    y = windows[i][1]
    z = windows[i][2]
    nanos = windows[i][3]
    var_total = statistics.variance(x) + statistics.variance(y) + statistics.variance(z)
    variance.append(var_total)
  threshold = max(variance)/4
  for i in range(len(windows)): 
     if variance[i] >= threshold:
       lift_windows.append(windows[i])
       if (i != len(variance)-1) and (variance[i+1] < threshold):
         break
  return lift_windows

def start_time_on_smaller_windows(windowed_data, threshold):
  start_time =[]
  windows = windowed_data[0]
  for i in range(len(windows)):
    x = windows[i][0]
    y = windows[i][1]
    z = windows[i][2]
    nanos = windows[i][3]
    var_total = statistics.variance(x) + statistics.variance(y) + statistics.variance(z)
    if var_total > threshold:
      start_time.append(windows[i])
  if (len(start_time) < 1):
    return "None"
  return start_time[0][3][0]

def find_precise_start_time(lift_windows, rows, window_size, stride, variance_threshold):
  front = lift_windows[0]
  np_front = np.array(front)
  windowed_data = view_as_windows(np_front, (rows, window_size), step=stride)
  start_time = start_time_on_smaller_windows(windowed_data, variance_threshold)
  return start_time

def initial_find_lift(sample, rows, window_size, stride):
  windowed_data = create_windows(sample, rows, window_size, stride)
  lift_windows = find_lift_windows(windowed_data)
  return lift_windows