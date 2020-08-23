import numpy as np
import cupy
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import cv2


# functions
def rnd(x):
	if type(x) is np.ndarray:
		return (x+0.5).astype(np.int)
	else:
		return round(x)


# r = [lower, upper]
def limit_values(x, r, xcopy=1):
	if xcopy == 1:
		ret = x.copy()
	else:
		ret = x
	ret[ret<r[0]] = r[0]
	ret[ret>r[1]] = r[1]
	return ret


# omni-drectional image
class OmniImage:
	def __init__(self, omni_image):
		if type(omni_image) is str:
			self.omni_image = plt.imread(omni_image)
		else:
			self.omni_image = omni_image
	
	def extract(self, camera_prm):
		# 2d-cordinates in omni-directional image [c2, r2]
		c2 = (camera_prm.polar_omni_cord[0] / (2.0 * np.pi) + 1.0 / 2.0) * self.omni_image.shape[1] - 0.5
		r2 = (-camera_prm.polar_omni_cord[1] / np.pi + 1.0/2.0) * self.omni_image.shape[0] - 0.5
		#[c2_int, r2_int] = [rnd(c2), rnd(r2)]
		c2_int = limit_values(rnd(c2), (0, self.omni_image.shape[1]-1), 0)
		r2_int = limit_values(rnd(r2), (0, self.omni_image.shape[0]-1), 0)
		#self.omni_cord = [c2, r2]
		return self.omni_image[r2_int, c2_int]

# SphereNet
class omni_filter:
	# omni_size, filter_size: [height, width]
	def __init__(self, omni_size, filter_size=[3,3], interpolation='bilinear'):
		self.omni_size = omni_size
		self.filter_size = filter_size
		
		# filter size in 3D space
		[wf, hf] = [(filter_size[1]-1)*np.tan(2*np.pi/omni_size[1]), (filter_size[0]-1)*np.tan(np.pi/omni_size[0])]
		
		# filter locations in 2D space
		[c_filter, r_filter] = np.meshgrid(np.arange(filter_size[1]), np.arange(filter_size[0]))
		x_filter = (c_filter - (filter_size[1] - 1)/2) / (filter_size[1] - 1) * wf
		y_filter = (r_filter - (filter_size[0] - 1)/2) / (filter_size[0] - 1) * hf
		xp = np.reshape(x_filter, (1,filter_size[1]*filter_size[0],1,1))
		yp = np.reshape(y_filter, (1,filter_size[1]*filter_size[0],1,1))
		
		# camera directions
		[c_omni, r_omni] = np.meshgrid(np.arange(omni_size[1]), np.arange(omni_size[0]))
		theta_c = 2 * np.pi * c_omni / (omni_size[1] - 1) - np.pi
		phi_c  =np.pi / 2.0 - np. pi *r_omni / (omni_size[0] - 1)
		# filter locations in 3D space
		xn = np.reshape(np.array([
				-np.sin(theta_c), 
				-np.cos(theta_c),
				np.zeros(theta_c.shape)
			]), (3,1, theta_c.shape[0], theta_c.shape[1]))
		yn = np.reshape(np.array([
				-np.sin(phi_c) * np.cos(theta_c), 
				np.sin(phi_c) * np.sin(theta_c),
				np.cos(phi_c)
			]), (3,1, theta_c.shape[0], theta_c.shape[1]))
		c0 = np.reshape(np.array([
				np.cos(phi_c) * np.cos(theta_c), 
				-np.cos(phi_c) * np.sin(theta_c), 
				np.sin(phi_c)
			]), (3,1, theta_c.shape[0], theta_c.shape[1]))
		
		p = xp * xn + yp * yn + c0
		
		# polar cordinates in 3D space
		norm_p = np.sqrt(p[0, :, :, :]**2 + p[1, :, :, :]**2 + p[2, :, :, :]**2) 
		phi = np.arcsin(p[2, :, :, :] / norm_p)
		theta_positive = np.arccos(p[0, :, :, :] / np.sqrt(p[0,:,:,:]**2+p[1,:,:,:]**2))
		theta_negative = - theta_positive
		theta = (p[1,:,:,:] > 0) * theta_negative + (p[1,:,:,:] <= 0) * theta_positive

		# 2d-cordinates in omni-directional image [c2, r2]
		c2 = (theta / (2.0 * np.pi) + 0.5) * omni_size[1] - 0.5
		r2 = (-phi / np.pi + 0.5) * omni_size[0] - 0.5

		self.x = limit_values(rnd(c2), (0, omni_size[1]-1), 0)
		self.y = limit_values(rnd(r2), (0, omni_size[0]-1), 0)
  