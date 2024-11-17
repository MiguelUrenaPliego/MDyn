# -*- coding: utf-8 -*-
"""
Created by Alfredo Camara Casado
For extra capabilities of the code, code development, and bug reporting please contact the author at: acamara@ciccp.es
"""

from MDyn_solver import *
from MDyn_utils import *

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
#from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
#import time
#import os
from subprocess import call
from bisect import bisect_left

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm



############## VIBRATING SYSTEM MODELS: STATE-SPACE DEFINITION ##############

def SDOFUnsprungMass(x,t,Mv,Cv,Kv,zg,zgdot):

	#State variables
	z,zdot = x

	# 2 first order equations
	d_z_dt = zdot
	d_zdot_dt = -(Kv/Mv)*(z-zg)-(Cv/Mv)*(zdot-zgdot)

	# Write the states in matrix form
	states = [d_z_dt,d_zdot_dt]

	return states

#############################################################################


class Tmds():

	def __init__(self, **kwargs):
		#self.t = kwargs.get("t")
		self.DOFactive = kwargs.get("DOFactive")
		self.NumberOfNodes = kwargs.get("NumberOfNodes")
		self.NodeNumber = kwargs.get("NodeNumber")
		self.TMDNodes = kwargs.get("TMDNodes")
		self.TMDDirection = kwargs.get("TMDDirection")
		self.TMDMass = kwargs.get("TMDMass")
		self.TMDStiffness = kwargs.get("TMDStiffness")
		self.TMDDamping = kwargs.get("TMDDamping")

		# Index of the active DOF in which the TMD is applied
		self.indexDOFTMD = np.zeros(len(self.TMDNodes))
		self.indexNodeTMD = np.zeros(len(self.TMDNodes))
		for i in range(len(self.TMDNodes)):
			self.indexDOFTMD[i] = self.DOFactive.index(self.TMDDirection[i])
			self.indexNodeTMD[i] = np.where(self.NodeNumber==self.TMDNodes[i])[0][0]

		self.indexDOFTMD = np.array(self.indexDOFTMD).astype(int)
		self.indexNodeTMD = np.array(self.indexNodeTMD).astype(int)

#		sdamper = np.zeros((len(DOFModelTMD),len(t)))	# Movement of the TMDs; Rows TMD; Cols. time
#		sdamperdot = np.zeros((len(DOFModelTMD),len(t)))	# Velocity of the TMDs; Rows TMD; Cols. time
#		sdamper2dot = np.zeros((len(DOFModelTMD),len(t)))	# Acceleration of the TMDs; Rows TMD; Cols. time

	def TMDresponse(self,r,rdot,sdamper,sdamperdot,ts):

		P = np.zeros(self.NumberOfNodes*len(self.DOFactive))

		for itmd in range(len(self.TMDNodes)): # Loop in TMD
			# Movement of the Structure at the position of the TMD
			rStructure = r[self.indexNodeTMD[itmd]+(self.indexDOFTMD[itmd]*self.NumberOfNodes)] 		# Movement of the structure in the node in which the TMD is located
			rdotStructure = rdot[self.indexNodeTMD[itmd]+(self.indexDOFTMD[itmd]*self.NumberOfNodes)] 	# Velocity of the structure in the node in which the TMD is located
			# Movement of the TMD
			i0ls = np.zeros(2)
			i0ls[0] = sdamper[itmd]
			i0ls[1] = sdamperdot[itmd]
			#ts = [t[i-1],t[i]]
			dti = ts[1] - ts[0]
			qlint = odeint(SDOFUnsprungMass,i0ls,ts,args=(self.TMDMass[itmd],self.TMDDamping[itmd],self.TMDStiffness[itmd],rStructure,rdotStructure,),mxstep=5000000)
			sdamper[itmd] = qlint[-1,0]
			sdamperdot[itmd] = qlint[-1,1]
			#sdamper2dot[itmd,i] = (sdamperdot[itmd,i]-sdamperdot[itmd,i-1])/dti
			# Definition of the loads and moments in the structure
			Ftmd = self.TMDStiffness[itmd]*(sdamper[itmd]-rStructure) + self.TMDDamping[itmd]*(sdamperdot[itmd]-rdotStructure)		# To do !!!! add force and moment due to self weight of the TMD in the corresponding direction !!!! Add moment due to eccentricity. Currently the TMD is assumed to be attached in the centreline of the model at the node of the TMD
			P[self.indexNodeTMD[itmd]+(self.indexDOFTMD[itmd]*self.NumberOfNodes)] = Ftmd

		return P,sdamper,sdamperdot
