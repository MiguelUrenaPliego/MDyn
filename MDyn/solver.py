# -*- coding: utf-8 -*-
"""
Created by Alfredo Camara Casado
For extra capabilities of the code, code development, and bug reporting please contact the author at: acamara@ciccp.es
"""

import numpy as np
#from scipy.interpolate import interp1d
from scipy.integrate import odeint
#import time
import os
from subprocess import call

# --------------- Solution - Modal dynamics

# MDOF dynamics integration

def InitialiseMDOFNewmark(Pn,u,udot,beta,gamma,dt,M,C,K):

	# Non-iterative Newmark method
	# Page 167 Chopra A.K. "Dynamics of Structures"

	# Pn: Modal force vector (Number of modes x 1) for i time step;
	# u: Modal displacement vector (Number of modes x 1) from i time step;
	# udot: Modal velocity vector (Number of modes x 1) from i time step;
	# u2dot: Acceleration vector (Number of modes x 1) from i time step;
	# beta,gamma: Time integration parameters;
	# dt: Time step;
	# M:  Modal mass vector (Number of modes x 1), unit vector if phi normalised wrt mass;
	# C:  Modal damping vector (Number of modes x Number of modes), 2*xi*w for each mode;
	# K:  Modal stifness vector (Number of modes x Number of modes), w**2 for each mode;

	tempv = Pn - np.multiply(C,udot) - np.multiply(K,u)

	u2dot = np.multiply((1./M),tempv)

	kbar=K+(float(gamma)/(beta*dt))*C+(1./(beta*(dt**2)))*M

	a=(1./(beta*dt))*M+(float(gamma)/beta)*C
	b=(1./(2*beta))*M+dt*((float(gamma)/(2*beta))-1)*C

	return u2dot,kbar,a,b

def MDOFNewmark(Pn,Pn1,u,udot,u2dot,beta,gamma,dt,kbar,a,b):

	# Pn: Modal force vector (Number of modes x 1) for i time step;
	# Pn1: Modal force vector (Number of modes x 1) for i+1 time step;
	# u: Modal displacement vector (Number of modes x 1) from i time step;
	# udot: Modal velocity vector (Number of modes x 1) from i time step;
	# u2dot: Acceleration vector (Number of modes x 1) from i time step;
	# beta,gamma: Time integration parameters;
	# dt: Time step;

	dpbar=(Pn1-Pn)+np.multiply(a,udot)+np.multiply(b,u2dot)
	du=np.multiply((1./kbar),dpbar)		# Increment of modal displacement (Number of modes x 1) in i+1 time step
	dudot=(float(gamma)/(beta*dt))*du-(float(gamma)/beta)*udot+dt*(1-(float(gamma)/(2*beta)))*u2dot
	du2dot=(1./(beta*(dt**2)))*du-(1./(beta*dt))*udot-(1./(2*beta))*u2dot

	u1 = u + du
	udot1 = udot + dudot
	u2dot1 = u2dot + du2dot

	return u1,udot1,u2dot1


def IntialiseMDOFNewmarkStatic(Pn,u,udot,beta,gamma,dt,M,C,K):

	# Non-iterative Newmark method
	# Page 167 Chopra A.K. "Dynamics of Structures"

	# Pn: Modal force vector (Number of modes x 1) for i time step;
	# u: Modal displacement vector (Number of modes x 1) from i time step;
	# udot: Modal velocity vector (Number of modes x 1) from i time step;
	# u2dot: Acceleration vector (Number of modes x 1) from i time step;
	# beta,gamma: Time integration parameters;
	# dt: Time step;
	# M:  Modal mass vector (Number of modes x 1), unit vector if phi normalised wrt mass;
	# C:  Modal damping vector (Number of modes x Number of modes), 2*xi*w for each mode;
	# K:  Modal stifness vector (Number of modes x Number of modes), w**2 for each mode;

	tempv = Pn # - np.multiply(C,udot) - np.multiply(K,u)

	#u2dot = 0 # np.multiply((1./M),tempv)

	kbar=K #+(float(gamma)/(beta*dt))*C+(1./(beta*(dt**2)))*M

	a=(1./(beta*dt))*M #+(float(gamma)/beta)*C
	b=(1./(2*beta))*M #+dt*((float(gamma)/(2*beta))-1)*C

	return kbar,a,b

def MDOFNewmarkStatic(Pn,Pn1,u,beta,gamma,dt,kbar,a,b):

	# Pn: Modal force vector (Number of modes x 1) for i time step;
	# Pn1: Modal force vector (Number of modes x 1) for i+1 time step;
	# u: Modal displacement vector (Number of modes x 1) from i time step;
	# udot: Modal velocity vector (Number of modes x 1) from i time step;
	# u2dot: Acceleration vector (Number of modes x 1) from i time step;
	# beta,gamma: Time integration parameters;
	# dt: Time step;

	dpbar=(Pn1-Pn)
	du=np.multiply((1./kbar),dpbar)		# Increment of modal displacement (Number of modes x 1) in i+1 time step
	u1 = u + du

	return u1

def QuasiStaticSol(Pn,wnv):
	print(Pn,wnv)
	u1 = Pn/wnv
	return u1

def structuralResponse(u,udot,u2dot,PhiReduced):

	r = np.dot(PhiReduced,u)		# Structural movement
	r = r[:,0]

	rdot = np.dot(PhiReduced,udot)		# Structural velocity
	rdot = rdot[:,0]

	r2dot = np.dot(PhiReduced,u2dot)	# Structural acceleration
	r2dot = r2dot[:,0]

	return r,rdot,r2dot

def structuralResponseStatic(u,PhiReduced):

	r = np.dot(PhiReduced,u)		# Structural movement
	r = r[:,0]

	return r
