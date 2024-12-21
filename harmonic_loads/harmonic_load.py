"""
Script that uses the Python Library MDyn for Modal Superposition Analysis
to solve the response of a bridge under stepping pedestrian loads without
human-structure interaction

Modal dynamic solver, cite as:
Camara A (2021). A fast mode superposition algorithm and its application to the analysis of bridges under moving loads,
Advances in Engineering Software, 151: 102934.

Pedestrian effects, cite as:
Fouli M, Camara A (2024) Human–structure interaction effects on lightweight footbridges with tuned mass dampers.
Structures, 62: 106263.
"""

import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append('..\code')
from MDyn_forcing import *
from MDyn_utils import *
from MDyn_solver import *
from MDyn_vibsystems import *



#from matplotlib import pyplot as plt
import time
t = time.time()

if __name__ == '__main__':

    ######################## ADD YOUR DATA HERE #########################
    caseName = 'results'	# Name of the folder that will contain the results (please don't add spaces and don't start it with a number)
    modesToInclude = range(1,9+1)  # [1,2,7,9] # List of modes to be included in the calculations
    DOFactive = [1,2,3,4,5,6]  # Interesting DOF of the problem, 1 is Ux, 2 is Uy, 3 is Uz, 4 is URx, 5 is URy, 6 is URz
    # Newmark analysis
    beta=1./4
    gamma=1./2

    ################# READ BUILDING INFORMATION FILES

    # Load modal information from mmodal analysis
    NodeInformationFile = './structureData/NodeInformation.txt'
    ModalInformationFile = './structureData/ModalInformation.txt'
    FrequencyInformationFile = './structureData/FrequencyInformation.txt'
    BeamInformationFile = './structureData/BeamInformation.txt'


    # DEFINE harmonic loading
    PHarmonicLoad = [1e3]	# Magnitude of load in N. 1D array with as many entries as harmonic loads
    wHarmonicLoad = [2*np.pi*6]	# Circular frequency of load in rad/s. 1D array with as many entries as harmonic loads
    NodeHarmonicLoad = [2]	# Node number where the load is applied. 1D array with as many entries as harmonic loads
    DofHarmonicLoad = [3]	# DOF of load. 1D array with as many entries as harmonic loads

    tmax = 3.    # Total calculation time. Keep this to have it 25% larger than the time it takes for a single load to cross the bridge if it starts at the left abutment
    dt = 1./500 # Step time in s.


    ##### Read structure files
    NodeNumber,NodeX,NodeY,NodeZ,NumberOfNodes,NumberOfModes,Phi,wnv,xinv,BeamNumber,BeamLength,BeamNode1,BeamNode2,BeamSectionLabel = readBridgeInfoFiles(NodeInformationFile,ModalInformationFile,FrequencyInformationFile,BeamInformationFile,DOFactive)

    ########## UNITS
    kPhiWithUnits = 1.  # Some software, e.g. sofistik, give mode shapes with "units", like tonf m, in that case kPhiWithUnits = X applies a correction factor (to be check). Otherwise set = 1.

    # OUTPUT
    #Create output folder and path to results
    NodeNumberToWrite = [2] #24 for midspan at centreline
    #VehicleOrderToWrite = [0]   # Order of vehicle to get the response from
    writeOutputRate = 1
    # Animation
    animationRate = 10   #10 # Rate of frame recording for the animations, = 0 if no animation is to be recorded
    scaleFactorAnimation = 1100
    durationAnimation = 100
    maxLoad = abs(PHarmonicLoad[0])  # To normalise the load, for visualisation
    xyzLoad = [[3,0,6]] # xyzLoad is a array with 3x1 arrays that contain the xyz coord of the loads
    removeAnimationFigures = 1 # = 1 to remove all files in the folder from which the animation is created. = 0 otherwise

    print('Pre-processing: COMPLETE')

    ######################################################

    tCPU = time.time()
    kwargs = {
        'DOFactive': DOFactive,
        'modesToInclude': modesToInclude,
        'NumberOfNodes': NumberOfNodes,
        'Phi': Phi,
        'wnv': wnv,
        'xinv': xinv,
        'beta': beta,
        'gamma': gamma,
        'NodeX': NodeX,
        'NodeY': NodeY,
        'NodeZ': NodeZ,
        'NodeNumber': NodeNumber,
        #'VehicleBeams': VehicleBeams,
        'BeamNode1': BeamNode1,
        'BeamNode2': BeamNode2,
        'BeamLength': BeamLength,
        'BeamNumber': BeamNumber,
        'BeamSectionLabel': BeamSectionLabel,
        'PHarmonicLoad': PHarmonicLoad,
        'wHarmonicLoad': wHarmonicLoad,
        'NodeHarmonicLoad': NodeHarmonicLoad,
        'DofHarmonicLoad': DofHarmonicLoad,
        'caseName':caseName,
        'scaleFactorAnimation':scaleFactorAnimation,
        #'plotStructurePart':'spineOnly',
        'NodeNumberToWrite':NodeNumberToWrite,
        'writeOutputRate':writeOutputRate
    }

    # Plot structural model and vibration modes in folder "./preprocessing"
    plots = PlotModel(**kwargs)
    #plots.plotUndeformedModel()
    plots.plotMode(1)
    #plots.plotMode(2)

    t = np.arange(0.,tmax,dt)

    # Simulation
    hl = HarmonicLoads(**kwargs)
    initfilter = ModalMatrices(**kwargs)

    # Set control of written output
    writers = WriteToFile(**kwargs)
    r_Output = writers.initVectorOutput(t)
    r2dot_Output = writers.initVectorOutput(t)

    # Filter mode shape matrix
    M,C,K,PhiReduced = initfilter.filterModalMatrices()
    # Initialise vectors
    Pn0,u,udot = initfilter.initialiseModalVectors()
    filenamesAnimation = []

    pbar = tqdm(total=len(t),leave=False)

    #### DYNAMIC response
    r = np.zeros(NumberOfNodes*len(DOFactive))
    rdot = np.zeros(NumberOfNodes*len(DOFactive))
    for i in range(1,len(t)): # Main loop - time
        # Initialise Newmark solver
        u2dot,kbar,a,b = InitialiseMDOFNewmark(Pn0,u,udot,beta,gamma,t[i]-t[i-1],M,C,K)
        # Initialise Modal forcing
        P = np.zeros(NumberOfNodes*len(DOFactive))
        #PHarmonic = np.zeros(NumberOfNodes*len(DOFactive))
        LoadAmplitude = np.zeros(len(PHarmonicLoad))
        ####### Sum of all the dynamic actions
        for dl in range(len(PHarmonicLoad)): # Loop in harmonic loads
            # Obtain load
            PHarmonic = hl.harmonicFixedLoad(t[i],P,dl)[0]
            LoadAmplitude[dl] = hl.harmonicFixedLoad(t[i],P,dl)[1]# For the animation
            P += PHarmonic

        # Solution
        Pn = np.dot(np.transpose(PhiReduced),P)
        Pn.shape = u.shape
        u,udot,u2dot = MDOFNewmark(Pn0,Pn,u,udot,u2dot,beta,gamma,t[i]-t[i-1],kbar,a,b)		# v1_1 don't consider impulsive load but ramp load from start of step to end.
        r,rdot,r2dot = structuralResponse(u,udot,u2dot,PhiReduced)
        if kPhiWithUnits != 1:
            r=r*kPhiWithUnits # Sofistik correction
            rdot=rdot*kPhiWithUnits # Sofistik correction
            r2dot=r2dot*kPhiWithUnits # Sofistik correction

        Pn0 = Pn	# Update Pn0 to value in previous step

        # Write output
        output_flag = writeOutputRate > 0 and (i % writeOutputRate == 0 or i == 1 or i == len(t))
        if output_flag:
            writers.writeOutput(r_Output,r,i)
            writers.writeOutput(r2dot_Output,r2dot,i)

        # Animation
        output_flag = animationRate > 0 and (i % animationRate == 0 or i == 1 or i == len(t))
        if output_flag:
#            plots.plotDeformedStructureSection(i,t[i],r,showPlotSection,plotSectionNodes)
#            plots.plotDeformedStructureSectionLoads(i,t[i],r,LoadAmplitude,maxLoad,xyCGVehiclev,showPlotSection,plotSectionNodes)
            plots.plotDeformedStructureLoads(i,t[i],r,LoadAmplitude,maxLoad,xyzLoad,DofHarmonicLoad)

            filenamesAnimation.append('deformation_iteration_'+str(i)+'.png')

        # Progress bar
        pbar.update(1)

    elapsed = time.time() - tCPU
    print('Calculation time: ', elapsed, ' s')


    print(' Calculation: COMPLETE')

    writers.writeOutputFile(r_Output,'displacement','plotsYes') # plotsYes
    writers.writeOutputFile(r2dot_Output,'acceleration','plotsYes') # plotsYes

    print(' Results output: COMPLETE')

    # Create animation from files
    if animationRate!=0:
        plots.createAnimation(filenamesAnimation,durationAnimation,removeAnimationFigures)
