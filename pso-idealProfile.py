# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from os import listdir # To get list of previous results
import pickle   # Save data

import imp
lumapi = imp.load_source("lumapi", "C:\\Program Files\\Lumerical\\2020a\\api\\python\\lumapi.py") # Lumerical API, Windows path
ctk = imp.load_source("colorToolkit", ".\\Color toolkit\\colorToolkit.py") # Personal set of functions, hopefully not too wonky




#==============================================================================
# Helping functions
#==============================================================================

#------------------------------------------------------------------------------
# Latin Square Hypercube to initialize the particles

def lhs(swarmSize, parameterSpace):

    numParameter =  parameterSpace.shape[0] # Number of parameter

    x = np.zeros((swarmSize, numParameter))
 
    # Make row and column

    cellWidth = 1 / swarmSize
    possibleCoordinates = 0.5 * cellWidth + np.array([i * cellWidth for i in range(swarmSize)])

    # Random permutation to get a Latin Square, and scaling to the parameter space
    for i in range(numParameter):
        x[:,i] = np.random.permutation(possibleCoordinates) * (parameterSpace[i,1] - parameterSpace[i,0]) + parameterSpace[i,0]

    return x


#------------------------------------------------------------------------------
# Load previous data

def initializeBest(previousFileFolder, numParameter, colorTarget, balanceFOM, sourceName):

    idealLambda, _ = ctk.determineIdealLambda(colorTarget)

    print('\nLoading previous data:\n')

    previousFileList = listdir(previousFileFolder)

    gFitness = 100 # starts at a value that will never be reached
    gBest = np.zeros(numParameter)      # Will get updated to proper value when we find the first result
    
    if previousFileList:
        for i in range(len(previousFileList)):
            print(previousFileList[i])
    
            with open(previousFileFolder + '\\' + previousFileList[i], 'rb') as handle:
                previousData = pickle.load(handle)
    
            previousStructure = previousData["matrixStructure"]
            previousT = previousData["matrixT"]
            previousLambda = previousData["Lambda"]
    
            FOM = np.zeros(previousStructure.shape[0:2])
    
            [sourceValues, xCIE1931, yCIE1931, zCIE1931] = ctk.loadCIEData(previousLambda, sourceName)
    
            for j in range(previousStructure.shape[0]):
                    for k in range(previousStructure.shape[1]):
                            FOM[j,k] = fctFOM(colorTarget, balanceFOM, idealLambda, previousLambda, previousT[j,k,:], 
                                              sourceValues, xCIE1931, yCIE1931, zCIE1931)
    
            gFitnessTemp = - np.nanmax(FOM)
    
            if gFitnessTemp < gFitness:
                gFitness = gFitnessTemp
                gBest = previousStructure[np.unravel_index(np.nanargmax(FOM, axis=None), FOM.shape)]
    else:
        print('No previous data.')

    print('\ngFitness: ', gFitness)

    return gFitness, gBest


#------------------------------------------------------------------------------
# FOM Function

def fctFOM(colorTarget, balanceFOM, idealLambda, Lambda, T, sourceValues, xCIE1931, yCIE1931, zCIE1931):

    whitePoint = np.array([0.31271, 0.32902])

    # (1) Chromaticity

    xyz = ctk.determineColor(Lambda, T, sourceValues, xCIE1931, yCIE1931, zCIE1931)[0] # Determine color

    vect = np.array([xyz[0] - whitePoint[0], xyz[1] - whitePoint[1]])
    vect_u = vect / np.linalg.norm(vect) # get a unit vector

    vectTarget = np.array([colorTarget[0] - whitePoint[0], colorTarget[1] - whitePoint[1]])
    vectTarget_u = vectTarget / np.linalg.norm(vectTarget)

    theta = np.arccos(np.clip(np.dot(vect_u, vectTarget_u), -1.0, 1.0)) # Get the angle between the two

    fomChromaticity = np.linalg.norm(vect) / np.linalg.norm(vectTarget) * np.exp(- np.power(theta,2)) * (1 - np.sin(theta))

    # (2) Intensity

    TTemp = T[(Lambda > idealLambda - 50) & (Lambda < idealLambda + 50)] # +/- 50 nm around the maximum
    maxValue = 1
    steepness = 10
    midpoint = 0.5

    fomIntensity = maxValue / (1 + np.exp(- steepness * (TTemp.max() - midpoint)))

    fom = np.power(fomChromaticity, balanceFOM[0])  \
        * np.power(fomIntensity, balanceFOM[1]) 

    return fom










#==============================================================================
# PSO function
#==============================================================================

def run(generationNumber, swarmSize, fileName, parameterName, parameterSpace, balanceFOM, colorTarget, sourceName):

    fdtd = lumapi.FDTD()
    fdtd.load(fileName)
    fdtd.switchtolayout()

    numParameter = len(parameterName)

    # Prepare structures to save data

    matrixStructure = np.zeros((generationNumber, swarmSize, numParameter))
    matrixFOM = np.zeros((generationNumber,swarmSize))
    matrixT = np.zeros((generationNumber, swarmSize, 400)) # Depend on optimization, not fully automated

    # Velocity / behaviors parameters

    c1 = 1.49   # Balance between personal best (c1), default: 1.49
    c2 = 1.49   # and global best (c2) when adjusting the velocity, default: 1.49
    w = 0.9     # Overall velocity multiplicator, will diminish over time (w - w_difference), default: 0.9

    wDifference = 0.5 / generationNumber # drives how fast w will decrease over the generations, default: 0.5
    wallType = 1 # 1, reflection; 2, absorbtion; drives behavior of particles around the boundaries, default: 1

    # Load previous data

    previousFileFolder = '.\\results\\ideal'


    #--------------------------------------------------------------------------
    # Initialization
    #--------------------------------------------------------------------------

    # Parameter initialization

    x = lhs(swarmSize, parameterSpace)  # Latin Hypercube Sampling
    velocity = np.zeros((swarmSize, numParameter))  # Velocity of each particle
    pBest = np.zeros_like(velocity)     # Personal best for the particles


    for i in range(swarmSize):
        for j in range(numParameter):
            pBest[i,j] = np.random.rand() * (parameterSpace[j,1] - parameterSpace[j,0]) + parameterSpace[j,0]   # The first personal best is random

    pFitness = np.ones(swarmSize) * 100 # Starts at a greater value and get corrected with first particles

    # Load previous results (or not) to determine the global best

    gFitness, gBest = initializeBest(previousFileFolder, numParameter, colorTarget, balanceFOM, sourceName) # Determined from previous data
    
    # Run one simulation to get Lambda and load CIE Data

    fdtd.select("::model")
    for k in range(len(parameterName)):
        fdtd.set(parameterName[k] , x[0,k].item()) # Convert to native Python type (necessary to change in FDTD)
    
    fdtd.run()

    test = fdtd.getresult('R','T')
    Lambda = test["lambda"] * 1e9 # Get the wavelength in nm (for scripts)
    Lambda = Lambda.ravel() # Shape from (400,1) to (400,) to math T_TM shape (weird FDTD stuff)

    [sourceValues, xCIE1931, yCIE1931, zCIE1931] = ctk.loadCIEData(Lambda, sourceName) # Load CIE data for our working range

    # Determine idealLambda to check the transmission around it

    idealLambda, _ = ctk.determineIdealLambda(colorTarget)

  
    #--------------------------------------------------------------------------
    # Prepare plot
    #--------------------------------------------------------------------------


    plt.ion()
    fig, axs = plt.subplots(1,3, figsize=(15,5))

    # Evolution of FOM

    evolMax,    = axs[0].plot(np.nanmax(matrixFOM, axis=1), '.') # Returns a tuple of line objects, thus the comma
    evolMean,   = axs[0].plot(np.nanmean(matrixFOM, axis=1), '.')

    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('FOM')
    axs[0].legend(['Maximum', 'Mean'])

    axs[0].set_xlim([-1, generationNumber])
    axs[0].set_ylim([0, 1.1])
    
    # Position of swarm
    
    axs[1].plot(x[:,0], x[:,1], '.', label='Swarm')
    axs[1].plot(gBest[0], gBest[1], '.', label='Global best')
    axs[1].legend()
    # Map arrow betwen x and pBest
    axs[1].set_xlabel(parameterName[0])
    axs[1].set_xlim([parameterSpace[0,0], parameterSpace[0,1]])
    axs[1].set_ylabel(parameterName[1])
    axs[1].set_ylim([parameterSpace[1,0], parameterSpace[1,1]])

    # Evolution of transmission

    bestIndex = np.unravel_index(np.nanargmax(matrixFOM, axis=None), matrixFOM.shape)

    # ax2 = fig.add_subplot(122)
    evolTM,     = axs[2].plot(Lambda, matrixT[bestIndex[0], bestIndex[1], :] * sourceValues)

    axs[2].set_xlabel('lambda (nm)')
    axs[2].set_ylabel('Reflected intensity')

    axs[2].set_xlim([Lambda.min(), Lambda.max()])
    axs[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.pause(5)     

    fig.canvas.draw()
    fig.canvas.flush_events()
        

    ##-------------------------------------------------------------------------
    ## Iteration
    ##-------------------------------------------------------------------------

    for i in range(generationNumber):
        for j in range(swarmSize):
            
            print("Generation", i,"Particle", j)

            #------------------------------------------------------------------
            # Update location and speed of the swarm

            if wallType == 1:
                for k in range(numParameter):
                    if x[j,k] < parameterSpace[k,0]:
                        x[j,k] = parameterSpace[k,0] + (parameterSpace[k,0] - x[j,k])
                        velocity[j,k] = - velocity[j,k]
                    elif x[j,k] > parameterSpace[k,1]:
                        x[j,k] = parameterSpace[k,1] - (x[j,k] - parameterSpace[k,1])
                        velocity[j,k] = - velocity[j,k]
            else:
                for k in range(numParameter):
                    if x[j,k] < parameterSpace[k,0]:
                        x[j,k] = parameterSpace[k,0]
                    elif x[j,k] > parameterSpace[k,1]:
                        x[j,k] = parameterSpace[k,1]


            #------------------------------------------------------------------
            # FDTD
                        
            fdtd.switchtolayout()
            
            # Adjust the structure            
            fdtd.select("::model")
            for k in range(len(parameterName)):
                fdtd.set(parameterName[k] , x[j,k].item()) # Convert to native Python type

            # Run the simulation - Intensity, chromaticity
            fdtd.run()
      
            # Get the result
            test = fdtd.getresult('R','T')
            T_TM0 = test["T"]   # test is a dictionary, can also access lambda, f, etc.


            #------------------------------------------------------------------
            # FOM

            fom = fctFOM(colorTarget, balanceFOM, idealLambda, Lambda, T_TM0, sourceValues, xCIE1931, yCIE1931, zCIE1931)
            xFitness = - fom

            print('FOM', fom, '\n')


            #------------------------------------------------------------------
            # Save relevant data for this particle

            matrixStructure[i,j,:] = x[j,:]
            matrixFOM[i,j] = fom

            matrixT[i,j,:] = T_TM0

            #------------------------------------------------------------------
            #  Update global/personal best
            
            if xFitness < gFitness:
                gFitness = xFitness
                for k in range(numParameter):
                    gBest[k] = x[j,k]

            if xFitness < pFitness[j]:
                pFitness[j] = xFitness
                for k in range(numParameter):
                    pBest[j,k] = x[j,k]


            #------------------------------------------------------------------
            # Update velocity

            for k in range(numParameter):
                velocity[j,k] = w * velocity[j,k] + c1 * np.random.rand() * (pBest[j,k] - x[j,k]) + c2 * np.random.rand() * (gBest[k] - x[j,k])
                x[j,k] = x[j,k] + velocity[j,k]
          
            
        #----------------------------------------------------------------------
        # At the end of a generation we reduce w

        w = w - wDifference


        #----------------------------------------------------------------------
        # Update graph of FOM evolution and best filter

        evolMax.set_ydata(np.nanmax(matrixFOM, axis=1))
        evolMean.set_ydata(np.nanmean(matrixFOM, axis=1))
        
        axs[1].clear()
        axs[1].plot(x[:,0], x[:,1], '.', label='Swarm')
        axs[1].plot(gBest[0], gBest[1], '.', label='Global best')
        axs[1].legend()
        axs[1].set_title('Generation: %i' % i)
        axs[1].set_xlabel(parameterName[0])
        axs[1].set_xlim([parameterSpace[0,0], parameterSpace[0,1]])
        axs[1].set_ylabel(parameterName[1])
        axs[1].set_ylim([parameterSpace[1,0], parameterSpace[1,1]])

        bestIndex = np.unravel_index(np.nanargmax(matrixFOM, axis=None), matrixFOM.shape)
        
        evolTM.set_ydata(matrixT[bestIndex[0], bestIndex[1], :] * sourceValues)
        
        axs[2].set_title('Generation: %i' % i)

        fig.canvas.draw()
        plt.pause(5) 

        fig.canvas.flush_events()

    fdtd.close() # At the end of process we close FDTD window

    return matrixStructure, matrixT, matrixFOM, Lambda










# =============================================================================
# Parameters for the user
# =============================================================================

# PSO Size

generationNumber = 2 #60
swarmSize = 2 #30

# Solution space

fileName = 'ideal_2d_rwg.fsp' # File to load

parameterName = ['gratingWidth', 'gratingThickness']   # Parameters to change in the FDTD model, precise name

parameterSpace =    \
    np.array([  [50 * 1e-9, 325 * 1e-9],   \
                [10 * 1e-9, 200 * 1e-9]])          # Min and max of each parameter

# FOM
        
# Balance: Chromaticity, Intensity, Polarisation, Angular tolerance
balanceFOM = [2, 1, 0, 0]

# Target chromaticity (CIE 1931 xyY coordinates):
#           sRGB            DCIP3           BT2020
# Red       [0.64, 0.33]    [0.68, 0.32]    [0.708, 0.292]
# Green     [0.3, 0.6]      [0.265, 0.690]  [0.17, 0.797]
# Blue      [0.15, 0.06]    [0.15, 0.06]    [0.131, 0.046]

# sourceList: 'D65', 'OLED_White', 'OLED_Blue', 'OLED_Green', 'OLED_Red'

colorTarget = np.array([0.265, 0.690]) # Send directly xy coordinates
sourceName = 'OLED_White'

# =============================================================================
# Run PSO
# =============================================================================

matrixStructure, matrixT, matrixFOM, Lambda = run(generationNumber, swarmSize, fileName, parameterName, parameterSpace, balanceFOM, colorTarget, sourceName)

# ============================================================================
# Save data
# =============================================================================

fullData = {
    "matrixStructure" : matrixStructure,
    "matrixT" : matrixT,
    "matrixFOM" : matrixFOM,
    "Lambda" : Lambda,
    "psoParameters" : {
        "filename" : fileName,
        "parameterName" : parameterName,
        "parameterSpace" : parameterSpace,
        "colorTarget" : colorTarget,
        "balanceFOM" : balanceFOM,
        "generationNumber" : generationNumber,
        "swarmSize" : swarmSize    
    }
}

with open('psoSaved2.pickle', 'wb') as handle:
    pickle.dump(fullData, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Optimization finished and results saved.')













































 
