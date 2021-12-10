## Code written by: Manhyuk Han (manhyukhan@kaist.ac.kr)
#

"""
This class is a collection of functions for showing data with VMD
"""
import os
import numpy as np
import h5py
import polychrom.polymerutils
from polychrom.hdf5_format import load_URI, load_hdf5_file
import tempfile
import time

def load(name,block=0):
    if block:
        file = name+"::"+str(block)
        data = polychrom.polymerutils.load(file)
    else:
        file = name
        data = load_hdf5_file(file)['pos']
    return data

def getTmpPath(folder=None, **kwargs):
    tmpFile = tempfile.NamedTemporaryFile(dir=folder, mode='w', **kwargs)
    tmpPath = tmpFile.name
    tmpFilename = os.path.split(tmpPath)[-1]
    tmpFile.close()
    return tmpPath, tmpFilename

def loopColoring(data, time,N):
    file = load_hdf5_file(data)['positions']
    loopPositions = file[time,:,:]          # 50
    
    # color is np-array with N dimension, where N is the length of the data
    st = [i for i in loopPositions[:,0]]        # 50
    end = [i for i in loopPositions[:,1]]       # 50
    st.sort()
    end.sort()   

    numCohesin = len(st)
    color = np.zeros(N)
    
    for i in range(numCohesin):
        color[st[i]]=100
        color[end[i]]=100
        color[st[i]+1:end[i]]=-100
    
    return color


def saveTmpPdb(data,filename):
    with open(filename,'w') as towrite:
        serialColors = np.array(
            [int((j * 450.0) / (len(data))) - 225 for j in range(len(data))]
        )
    
        newData = np.zeros((len(data), 5))
        
        newData[:,:3] = data
        newData[:,3] = 1.0
        newData[:,4] = serialColors

        j = 0
        for i in newData:
            j += 1
            i[0] = round(i[0],2)
            i[1] = round(i[1],2)
            i[2] = round(i[2],2)
            towrite.write(
                f"{'ATOM'}{str(j).rjust(7)} {'H'.rjust(4)}{'POL'.rjust(4)} X{'1'.rjust(4)}    {str(i[0]).rjust(8)}{str(i[1]).rjust(8)}{str(i[2]).rjust(8)}{str(i[3]).rjust(6)}{str(int(i[4])).rjust(6)}\n"
            )

        for k in range(len(data)):
            if k+1==len(data): break
            towrite.write(
                f"CONECT{str(k+1).rjust(5)}{str(k+2).rjust(5)}\n"
            )
        towrite.write("END\n")
        towrite.flush()

def savePdb(
    xyzs,
    filename,
    types=False
):
    with open(filename, 'w') as pdb:
        if not types:
            atomflag = 'ATOM'
            atomtype = 'H'.rjust(4)
            resname = 'POL'.rjust(4)
            chainname = 'X'.rjust(2)
            resid = '1'.rjust(4)
            occupancy = '1.0'.rjust(6)
            beta = '1.0'.rjust(6)
        elif not isinstance(types, list):
            raise TypeError
        
        coordProcess = lambda x: x - np.min(x, axis=0)[None,:]
        
        ## load xyz data
        if not isinstance(xyzs,list): xyzs = [xyzs]
        
        ## make 1st frame
        xyz = xyzs.pop(0)
        xyz = polychrom.polymerutils.load(xyz) if isinstance(xyz,str) else xyz
        #xyz -= np.min(xyz, axis=0)[None,:]
        xyz = coordProcess(xyz)
        
        try:
            if len(types) != len(xyz): raise ValueError
        except TypeError:
            pass
        
        ## write pdb file/ line by line
        for i in range(len(xyz)):
            if types:
                atomflag = types[i]['atomflag']
                atomtype = types[i]['atomtype'].rjust(4)
                resname =  types[i]['resname'].rjust(4)
                resid = types[i]['resid'].rjust(4)
                chainname = types[i]['chainname'].rjust(2)
                occupancy = types[i]['occupancy'].rjust(6)

            serial = str(i+1).rjust(11 - len(atomflag))
            beta = str(int((i * 450.0) / (len(xyz))) - 225).rjust(6)
            
            x = str(round(xyz[i,0],2)).rjust(8)
            y = str(round(xyz[i,1],2)).rjust(8)
            z = str(round(xyz[i,2],2)).rjust(8)
            
            pdb.write(atomflag
                    +serial
                    +' '
                    +atomtype
                    +resname
                    +chainname
                    +resid
                    +'    '
                    +x
                    +y
                    +z
                    +occupancy
                    +beta
                    +'\n')
            
        ## write CONECT code
        for j in range(len(xyz)-1):
            pdb.write(
                f"CONECT{str(j+1).rjust(5)}{str(j+2).rjust(5)}\n"
            )
        pdb.write('END\n')
    if len(xyz)==0: return
    if os.path.isfile(filename):
        from vmd import Molecule, atomsel
        molecule = Molecule.Molecule()
        molecule.load(filename)
        assert molecule.numFrames()==1
        assert molecule.numAtoms()==len(xyz)
        
        for xyz in xyzs:
            xyz = polychrom.polymerutils.load(xyz) if isinstance(xyz,str) else xyz
            #xyz -= np.min(xyz, axis=0)[None,:]
            xyz = coordProcess(xyz)
            
            assert molecule.dupFrame()==molecule
            assert molecule.numAtoms()==len(xyz)
            curframe = molecule.curFrame()
        
            beta = []
            
            sel = atomsel('all', frame=curframe)
            
            for i in range(molecule.numAtoms()):
                beta.append(int((i * 450.0) / (len(xyz))) - 225)
            
            sel.x = xyz[:,0]
            sel.y = xyz[:,1]
            sel.z = xyz[:,2]
            sel.beta = beta
            
            print(f'coordinate of frame {curframe} is updated')
        
        # save dcd file for 2nd to last frame
        assert molecule.numFrames()==1+len(xyzs)
        molecule.delFrame(first=0,last=0)
        dcdFileName = ''.join(filename.split('.pdb')[0])+'.dcd'
        molecule.save(dcdFileName,filetype='dcd')
        time.sleep(0.2)
        

def show(
    data, showChain='worm', color=False
):
    """Show a single rainbow-colored chain using VMD.
    
    Arguements:
    showChain - 'worm' or 'spheres'
    color : loopColoring(data,time), default is zero
    
    Keywords arguements
    chain_radius : the radius """
    
    ## single frame ##
    if isinstance(data, str):
        data = load_hdf5_file(data)['pos']
    if isinstance(data, np.ndarray):
        data -= np.min(data, axis=0)[None, :]
        print(data.min())
        
        tmpPdbPath, _ = getTmpPath(suffix='.pdb')
        saveTmpPdb(data,tmpPdbPath)
        
    ## multi frame (data type list of URIs...) ##
    if isinstance(data,list):
        if color and (not isinstance(color,list)):
            raise TypeError("color must be assigned for all frames")
        
        filenames = []
        for i in range(len(data)):
            frame = polychrom.polymerutils.load(data[i])
            frame -= np.min(frame, axis=0)[None, :]
            tmpPdbPath, _ = getTmpPath(suffix='.pdb')
            filenames.append(tmpPdbPath)
            saveTmpPdb(frame,tmpPdbPath)
    
    tmpScript = tempfile.NamedTemporaryFile(mode='w')
    tmpScript.write('display depthcue off\n')
    tmpScript.write('display rendermode GLSL\n')
    tmpScript.write('color Display Background black\n')
    
    tmpScript.write('mol modselect 0 0 resname POL\n')
    if showChain == 'worm':
        tmpScript.write('mol modstyle 0 0 Licorice 0.1\n')
    elif showChain == 'spheres':
        tmpScript.write('mol modstype 0 0 Points 1.0\n')
    else: raise ValueError('chainType must be worm or sphere')    
    
    # Spectrum
    tmpScript.write('mol modmaterial 0 0 BrushedMetal\n')
    tmpScript.write('mol modcolor 0 0 Beta\n')
    tmpScript.write('mol scaleminmax 0 0 -225 -225\n')
        
    tmpScript.write('display resetview')
    
    # Multiframes
    try:
        for file in filenames[1:]:
            tmpScript.write(f"mol addfile {file} type pdb waitfor all\n")
        tmpScript.flush()
        
        if os.name == "posix":
            os.system(f"vmd -m {filenames[0]} -e {tmpScript.name}")
        
        tmpScript.close()
        
        for file in filenames:
            os.remove(file)
    except NameError:    
        tmpScript.flush()
        
        if os.name == "posix":  # if linux
            os.system("vmd -m %s -e %s" % (tmpPdbPath, tmpScript.name))
        
        tmpScript.close()
        try:
            os.remove(tmpPdbPath)
        except FileNotFoundError:
            time.sleep(0.2)
    