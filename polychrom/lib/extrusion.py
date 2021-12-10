"""
A set of utilities for performing loop extrusion simulations. 

"""
import numpy as np

class leg(object):
    def __init__(self, pos, attrs={"stalled":False, "CTCF":False}):
        """A leg has two important attributes: pos (positions) and attrs (a custom list of attributes)

        Args:
            pos ([type]): position of the leg
            attrs (dict): [description]. Defaults to {"stalled":False, "CTCF":False}.
        """
        self.pos = pos
        self.attrs = dict(attrs)
                
class cohesin(object):
    """
    A cohesin class provides fast access to attributes and positions
    
    cohesin.left is a left leg of cohesin, cohesin.right is a right leg
    cohesin[-1] is also a left leg and cohesin[1] is a right leg
    
    Also, cohesin.any("myattr") is True if myattr==True in at least one leg
    cohesin.all("myattr") is True iff myattr==True in both legs
    """
    def __init__(self, leg1, leg2):
        self.left = leg1
        self.right = leg2
        
    def any(self, attr):
        return self.left.attrs[attr] or self.right.attrs[attr]
         
    def all(self, attr):
        return self.left.attrs[attr] and self.right.attrs[attr]
        
    def __getitem__(self,item):
        if item == -1:
            return self.left
        if item == 1:
            return self.right
        else:
            raise ValueError()

def unloadProb(cohesin, args):
    """
    Defines unload probability based on a state of cohesin
    """
    if cohesin.any("stalled"):
        # if one side is stalled, we have different unloading probability
        # Note that here we define stalled cohesins as those stalled not at CTCFs
        return 1 / args["LIFETIME_STALLED"]
    # otherwise we are just simply unloading
    return 1 / args["LIFETIME"]
        
def loadOne(cohesins, occupied, args):
    """
    A function to load one cohesin (random place)
    """
    while True:
        a = np.random.randint(args["N"])
        if (occupied[a]==0) and (occupied[a+1] == 0):
            occupied[a] = 1
            occupied[a+1] = 1
            cohesins.append(cohesin(leg(a),leg(a+1)))
            break
        
def loadOneSpecific(cohesins, occupied, args):
    """
    A function to load one cohesin at specific position.
    """
    sites = args["loadSites"]
    prob = args['loadProb']
    if not prob:
        prob = list(np.ones(len(sites)))
    assert isinstance(sites,list)
    assert isinstance(sites[0],int) or isinstance(sites[0],np.int64)
    sumProb = sum(prob)
    culProb = [sum(prob[:i+1])/sumProb for i in range(len(prob))]
    assert culProb[-1] == 1
    currentCohesins = len(cohesins)
    
    while len(cohesins) == currentCohesins:
        trigger = np.random.random()
        for i in range(len(culProb)):
            if trigger < culProb[i]:
                if (occupied[sites[i]]==0) and (occupied[sites[i]+1]==0):
                    occupied[sites[i]] == 1
                    occupied[sites[i]+1] == 1
                    cohesins.append(cohesin(leg(sites[i]),leg(sites[i]+1)))
                    break

def capture(cohesin, args):
    """
    Descirption of CTCF capture
    
    Note the for-loop over left/right sites below, and using cohesin[side] to get left/right leg
    
    Also note how I made ctcfCapture, a dict with -1 coding for left side, and 1 for the right side
    and ctcfCapture are dicts as well: keys are locations, and values are probabilities of capture
    """
    for side in [1,-1]:
        # get probability of capture or otherwise it is 0
        if np.random.random() < args["ctcfCapture"][side].get(cohesin[side].pos,0):     # dict.get(key, ValueOtherThan)
            cohesin[side].attrs["CTCF"] = True
    return cohesin

def release(cohesin,args):
    """
    An opposite to capture - releasing cohesins from CTCF
    """
    
    if not cohesin.any("CTCF"):
        return cohesin # no CTCF: no release
    
    # attempting to release either side
    for side in [-1,1]:
        if (np.random.random() < args["ctcfRelease"][side].get(cohesin[side].pos,0)) and (cohesin[side].attrs["CTCF"]):
            cohesin[side].attrs["CTCF"] = False
    return cohesin

def translocate(cohesins, occupied, args):
    """
    This function describes everything that happened with cohesins; loading/unloading them and stalling against each others
    
    It relies on the functions cohesins and free the matching occupied sites
    """
    probSpecific = args['loadProb']
    # first, we try to unload cohesins and free the matching occupied sites
    for i in range(len(cohesins)):
        prob = unloadProb(cohesins[i],args)
        if np.random.random() < prob:
            occupied[cohesins[i].left.pos] = 0
            occupied[cohesins[i].right.pos] = 0
            del cohesins[i]
            if not probSpecific: loadOne(cohesins,occupied,args)
            else: loadOneSpecific(cohesins,occupied,args)
            
    # then we try to capture and release them by CTCF sites
    for i in range(len(cohesins)):
        cohesins[i] = capture(cohesins[i],args)
        cohesins[i] = release(cohesins[i],args)
        
    # finally we translocate, and mark stalled cohesins because the unloadProb needs this
    for i in range(len(cohesins)):
        cohesin = cohesins[i]
        for dir in [-1,1]:
            if not cohesin[dir].attrs["CTCF"]:
                # cohesins that are not at CTCFs but cannot move: labeled as stalled
                if occupied[cohesin[dir].pos + dir] != 0:       # already occupied position
                    cohesin[dir].attrs["stalled"] = True
                else:
                    cohesin[dir].attrs["stalled"] = False
                    occupied[cohesin[dir].pos] = 0
                    occupied[cohesin[dir].pos + dir] = 1
                    cohesin[dir].pos += dir
        cohesins[i] = cohesin

def color(cohesins,args):
    """A helper function that converts a list of cohesins to an array colored by cohesin state"""
    def state(attrs):
        if attrs["stalled"]:
            return 2
        if attrs["CTCF"]:
            return 3
        return 1
    ar = np.zeros(args["N"])
    for i in cohesins:
        ar[i.left.pos] = state(i.left.attrs)
        ar[i.right.pos] = state(i.right.attrs)
    return ar

class bondUpdater(object):

    def __init__(self, LEFpositions):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks=100, smcStepsPerBlock=1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """

        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        #precalculating all bonds
        allBonds = []
        
        loaded_positions  = self.LEFpositions[self.curtime : self.curtime+blocks]
        allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                        for j in range(loaded_positions.shape[1])] for i in range(blocks)]

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))         # remove redundancies. all possible bonds remmain [(left,right),(...),...]

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)     # start with first block, one element from the allBonds deleted

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks 
        
        return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict        # update parameter set
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        
        return self.curBonds, pastBonds
