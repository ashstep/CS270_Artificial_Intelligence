'''
About: Working with example constraint satisfaction problems (CSPs) and the basics of probabilistic inference
@author Ashka  
Created on Mar 1, 2017
'''


from collections import defaultdict

#########################  Example CSP: Street Puzzle and n-Queens with backtracking algorithm #####################

class Constraint:
    """A constraint of a CSP.  Members include
     - name: a string for debugging
     - domain, a list of variables on which the constraint acts
     - predicate, a boolean function with the same arity as the domain.
     """
    def __init__(self,name,domain,pred):
        self.name = name
        self.domain = domain
        self.predicate = pred
        
    def isSatisfied(self,vars):
        """Given a dictionary of variables, evaluates the predicate.
        If a variable in the domain isn't present, raises a KeyError."""
        args = [vars[v] for v in self.domain]
        return self.predicate(*args)

class CSP:
    """Defines a constraint satisfaction problem.  Contains 4 members:
    - variables: a list of variables
    - domains: a dictionary mapping variables to domains
    - constraints: a list of Constraints.
    - incidentConstraints: a dict mapping each variable to a list of
      constraints acting on it.
    """
    
    def __init__(self,variables=[],domains=[]):
        """Input: a list of variables and a list of domains.

        Note: The variable names must be unique, otherwise undefined behavior
        will result.
        """
        self.variables = variables[:]
        self.domains = dict(zip(variables,domains))
        self.constraints = []
        self.incidentConstraints = dict((v,[]) for v in variables)
        
    def addVariable(self,var,domain):
        """Adds a new variable with a given domain.  var must not already
        be present in the CSP."""
        if var in self.domains:
            raise ValueError("Variable with name "+val+" already exists in CSP")
        self.variables.append(var)
        self.domains[var] = domain
        self.incidentConstraints[var] = []

    def addConstraint(self,varlist,pred,name=None):
        """Adds a constraint with the domain varlist, the predicate pred,
        and optionally a name for printing."""
        if name==None:
            name = "c("+",".join(str(v) for v in varlist)+")"
        self.constraints.append(Constraint(name,varlist,pred))
        for v in varlist:
            self.incidentConstraints[v].append(self.constraints[-1])

    def addUnaryConstraint(self,var,pred,name=None):
        """Adds a unary constraint with the argument var, the predicate pred,
        and optionally a name for printing."""
        self.addConstraint((var,),pred,name)

    def addBinaryConstraint(self,var1,var2,pred,name=None):
        """Adds a unary constraint with the arguments (var1,var2), the
        predicate pred, and optionally a name for printing."""
        self.addConstraint((var1,var2),pred,name)

    def fixValue(self,var,value,name=None):
        """Adds a constraint that states var = value."""
        if name==None:
            name = str(var)+'='+str(value)
        self.addUnaryConstraint(var,lambda x:x==value,name)

    def nAryConstraints(self,n,var=None):
        """Returns a list of all n-ary constraints in the CSP if var==None,
        or if var is given, returns a list of all n-ary constraints involving
        var."""
        if var==None:
            return [c for c in self.constraints if len(c.domain)==n]
        else:
            return [c for c in self.incidentConstraints[var] if len(c.domain)==n]

    def incident(self,*vars):
        """incident(var1,...,varn) will return a list of constraints
        that involve all of var1 to varn."""
        if len(vars)==0: return self.constraints
        res = set(self.incidentConstraints[vars[0]])
        for v in vars[1:]:
            res &= set(self.incidentConstraints[v])
        return [c for c in res]

    def isConstraintSatisfied(self,c,partialAssignment):
        """Checks if the partial assignment satisfies the constraint c.
        If the partial assignment doesn't cover the domain, this returns
        None. """
        try:
            res = c.isSatisfied(partialAssignment)
            return res
        except KeyError:
            return None

    def isValid(self,partialAssignment,*vars):
        for c in self.incident(*vars):
            #all entries in partialAssignment must be in the domain of c
            #for this to be checke
            if self.isConstraintSatisfied(c,partialAssignment)==False:
            	#print
            	return False
        return True
        
        """Checks if the assigned variables in a partial assignment
        are mutually compatible.  Only checks those constraints
        involving assigned variables, and ignores any constraints involving
        unassigned ones.

        If no extra arguments are given, checks all constraints relating
        assigned variables.
        
        If extra arguments var1,...,vark are given, this only checks
        constraints that are incident to those given variables."""

def streetCSP():
    """Returns a CSP corresponding to the street puzzle covered in class."""
    #these are variables to hold the info -> boxes to fill in with correct info
    nationalityVars = ['N1','N2','N3','N4','N5']
    colorVars = ['C1','C2','C3','C4','C5']
    drinkVars = ['D1','D2','D3','D4','D5']
    jobVars = ['J1','J2','J3','J4','J5']
    animalVars = ['A1','A2','A3','A4','A5']
    #these are the values we tryin to map
    nationalities = ['E','S','J','I','N']
    colors = ['R','G','W','Y','B']
    drinks = ['T','C','M','F','W']
    jobs = ['P','S','Di','V','Do']
    animals = ['D','S','F','H','Z']
               
    csp = CSP(nationalityVars+colorVars+drinkVars+jobVars+animalVars,
              [nationalities]*5+[colors]*5+[drinks]*5+[jobs]*5+[animals]*5)
    #TODO: fill me in.  Slide 18 is filled in for you.  Don't forget to enforce
    #that all nationalities, colors, drinks, jobs, and animals are distinct!

    #all different:
    csp.addConstraint(('N1','N2','N3','N4','N5'),lambda a,b,c,d,e:(a)!=(b)!=(c)!=(d)!=(e), 'AllDiffN')
    csp.addConstraint(('C1','C2','C3','C4','C5'),lambda a,b,c,d,e:(a)!=(b)!=(c)!=(d)!=(e), 'AllDiffC')
    csp.addConstraint(('D1','D2','D3','D4','D5'),lambda a,b,c,d,e:(a)!=(b)!=(c)!=(d)!=(e), 'AllDiffD')
    csp.addConstraint(('J1','J2','J3','J4','J5'),lambda a,b,c,d,e:(a)!=(b)!=(c)!=(d)!=(e), 'AllDiffJ')
    csp.addConstraint(('A1','A2','A3','A4','A5'),lambda a,b,c,d,e:(a)!=(b)!=(c)!=(d)!=(e), 'AllDiffA')

    #Englishman lives in the red house
    for Ni,Ci in zip(nationalityVars,colorVars):
        csp.addBinaryConstraint(Ni,Ci,lambda x,y:(x=='E')==(y=='R'),'Englishman lives in the red house')
    #ADDED : The Spaniard has a Dog
    for Ni,Ai in zip(nationalityVars,animalVars):
        csp.addBinaryConstraint(Ni,Ai,lambda x,y:(x=='S')==(y=='D'),'Spaniard has a Dog')
    #Japanese is a painter
    for Ni,Ji in zip(nationalityVars,jobVars):
        csp.addBinaryConstraint(Ni,Ji,lambda x,y:(x=='J')==(y=='P'),'Japanese is a painter')
    #ADDED : The Italian drinks Tea
    for Ni,Di in zip(nationalityVars,drinkVars):
        csp.addBinaryConstraint(Ni,Di,lambda x,y:(x=='I')==(y=='T'),'Italian drinks Tea')
    #Norwegian lives in first house on the left
    csp.fixValue('N1','N','Norwegian lives in the first house')
    #The owner of the Green house drinks Coffee
    for Ci,Di in zip(colorVars,drinkVars):
        csp.addBinaryConstraint(Ci,Di,lambda x,y:(x=='G')==(y=='C'),'owner of the Green house drinks Coffee')
    #green house is to the right of the white house
    for Ci,Cn in zip(colorVars[:-1],colorVars[1:]):
    	csp.addBinaryConstraint(Ci,Cn,lambda x,y:(x=='W')==(y=='G'),'Green house is to the right of the white house')  #i is white and i+1 is Green
    csp.addUnaryConstraint('C5',lambda x:x!='W','Green house is to the right of the white house')                  #fifth house is NOT W
    csp.addUnaryConstraint('C1',lambda x:x!='G','Green house is to the right of the white house')                  #first house is NOT G
    #ADDED: The Sculptor breeds Snails
    for Ji,Ai in zip(jobVars,animalVars):
    	csp.addBinaryConstraint(Ji,Ai,lambda x,y:(x=='S')==(y=='S'),'Sculptor breeds Snails')
   	#ADDED: The Diplomat lives in the Yellow house
    for Ji,Ci in zip(jobVars,colorVars):
    	csp.addBinaryConstraint(Ji,Ci,lambda x,y:(x=='Di')==(y=='Y'),'Diplomat lives in the Yellow house')
   	#ADDED: The owner of the middle house drinks Milk
    csp.fixValue('D3','M','owner of the middle house drinks Milk')
    #ADDED: The Violinist drinks Fruit juice
    for Ji,Di in zip(jobVars,drinkVars):
    	csp.addBinaryConstraint(Ji,Di,lambda x,y:(x=='V')==(y=='F'),'Violinist drinks Fruit juice')
    #ADDED: Norwegian lives next door to the Blue house
    csp.fixValue('C2','B','Norwegian lives next door to the Blue house')
    #ADDED: Fox is in the house NEXT DOOR TO the Doctors
    csp.addBinaryConstraint('A1', 'J1', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A1', 'J3', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A1', 'J4', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A1', 'J5', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A2', 'J2', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A2', 'J5', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A2', 'J4', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A3', 'J5', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A3', 'J1', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A3', 'J3', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A4', 'J4', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A4', 'J1', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A4', 'J2', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A5', 'J1', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A5', 'J5', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A5', 'J2', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    csp.addBinaryConstraint('A5', 'J3', lambda x,y:(y !='Do') or not (x == 'F') , 'Fox is in the house next to the Doctors')
    #ADDED: The Horse is NEXT DOOR TO the Diplomats
    csp.addBinaryConstraint('A1', 'J1', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A1', 'J3', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A1', 'J4', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A1', 'J5', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A2', 'J2', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A2', 'J5', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A2', 'J4', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A3', 'J1', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A3', 'J3', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A3', 'J5', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A4', 'J1', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A4', 'J2', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A4', 'J4', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A5', 'J1', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A5', 'J3', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A5', 'J2', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    csp.addBinaryConstraint('A5', 'J5', lambda x,y:(y !='Di') or not (x == 'H'), 'Horse is next to the Diplomats')
    print "CSP has",len(csp.constraints),"constraints"
    #TODO:
    return csp

def p1():
    csp = streetCSP()
    solution = dict([('A1', 'F'), ('A2', 'H'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'W'), ('D2', 'T'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])

    invalid1 = dict([('A1', 'F'), ('A2', 'H'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'T'), ('D2', 'W'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])

    invalid2 = dict([('A1', 'F'), ('A2', 'F'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'W'), ('D2', 'T'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])



    print "Valid assignment valid?",csp.isValid(solution)
    print "Invalid assignment valid?",csp.isValid(invalid1)
    print "Invalid assignment valid?",csp.isValid(invalid2)
    
    #you may wish to check the solver once you've solved problem 2
    #solver = CSPBacktrackingSolver(csp)
    #res = solver.solve()
    #print "Result:",sorted(res.items())

############################  N-Queens and Backtracking #######################

class CSPBacktrackingSolver:
    """ A CSP solver that uses backtracking.
    A state is a partial assignment dictionary {var1:value1,...,vark:valuek}.
    Also contains a member oneRings that is a dict mapping each variable to
    all variables that share a constraint.
    """
    def __init__(self,csp,doForwardChecking=True,doConstraintPropagation=False):
        self.csp = csp
        self.doForwardChecking = doForwardChecking
        self.doConstraintPropagation = doConstraintPropagation
        #compute 1-rings
        self.oneRings = dict((v,set()) for v in csp.variables)
        for c in csp.constraints:
            cdomain = set(c.domain)
            for v in c.domain:
                self.oneRings[v] |= cdomain
        for v in csp.variables:
            if v in self.oneRings[v]:
                self.oneRings[v].remove(v)

    def solve(self):
        """Solves the CSP, returning an assignment if solved, or False if
        failed."""
        domains = self.initialDomains()
        return self.search({},domains)

    def search(self,partialAssignment,domains):
        """Runs recursive backtracking search."""
        if len(partialAssignment)==len(self.csp.variables):
            return partialAssignment
        if self.doConstraintPropagation:
            domains = self.constraintPropagation(partialAssignment,domains)
            #contradiction detected
            if any(len(d)==0 for (v,d) in domains.iteritems()):
                return False
        indent = " "*len(partialAssignment)
        X = self.pickVariable(partialAssignment,domains)
        values = self.orderValues(partialAssignment,domains,X)
        for v in values:
            partialAssignment[X] = v
            if self.doForwardChecking:
                print indent+"Trying",X,"=",v
                #do forward checking
                newDomains = self.forwardChecking(partialAssignment,X,domains)
                if any(len(d)==0 for (v,d) in newDomains.iteritems()):
                    #contradiction, go on to next value
                    emptyvars = [v for (v,d) in newDomains.iteritems() if len(d)==0]
                    print indent+" Forward checking found contradiction on",emptyvars[0]
                    continue
                #recursive call
                res = self.search(partialAssignment,newDomains)
                if res!=False: return res
            else:
                #check whether the assignment X=v is valid
                if self.csp.isValid(partialAssignment,X):
                    print indent+"Trying",X,"=",v
                    #recursive call
                    res = self.search(partialAssignment,domains)
                    if res!=False: return res
        #remove the partial assignment to X, backtrack
        del partialAssignment[X]
        return False
        
    def initialDomains(self):
        """Does the basic step of checking all unary constraints"""
        domains = dict()
        for v,domain in self.csp.domains.iteritems():
            #save only valid constraints
            vconstraints = self.csp.nAryConstraints(1,v)
            dvalid = [val for val in domain if all(c.predicate(val) for c in vconstraints)]
            domains[v] = dvalid
        return domains

    def pickVariable(self,partialAssignment,domains):
        """Return an unassigned variable to assign next"""
        #TODO: implement heuristics
        #MOST CONSTRAINED -> minimum remaining values (MRV) heuristic
        minDomain = float('inf')
        equal = []
        for v in domains:

        	if (v not in partialAssignment):
        		smallestVar = v
	        	if len(domains[v]) < minDomain:
	        		minDomain = len(domains[v])
	        		smallestVar = v
	        		del equal[:] 
	        		equal.append(smallestVar)
	        	if len(domains[v]) == minDomain:
	        		equal.append(v)

        if len(equal) == 1:
        	return equal[0]
        if len(equal) > 1:
       		mostConstraining = float('-inf')
        	for var in equal:
        		if (var not in partialAssignment):
	        		if len(self.csp.incidentConstraints[var]) > mostConstraining:
	        			mostConstraining = len(self.csp.incidentConstraints[var])
	        			smallestVar = var

        return smallestVar


    def orderValues(self,partialAssignment,domains,var):
        """Return an ordering on the domain domains[var]"""
        #TODO: implement heuristics.  Currently doesn't do anything
        return domains[var]

    def constraintPropagation(self,partialAssignment,domains):
        """domains is a dict mapping vars to valid values.
        Return a copy of domains but with all invalid values removed."""
        #TODO: implement AC3. Currently doesn't do anything
        return domains

        #implemnt constraint checking
        #Forward checking 
    def forwardChecking(self,partialAssignment,var,domains):
        """domains is a dict mapping vars to valid values.  var has just been
        assigned.
        Return a copy of domains but with all invalid values removed"""
        resdomain = dict()
        # shallow copy for all unaffected domains, this saves time
        for v,domain in domains.iteritems():
            resdomain[v] = domain
        resdomain[var] = [partialAssignment[var]]
                
        #TODO: perform forward checking on binary constraints
        for c in self.csp.incidentConstraints[var]:
            kassigned = 0
            unassigned = None
            for v in c.domain:
            	print "v is: ", v
                if v in partialAssignment:
                    kassigned += 1
                else:
                    unassigned = v
            if kassigned+1 == len(c.domain):
                validvalues = []
                diction = dict()
            	for each,domain in partialAssignment.iteritems():
            		diction[each]=domain
                #TODO: check whether each values in the domain of unassigned
                #(resdomain[unassigned]) is compatible under c. May want to use
                #self.csp.isConstraintSatisfied(c,assignment).  If compatible,
                #append the value to validvalue

            	for val in resdomain[unassigned]:
            		diction[unassigned] = val
            		#partialAssignment[unassigned] = val
            		#if self.csp.isConstraintSatisfied(c,partialAssignment):
            		#	validvalues.append(val)
            		if self.csp.isConstraintSatisfied(c,diction):
            			validvalues.append(val)
            		#del partialAssignment[unassigned]
            		del diction[unassigned]

            	resdomain[unassigned] = validvalues
            	if len(validvalues)==0:
            		print "Domain of",unassigned,"emptied due to", c.name
            		#early terminate, this setting is a contradiction
            		return resdomain
        return resdomain

def nQueensCSP(n):
    """Returns a CSP for an n-queens problem"""
    vars = ['Q'+str(i) for i in range(1,n+1)]
    domain = range(1,n+1)
    csp = CSP(vars,[domain]*len(vars))
    for i in range(1,n+1):
        for j in range(1,i):
            Qi = 'Q'+str(i)
            Qj = 'Q'+str(j)
            ofs = i-j
            #this weird default argument thing is needed for lambda closure
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y: x!=y),Qi+"!="+Qj)
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y,ofs=ofs: x!=(y+ofs)),Qi+"!="+Qj+"+"+str(i-j))
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y,ofs=ofs: x!=(y-ofs)),Qi+"!="+Qj+"-"+str(i-j))
    return csp

def p2():
    csp = nQueensCSP(4)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=True)
    res = solver.solve()
    print "Result:",sorted(res.items())

    csp = nQueensCSP(8)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=True)
    res = solver.solve()
    print "Result:",sorted(res.items())
    raw_input()

    csp = nQueensCSP(12)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=True)
    res = solver.solve()
    print "Result:",sorted(res.items())
    raw_input()

    csp = nQueensCSP(12)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=True)
    res = solver.solve()
    print "Result:",sorted(res.items())
    raw_input()

    csp = nQueensCSP(30)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=True)
    res = solver.solve()
    print "Result:",sorted(res.items())



#Implement basic marginalizing and conditioning operations on P(A,B,C) - a given Python dictionary #

def marginalize(probabilities,index):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    """

    #TODO: you may hard-code two routines for n=2 and n=3, but there's an
    #elegant solution that uses defaultdict(float)

    dicta = probabilities.items()
    tup = dicta[0][0]
    tupleSize = len(tup)

    if tupleSize == 2:
     	index = abs(index-1)

    	sumret0 = 0
    	sumret1 = 0
    	for key, val in probabilities.iteritems():
    		if(key[index] == 0):
    			sumret0 = sumret0 + val
    		if(key[index] == 1):
    			sumret1 = sumret1 + val
    	margA = {(0,):sumret0, (1,):sumret1}

    if tupleSize == 3:
    	if index == 2:
    		firstindex = 0		#0
    		secondindex = 1		#1
    	if index == 1:
    		firstindex = 0		#0
    		secondindex = 2		#2
    	if index == 0:
    		firstindex = 1		#1
    		secondindex = 2		#2
    	sumret0 = 0
    	sumret1 = 0
    	sumret0a = 0
    	sumret1a = 0    	
    	for key, val in probabilities.iteritems():
    		#if a and b are 1 -> TT
    		if(key[firstindex]) == 1 and (key[secondindex] == 1) :
    			sumret0 = sumret0 + val
    		#if a is 1 b is zero -> TF
    		if (key[firstindex]) == 1 and (key[secondindex] == 0):
    			sumret0a = sumret0a + val
    		#if a is 0 b is 1 -> FT
    		if(key[firstindex]) == 0 and (key[secondindex] == 1):
    			sumret1 = sumret1 + val
    		#if a is 0 b is 0 -> FT
    		if(key[firstindex]) == 0 and (key[secondindex] == 0):
    			sumret1a = sumret1a + val
    	margB = {(1,1):sumret0,(1,0): sumret0a,(0,1):sumret1,(0,0):sumret1a}

    if tupleSize == 2:
    	return margA
    if tupleSize == 3:
    	return margB
    pass

def marginalize_multiple(probabilities,indices):
    """Safely marginalizes multiple indices"""
    pmarg = probabilities
    for index in reversed(sorted(indices)):
        pmarg = marginalize(pmarg,index)
    return pmarg

def condition1(probabilities,index,value):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn | Xi=v).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    - value: the value of v
    """
    dicta = probabilities.items()
    tup = dicta[0][0]
    tupleSize = len(tup)
    if tupleSize == 2:
    	sum1 = 0
    	sum2 = 0
    	other = abs(index-1)
    	for key, prob in probabilities.iteritems():
    		if ((key[index]) == value) and (key[other] == 0) :
    			sum1 = sum1 + prob
    		if ((key[index]) == value) and (key[other] == 1) :
    			sum2 = sum2 + prob
    	denom  = marginalize(probabilities,other)
    	d = denom[(index,)]
    	div1 = sum1 / d
    	div2 = sum2 / d
    	conditional = {(0,):div1, (1,):div2}
    	return conditional

    if tupleSize == 3:
    	if index == 2:
    		firstindex = abs(index-2)		#0
    		secondindex = abs(index-1)		#1
    	if index == 1:
    		firstindex = abs(index-1)		#0
    		secondindex = abs(index+1)		#2
    	if index == 0:
    		firstindex = 1		#1
    		secondindex = 2		#2
     	sum1 = 0
     	sum2 = 0
     	sum3 = 0
     	sum4 = 0
     	for key, prob in probabilities.iteritems():
     		if ((key[index] == value) and (key[firstindex] == 1)  and (key[secondindex] == 1)) :
     			sum1 = sum1 + prob
     		if ((key[index] == value) and (key[firstindex] == 1) and (key[secondindex] == 0) ):
     			sum2 = sum2 + prob
     		if ((key[index] == value) and (key[firstindex] == 0) and (key[secondindex] == 1) ):
     			sum3 = sum3 + prob
     		if ((key[index] == value )and (key[firstindex] == 0) and (key[secondindex] == 0) ):
     			sum4 = sum4 + prob

     	val = marginalize_multiple(probabilities, [firstindex, secondindex] )
     	d = val[(value,)]
     	div1 = sum1 / d
     	div2 = sum2 / d
     	div3 = sum3 / d
     	div4 = sum4 / d
     	conditional = {(1,1):div1, (1,0):div2, (0,1):div3, (0,0):div4}
     	return conditional

def normalize(probabilities):
    """Given an unnormalized distribution, returns a normalized copy that
    sums to 1."""
    vtotal = sum(probabilities.values())
    return dict((k,v/vtotal) for k,v in probabilities.iteritems())

def condition2(probabilities,index,value):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    returns the distribution P(X1,...,Xi-1,Xi+1,...,Xn | Xi=v).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    - value: the value of v
    """
    #TODO Compute the result by normalizing
    dicta = probabilities.items()
    tup = dicta[0][0]
    tupleSize = len(tup)


    if tupleSize == 2:
    	sum1 = 0
    	sum2 = 0
    	other = abs(index-1)
    	for key, prob in probabilities.iteritems():
    		if ((key[index]) == value) and (key[other] == 0) :
    			sum1 = sum1 + prob
    		if ((key[index]) == value) and (key[other] == 1) :
    			sum2 = sum2 + prob
    	conditional = {(0,):sum1, (1,):sum2}
    	norm  = normalize(conditional)
    	return norm
    if tupleSize == 3:
    	if index == 2:
    		firstindex = abs(index-2)		#0
    		secondindex = abs(index-1)		#1
    	if index == 1:
    		firstindex = abs(index-1)		#0
    		secondindex = abs(index+1)		#2
    	if index == 0:
    		firstindex = 1		#1
    		secondindex = 2		#2
     	sum1 = 0
     	sum2 = 0
     	sum3 = 0
     	sum4 = 0
     	for key, prob in probabilities.iteritems():
     		if (key[index]) == value and (key[firstindex] == 1)  and (key[secondindex] == 1) :
     			sum1 = sum1 + prob
     		if (key[index]) == value and (key[firstindex] == 1) and (key[secondindex] == 0) :
     			sum2 = sum2 + prob
     		if (key[index]) == value and (key[firstindex] == 0) and (key[secondindex] == 1) :
     			sum3 = sum3 + prob
     		if (key[index]) == value and (key[firstindex] == 0) and (key[secondindex] == 0) :
     			sum4 = sum4 + prob
     	idk = marginalize_multiple(probabilities, [firstindex, secondindex] )
     	conditional = {(1,1):sum1, (1,0):sum2, (0,1):sum3, (0,0):sum4}
     	norm2  = normalize(conditional)
     	return norm2
def p4():
    pAB = {(0,0):0.5,
           (0,1):0.3,
           (1,0):0.1,
           (1,1):0.1}
    pA = marginalize(pAB,1)
    print (pA[(0,)],pA[(1,)]),"should be",(0.8,0.2)


    pABC = {(0,0,0):0.2,
            (0,0,1):0.3,
            (0,1,0):0.06,
            (0,1,1):0.24,
            (1,0,0):0.02,
            (1,0,1):0.08,
            (1,1,0):0.06,
            (1,1,1):0.04}

    print "marginalized p(A,B): ",dict(marginalize(pABC,2))
    pA = marginalize(marginalize(pABC,2),1)
    print (pA[(0,)],pA[(1,)]),"should be",(0.8,0.2)

    pA_B = condition1(pAB,1,1)
    print (pA_B[(0,)],pA_B[(1,)]),"should be",(0.75,0.25)

    pA_B = condition2(pAB,1,1)
    print (pA_B[(0,)],pA_B[(1,)]),"should be",(0.75,0.25)

    pAB_C = condition1(pABC,2,1)
    print "p(A,B|C): ",dict(pAB_C)

    pAB_C = condition2(pABC,2,1)
    print "p(A,B|C): ",dict(pAB_C)

    pA_BC = condition1(condition1(pABC,2,1),1,1)
    print "p(A|B,C): ",dict(pA_BC)

    pA_BC = condition2(condition2(pABC,2,1),1,1)
    print "p(A|BC): ",dict(pA_BC)


if __name__=='__main__':
    print "###### 1 ######"
    p1()
    raw_input()
    print
    print "###### 2 ######"
    p2()
    raw_input()
    print
    print "###### 4 ######"
    p4()
    
