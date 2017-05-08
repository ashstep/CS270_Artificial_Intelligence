'''
    An algorithm for learning spam filter probabilities from data.
    Created on Mar 24, 2017 
    @author: Ashka Stephen 
'''

def uniform(domain):
    """Return a uniform distribution over the given domain"""
    return dict((v,1.0/len(domain)) for v in domain)

def valid_probability_distribution(p):
    if any(v<0 for v in p.values()): return False
    if(abs(sum(p.values())-1.0) > 1e-5): return False
    return True

def learn_discrete(dataset,virtual_count=1,domain=None):
    """Given a list of values in dataset, learns a discrete distribution
    over the value."""
    if(domain==None):
        domain = set(dataset)
    #TODO: generate a distribution over the domain from the data 
    #TODO: take into account the virtual counts
    final = {}
    #initlaize to zero
    for each in domain:
        final[each] = 0

    #returns the counts in a map
    for c in dataset:
        if c in final.keys():
            final[c] = final[c] + 1
        else:
            final[c] = 1
    #add in VC to numerators
    for each in final:
        final[each] =  final[each] + virtual_count

    #get total (includes all VCs)
    total = 0
    for a in final:
        total = total + final[a]

    #divide by total -> final ratio
    for fin in final:
        final[fin] = float(final[fin]) / total

    #checks if final is one
    return final

def learn_naive_bayes(class_key,feature_keys,
                      dataset,
                      class_prior_count=1,feature_posterior_count=1,
                      class_domain=None,feature_domains=None):
    """Estimating a Naive Bayes model from data.  Given a list of instances,
    learns a class prior P(C) and feature posteriors P(F1|C),...,P(Fk|C).
    Returns a pair (PC,PF) where PF is a dictionary mapping feature names
    to a conditional distributions.  Like in problem 1, a conditional
    distribution p gives P(F=f|C=f) in a table p[c_value][f_value].
    
    The prior counts are "virtual counts" for each value in the class's domain
    and the features' domains.

    pf is features to the conditional

    """
    if class_domain == None:
        #compute the set of values that the class can take on
        class_domain = set([instance[class_key] for instance in dataset])
    if feature_domains == None:
        #compute the set of values that the features can take on
        feature_domains = dict()
    for f in feature_keys:
        if f not in feature_domains:
            feature_domains[f] = set([instance[f] for instance in dataset])

    #TODO: replace these lines with your work
    #create a uniform class prior
    PCuniform = uniform(class_domain)
    PFuniform = dict()

    #create uniform feature priors
    PFlearned = dict()
    for f in feature_keys:
        PFf = dict()
        #for all values v in the class domain, PFf[v] is a distribution over f's
        for class_v in class_domain:
            PFf[class_v] = learn_discrete([instance[f] for instance in dataset if class_v == instance[class_key]], feature_posterior_count, feature_domains[f])
        PFlearned[f] = PFf

    PClearned = learn_discrete([instance[class_key] for instance in dataset],
                                class_prior_count,
                                class_domain)    
    print "PFlearned, ", PFlearned
    return (PClearned,PFlearned)

def p2():
    labeled_instances = [
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':0},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':0,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':0,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':0,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Not-Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Spam','f1':1,'f2':0,'f3':0,'f4':0},
        {'Label':'Spam','f1':1,'f2':1,'f3':1,'f4':1},
        {'Label':'Spam','f1':1,'f2':1,'f3':1,'f4':0},
        {'Label':'Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Spam','f1':0,'f2':1,'f3':0,'f4':1},
        {'Label':'Spam','f1':1,'f2':0,'f3':1,'f4':1},
        {'Label':'Spam','f1':1,'f2':0,'f3':1,'f4':1},
        {'Label':'Spam','f1':1,'f2':0,'f3':1,'f4':1},
        ]
    (PC,PF) = learn_naive_bayes('Label',['f1','f2','f3','f4'],
                                labeled_instances)
    print "Spam prior:",PC["Spam"]
    for f,PFf in PF.iteritems():
        print PFf
        print f," given Spam:",PFf["Spam"][1]
        print f," given Not-Spam:",PFf["Not-Spam"][1]
        
    assert(valid_probability_distribution(PC)),"Class prior is invalid"
    for f,PFf in PF.iteritems():
        assert(valid_probability_distribution(PFf["Spam"])),"Feature "+f+" given Spam is invalid"
        assert(valid_probability_distribution(PFf["Not-Spam"])),"Feature "+f+" given Not-Spam is invalid"

if __name__=="__main__":
    p2()
    
