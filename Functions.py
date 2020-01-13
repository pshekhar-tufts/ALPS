
#############################################################################################################################
##################################### CREATED BY: PRASHANT SHEKHAR                           ################################
##################################### ALPS python code                                       ################################
#############################################################################################################################

##### This File contains Functions for ALPS in Python
#### First are the set of Basic functions on which P-spline functions are built

## 1: Bspline_Basis(p,i,u,U)
## 2: Bspline_Basis_temp(p,i,u,U,row)
## 3: Derivative_bspline_basis(i,p,k,u,U)
## 4: Knot_pspline(Data,p,n)
## 5: quantile_mine(Data,q,k)
## 6: Kno_pspline_opt(Data,p,n)
## 7: Basis_Pspline(n,p,U,loc)
## 8: Basis_derv_Pspline(n,p,U,loc)
## 9: Penalty_p(q,c)
## 10: XZsigma(B,P,q)


#### Then We have the Higher level functions for Pspline and Derivative Computation

## Model fitting through Generalized Cross Validation
## 1: Var_bounds(Data,B,B_dat,theta,P,lamb,confidence = 0.95)
## 2: Smoothing_cost(lamb,Data,B,q,c,choice)
## 3: Smoothing_par(Data,B,q,c,lamb,choice)
## 4: full_search_nk(Data,p,q)

## Mixed Model Formulation with model fitting through Restricted Maximum Likelihood
## 1: REML(par,Data,X,Z,sigma)
## 2: max_reml(par,Data,X,Z,sigma)
## 3: Inference(Data,Cpred,C,lamb,sig,D,confidence = 0.95)
## 4: Inference_effects(q,Data,Cpred,C,lamb,sig,D,confidence = 0.95)


#### Additional Functionality
## 1: Polynomials_fit(Data,points)
## 2: Outlier(Data,thresh1,thresh2)

#############################################################################################################################

from numpy import *
import pandas as pd
from numpy.linalg import inv,det
from scipy.optimize import minimize
import scipy.stats

########################### ########################### ########################### 
########################### GENERAL FUNCTIONS ####################################  
########################### ########################### ########################### 

def Bspline_Basis(p,i,u,U):
    ## Objective: Compute and return the value of ith basis function at location u
    
    ## Input
    ## 1: p: degree of basis function
    ## 2: i: Basis function index
    ## 3: u: parameter for location of Basis function evaluation
    ## 4: U: Knot vector
    
    ## Output 
    ## 1. ith Basis function evaluated at u
    
    m = len(U)-1
    n = m-p-1
    if (u == U[0] and i == 0) or (u == U[m] and i == n):    ## tackling at the boundary
        N = 1
        return N
    
    if (u == U[0] and i!=0) or (u == U[m] and i!=n):
        N = 0
        return N
    
    if (u<U[i] or u>=U[i+p+1]):
        N = 0
        return N
    
    ## N_i,p will only be non-zero in [u_i,u_{i+p+1}
    ## So there are p+1 intervals to be checked in this range [i,i+1), [i+1,i+2), [i+2,i+3), ..,[i+p,i+p+1)
    
    for d in range(i,i+p+1):
        if u>=U[d] and u<U[d+1]:
            interval = d
                    
    N1 = zeros([p+1,len(U)-1])
    N1[0,interval] = 1
    
    reduce_len = i+p  
    for deg in range(1,p+1):
        for i1 in range(i,reduce_len):
            if N1[deg-1,i1]!=0 and N1[deg-1,i1+1]!=0:
                N1[deg,i1] = ((u-U[i1])/(U[i1+deg]-U[i1]))*N1[deg-1,i1]+((U[i1+deg+1]-u)/(U[i1+deg+1]-U[i1+1]))*N1[deg-1,i1+1]
                
                
            if N1[deg-1,i1]!=0 and N1[deg-1,i1+1]==0:
                N1[deg,i1] = ((u-U[i1])/(U[i1+deg]-U[i1]))*N1[deg-1,i1]
                
                
            if N1[deg-1,i1]==0 and N1[deg-1,i1+1]!=0:
                N1[deg,i1] = ((U[i1+deg+1]-u)/(U[i1+deg+1]-U[i1+1]))*N1[deg-1,i1+1]


        reduce_len = reduce_len-1
        
    return N1[p,i]





def Bspline_Basis_temp(p,i,u,U,row):
    ## Objective: To generate the intermediate basis functions required at the base of the computations for the kth derivative
    
    ## Input:
    ## 1. p: degree of basis function
    ## 2. i: Basis function index
    ## 3. u: parameter for location of Basis function evaluation
    ## 4. U: Knot vector
    ## 5. row: the row number in the original Basis contruction table which is required for derivative computation 
    
    ## Output:
    ## The base of the pyramid for derivative computation
    

    m = len(U)-1
    n = m-p-1
    k = p-row
    vec = zeros([1,k+1])
    if (u == U[0] and i == 0):  ## tackling at the boundary
        vec[0,-1] = 1
        return vec
    
    if (u == U[m] and i == n):  
        vec[0,0] = 1
        return vec

    
    if (u<U[i] or u>=U[i+p+1])and u!=U[m]:
        return vec

    
    for d in range(i,i+p+1):
        if u>=U[d] and u<U[d+1]:
            interval = d
            
    if u ==U[m]:
        interval = n
                    
    N1 = zeros([p+1,len(U)-1])
    N1[0,interval] = 1
    
    reduce_len = i+p  ## because at degree = 1 only i to i + p-1 N needs to be calculated
    for deg in range(1,p+1):  # as bottom row is degree 0 and top row is degree p (bottom row already alloted)
        for i1 in range(i,reduce_len):
            if N1[deg-1,i1]!=0 and N1[deg-1,i1+1]!=0:
                N1[deg,i1] = ((u-U[i1])/(U[i1+deg]-U[i1]))*N1[deg-1,i1]+((U[i1+deg+1]-u)/(U[i1+deg+1]-U[i1+1]))*N1[deg-1,i1+1]
                
                
            if N1[deg-1,i1]!=0 and N1[deg-1,i1+1]==0:
                N1[deg,i1] = ((u-U[i1])/(U[i1+deg]-U[i1]))*N1[deg-1,i1]
                
                
            if N1[deg-1,i1]==0 and N1[deg-1,i1+1]!=0:
                N1[deg,i1] = ((U[i1+deg+1]-u)/(U[i1+deg+1]-U[i1+1]))*N1[deg-1,i1+1]


        reduce_len = reduce_len-1
        
    
    ret = array(N1[row,i:i+k+1]).reshape(1,k+1)
        
    return ret




def Derivative_bspline_basis(i,p,k,u,U):
    ## Objective: computes the derivative of the ith Bspline Basis of degree p at parameter u in U
    ## INPUT:
    ## 1. i: Basis function index
    ## 2. p: degree of basis function
    ## 3. k: order of the derivative
    ## 4. u: Parameter value for the computation of the derivative
    ## 5. U: Knot vector
    
    ## OUTPUT:
    ## 1. Respective derivative value
    
    N_der = zeros([p+1,len(U)-1])
    N_der[p-k,i:i+k+1] = Bspline_Basis_temp(p,i,u,U,p-k)
    term = i+k
    for x in range(p-k+1,p+1):
        for y in range(i,term):
            xcoor = x
            ycoor = y
            term1 = N_der[xcoor-1,ycoor]
            term2 = N_der[xcoor-1,ycoor+1]
            
            if term1!=0 and term2==0:
                N_der[xcoor,ycoor] = x*(term1/(U[y+x]-U[y]))
                
            if term2!=0 and term1==0:
                N_der[xcoor,ycoor] = -x*(term2/(U[y+x+1]-U[y+1]))
                
            if term1!=0 and term2!=0:
                N_der[xcoor,ycoor] = x*(term1/(U[y+x]-U[y])) -x*(term2/(U[y+x+1]-U[y+1]))
            
        term = term-1
        
    return(N_der[p,i])





def Knot_pspline(Data,p,n):
    ## Objective: Compute the knot vector with equidistant knots
    
    ## Input:
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: p: degree
    ## 3: n: number of sections on the curve
    
    ## Output:
    ## 1: U: Knot vector
    
    ## Gives equidistant Knot vector
    U = zeros([n+2*p+1,])
    ## Starting to formulate the Knot vector
    U[p] = Data[0,0]
    U[n+p] = Data[-1,0]
    
    dist = (Data[-1,0] - Data[0,0])/n
    
    count = p+1
    for d in range(n+p):
        U[count] = U[count-1] + dist
        count = count+1 
        
        
    count = p-1
    for d in range(p):
        U[count] = U[count+1] - dist
        count = count-1
    
    return U


def quantile_mine(Data,q,k):
    ## Objective: Computes the quantile value
    
    ### Input
    ## 1: q: The knot number for which location is desired
    ## 2: k: number of sections on the curve
    
    ## Output
    ## 1: computed location 
    
    n1 = Data.shape[0]
    fac = (q/k)*n1
    if fac%1!= 0:
        val = Data[round(fac)-1,0]
    else:
        if fac == n1:
            val = Data[round(fac)-1,0]
        else:
            val = (Data[round(fac)-1,0]+Data[round(fac),0])/2
    return val



def Kno_pspline_opt(Data,p,n):
    ## Objective: Compute the knot vector with data quantile based knots
    
    ## Input:
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: p: degree
    ## 3: n: number of sections on the curve
    
    ## Output:
    ## 1: U: Knot vector
    
    U = zeros([n+2*p+1,])
    ## Starting to formulate the Knot vector
    U[p] = Data[0,0]
    U[n+p] = Data[-1,0]
    dist = (Data[-1,0] - Data[0,0])/n
    
    ### This will write till the end
    count = p+1
    for d in range(n+p):
        U[count] = U[count-1] + dist
        count = count+1 
    
    ## This will overwrite the Useful part
    count = p+1
    for d in range(n):
        U[count] = quantile_mine(Data,d+1,n)
        count = count+1 
        
    count = p-1
    for d in range(p):
        U[count] = U[count+1] - dist
        count = count-1
    
    return U

    
    
    
def Basis_Pspline(n,p,U,loc):
    ## Objective: Compute the Bases matrix at given locations
    ## Input
    ## 1: n: number of sections on the curve
    ## 2: p: degree
    ## 3: U: Knot vector
    ## 4: loc: the locations at which we want basis functions to be evaluated
    
    ## Output
    ## 1: B: bases matrix
    
    num = len(loc)
    
    B = zeros([num,n+p])
    c1 = 0
    for i in range(n+p):
        c2 = 0
        #for u in linspace(U[p],U[n+p],num):
        for u in loc:
            B[c2,c1] = Bspline_Basis(p,i,u,U)
            c2 = c2+1
        c1 = c1+1
        
    return B



def Basis_derv_Pspline(n,p,U,loc):
    
    ## Objective: Compute the derivative bases matrix at given locations
    ## Input
    ## 1: n: number of sections on the curve
    ## 2: p: degree
    ## 3: U: Knot vector
    ## 4: loc: the locations at which we want basis function derivatives to be evaluated
    
    ## Output
    ## 1: Derivative matrix
    
    num = len(loc)
    
    B = zeros([num,n+p])
    c2 = 0
    for u in loc :
        c1 = 0
        #for u in linspace(U[p],U[n+p],num):
        for i in range(n+p):
            B[c2,c1] = Derivative_bspline_basis(i,p,1,u,U)
            c1 = c1+1
        c2 = c2+1
    return B



    
def Penalty_p(q,c):
    ## Objective: Compute the Penalty matrix
    ## Input
    ## 1: q: It is the order of difference which is being considered
    ## 2: c: This is the number of basis vectors under consideration
    
    ## Output
    ## 1: Penalty matrix P
    
    if q == 1:
        D = zeros([c-1,c])
        for i in range(c-1):
            D[i,i] = -1
            D[i,i+1] = 1
    if q == 2:
        D = zeros([c-2,c])
        for i in range(c-2):
            D[i,i]= 1
            D[i,i+1] = -2
            D[i,i+2] = 1
            
    if q == 3:
        D = zeros([c-3,c])
        for i in range(c-3):
            D[i,i] = -1
            D[i,i+1] = 3
            D[i,i+2] = -3
            D[i,i+3] = 1
    
    if q == 4:
        D = zeros([c-4,c])
        for i in range(c-4):
            D[i,i] = 1
            D[i,i+1] =-4
            D[i,i+2] = 4
            D[i,i+3] = -4
            D[i,i+3] = 1
    
    P = D.T.dot(D)
    return P


def XZsigma(B,P,q):
    ## Objective: Compute the decomposed bases X and Z, Combined bases C
    ## Input:
    ## 1: B: Bases function matrix
    ## 2: P: Penalty matrix
    ## 3: q: order of penalty
    
    ## Output:
    ## 1: X, Z: Decomposed bases
    ## 2: C: Combined bases
    ## 3: sigma, D: matrices with singular values

    c = P.shape[0]
    r = c-q
    
    U,s,V = linalg.svd(P, full_matrices=True)
    
    Z = B.dot(U[:,0:r])
    X = B.dot(U[:,r:])
    
    sigma = zeros([r,r])
    sigma[:r,:r] = diag(s[:r])
    
    D = zeros([c,c])
    D[q:,q:] = diag(s[:r])
    
    C = zeros([X.shape[0],X.shape[1]+Z.shape[1]])
    C[:,0:X.shape[1]] = X
    C[:,X.shape[1]:] = Z
    return (X,Z,C,sigma,D)



########################## ########################### ########################### 
### Model fitting through Generalized Cross Validation 
########################### ########################### ########################### 



def Var_bounds(Data,B,B_dat,theta,P,lamb,confidence = 0.95):
    ## Objective: Compute the Confidence Intervals (Normal and t-distribution)
    ## Input:
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: B: bases matrix for prediction
    ## 3: B_data: bases matrix at data locations
    ## 4: theta: coordinate of projection on the bases
    ## 5: P: Penalty matrix
    ## 6: lamb: Optimal lambda computed
    ## 7: confidence: defaults to 95% if no value provided
    
    ## Output
    ## 1: stdev_t: t-distribution bound
    ## 2: stdev_n: normal bound
    
    P = lamb*P
    nr = (Data[:,1].reshape(-1,1) - B_dat.dot(theta)).reshape(-1,1)
    
    #term = inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T.dot(B_dat))
    term = B_dat.dot(inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T))
    
    n = Data.shape[0]
    df_res = n - 2*trace(term) + trace(term.dot(term.T))
    sigmasq = (nr.T.dot(nr))/(df_res)
    sigmasq = sigmasq[0][0]
    std = sqrt(diag(sigmasq*B.dot(inv(B_dat.T.dot(B_dat) + P)).dot(B.T)))
    
    stdev_t = scipy.stats.t.ppf((1+confidence)/2.,df_res)*std
    stdev_n = scipy.stats.norm.ppf((1+confidence)/2.)*std
    return(stdev_t,stdev_n)

def Smoothing_cost(lamb,Data,B,q,c,choice):
    ## Objective: Compute and return the generalization cost
    ## Input:
    ## 1: lamb: Value of the smoothing parameter lambda
    ## 2: Data: dataset with dimensions: number of points x 2
    ## 3: B: Bases matrix at data locations
    ## 4: q: order of penalty
    ## 5: c: Number of basis functions
    ## 6: choice
    
    ## Output
    ## 1: obj: Computed metric value

    P = lamb*Penalty_p(q,c)
    
    H = B.dot(inv(B.T.dot(B)+P)).dot(B.T)
    y_cap = H.dot(Data[:,1].reshape(-1,1))
    
    ## Choice 1: Cross Validation
    if choice == 1:
        n = Data.shape[0]
        t = 0
        for i in range(n):
            t = t+ ((Data[i,1] - y_cap[i])/(1-H[i,i]))**2
        
        #obj = (1/n)*t
        obj = t
    
    ## Choice 2: Generalized Cross Validation
    if choice == 2:
        n = Data.shape[0]
        t = 0
        d = sum(diag(H))/n
        for i in range(n):
            t = t+ ((Data[i,1] - y_cap[i])/(1-d))**2
        
        #obj = (1/n)*t
        obj = t

    return obj
    
    


def Smoothing_par(Data,B,q,c,lamb,choice):
    ## Objective: Compute the optimized value of the hyperparameter lambda
    ## Input
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: B: Bases matrix at data locations
    ## 3: q: order of penalty
    ## 4: c: Number of basis functions
    ## 5: lamb: Initialization for lambda
    ## 6: Choice = 1: Cross Validation and Choice = 2: Generalized Cross Validation
    
    ## Output
    ## 1: Optimal parameter (containing information for optimized cost and corresponding parameter)
    
    args = (Data,B,q,c,choice)
    bnds = [(1.0e-2, None)]
    lamb = [lamb]
    lam = minimize(Smoothing_cost,lamb,args,bounds=bnds,method='SLSQP')
    return lam

    
    

def full_search_nk(Data,p,q):
    ## Objective: Compute Optimal number of sections for given data and corresponding optimal lambda
    ## Input
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: p: degree of bases
    ## 3: q: order of penalty
    
    ## Output
    ## 1. Opt_n: Optimal number of sections
    ## 2. Opt_lam: Corresponding optimal lambda
    ## 3: sigmasq: Fitting Variance
    
    
    n = 1 ## number of sections on the curve
    inc = 1
    fact = 1
    choice = 2  ### always using GCV for now
    comp = 1.0e+9
    #while n<Data.shape[0]-p-1:
    while n<Data.shape[0]:
        c = n+p
        U = Kno_pspline_opt(Data,p,n)
        B = Basis_Pspline(n,p,U,Data[:,0])
        lamb = 0.1
        lam = Smoothing_par(Data,B,q,c,lamb,choice)
        #print(lam.x[0],lam.fun)
        if lam.fun<comp:
            comp = lam.fun
            opt_n = n
            opt_lam = lam.x[0]
            
        n = n+1
        
    ## Computing sig
    c = opt_n+p
    P = opt_lam*Penalty_p(q,c)
    U = Kno_pspline_opt(Data,p,opt_n)
    B_dat = Basis_Pspline(opt_n,p,U,Data[:,0])
    theta = linalg.solve(B_dat.T.dot(B_dat) + P, B_dat.T.dot(Data[:,1].reshape(-1,1)))
    nr = (Data[:,1].reshape(-1,1) - B_dat.dot(theta)).reshape(-1,1)
 
    term = B_dat.dot(inv(B_dat.T.dot(B_dat) + P).dot(B_dat.T))
    n = Data.shape[0]
    df_res = n - 2*trace(term) + trace(term.dot(term.T))
    sigmasq = (nr.T.dot(nr))/(df_res)
    sigmasq = sigmasq[0][0]
    return [opt_n,opt_lam,sigmasq]




########################### ########################### ########################### 
### 2. Mixed Model Formulation with model fitting through Liklihood maximization
########################### ########################### ###########################    
  

def REML(par,Data,X,Z,sigma):
    ## Objective: Compute the REML metric
    ## Input:
    ## 1: par: parameter values for lambda and error variance
    ## 2: Data: dataset with dimensions: number of points x 2
    ## 3: X, Z: Decomposed bases matrices
    ## 4: sigma: S matrix from the SVD of P (c-q x c-q)
    
    ## Output:
    ## 1: reml: value of the metric
    
    lamb = par[0]
    sig = par[1]
    G = sig*(1/lamb)*inv(sigma)
    R = sig*eye(Data.shape[0])
    
    V = Z.dot(G).dot(Z.T) + R
    y = Data[:,1].reshape(-1,1)
    t11 = log(det(V))
    t12 = (y.T.dot(inv(V))).dot(eye(Data.shape[0]) - X.dot(inv(X.T.dot(inv(V)).dot(X))).dot(X.T.dot(inv(V)))).dot(y)
    t1 = -0.5*(t11+t12)
    t2 = (-Data.shape[0]/2)*log(2*pi)
    t3 = -0.5*log(det(X.T.dot(inv(V)).dot(X)))
    
    reml = -(t1 + t2 + t3)  ## extra minus will correspond to minimization of the objective
    
    return reml

def max_reml(par,Data,X,Z,sigma):
    ## Objective: compute the parameters that give maximized REML
    ## Input:
    ## 1: par: parameter values for lambda and error variance
    ## 2: Data: dataset with dimensions: number of points x 2
    ## 3: X,Z: decomposed matrices
    ## 4: sigma: S matrix from the SVD of P (c-q x c-q)
    
    ## Output:
    ## 1: lam: optimal lambda
    ## 2: sig: Optimal variance 

    args = (Data,X,Z,sigma)
    
    bnds = array([(e-2, None),(e-2, None)])
    opt_par = minimize(REML,par,args,bounds = bnds,method='SLSQP')
    lam = opt_par.x[0]
    sig = opt_par.x[1]
    return [lam,sig]


def Inference(Data,Cpred,C,lamb,sig,D,confidence = 0.95):
    ## Objective: Compute the mean prediction and confidence intervals
    ## Input
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: Cpred: Combined bases matrix at prediction points
    ## 3: C: Combined bases matrix at data locations
    ## 4: lamb: lambda value
    ## 5: sig: variance
    ## 6: D: diagonal matrix with singular values
    ## 7: confidence: percentage
    
    ## Output:
    ## 1: f: Mean prediction
    ## 2: stdev_t: t-CI
    ## 3: stdev_n: Normal CI

    term = inv(C.T.dot(C) + lamb*D)
    f = Cpred.dot(term).dot(C.T.dot(Data[:,1].reshape(-1,1)))
    

    Slam = C.dot(term).dot(C.T)
    df_res = Data.shape[0]-2*trace(Slam) + trace(Slam.dot(Slam.T))
    se = sqrt(diag(sig*Cpred.dot(term).dot(Cpred.T)))
    stdev_t = scipy.stats.t.ppf((1+confidence)/2.,df_res)*se
    stdev_n = scipy.stats.norm.ppf((1+confidence)/2.)*se
    return(f,stdev_t,stdev_n)


def Inference_effects(q,Data,Cpred,C,lamb,D):
    ## Objective: Compute the high frequency and low frequency component
    ## Input:
    ## 1: q: Order of penalty
    ## 2: Data: dataset with dimensions: number of points x 2
    ## 3: Cpred: Combined bases matrix at prediction points
    ## 4: C: Combined bases matrix at data locations
    ## 5: lamb: lambda value
    ## 6: D: diagonal matrix with singular values
    
    ## Output:
    ## 1: f_low: low frequency component
    ## 2: f_high: high frequenc component
    
    term = inv(C.T.dot(C) + lamb*D)
    beta_alpha = term.dot(C.T.dot(Data[:,1].reshape(-1,1)))    
    f_low = Cpred[:,:q].dot(beta_alpha[:q].reshape(-1,1))
    f_high = Cpred[:,q:].dot(beta_alpha[q:].reshape(-1,1))
    return(f_low,f_high)


###########################################################################################################################
#### Polynomials ############################################
###########################################################################################################################

def Polynomials_fit(Data,points):
    ## Objective: Comoute the polynomial approximation to a given dataset
    ## Input
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: points: prediction points
    
    ## Output
    ## 1: p: Cubic polynomial prediction
    ## 2: r: Residual of prediction

    
    vander = []
    for x in Data:
        vander.append([1,x[0],x[0]**2, x[0]**3])
        
    vander = array(vander)
    y = array(Data[:,1]).reshape(vander.shape[0],1)
    param = array(linalg.lstsq(vander,y,rcond=None)[0])
    
    
    ## Prediction at the new points
    vandp = []
    
    for xc in points:
        vandp.append([1,xc,xc**2, xc**3])
    vandp = array(vandp)
    
    prediction = vandp.dot(param)
    p = zeros([prediction.shape[0],2])
    p[:,0] = points.flatten()
    p[:,1] = prediction.flatten()
    
    ## Computing the Residual on the original points
    ress = vander.dot(param)
    res = []
    c = 0
    for x in ress:
        res.append(Data[c,1]-x)
        c = c+1
    
    res = array(res)
    r = zeros(Data.shape)
    r[:,0] = Data[:,0].flatten()
    r[:,1] = res.flatten()
    return [p,r]


###########################################################################################################################
### Outlier Detection
###########################################################################################################################

def Outlier(Data,thresh1,thresh2):
    ## Objective: Compute the outliers in a given dataset
    ## Input
    ## 1: Data: dataset with dimensions: number of points x 2
    ## 2: thresh1: threshold 1 for scaling the interval
    ## 3: thresh2: threshold 2 for scaling the interval
    
    ## Output
    ## 1: Dataa: Clean Data
    ## 2: point: Outliers detected
    
    p = 4
    q = 2
    [n,lamb,sigmasq] = full_search_nk(Data,p,q)
    c = n+p
    U = Kno_pspline_opt(Data,p,n)
    B = Basis_Pspline(n,p,U,Data[:,0])
    P = Penalty_p(q,c)
    theta = linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(Data[:,1].reshape(-1,1)))
    xpred = Data[:,0]
    Bpred = Basis_Pspline(n,p,U,xpred)
    ypred1 = Bpred.dot(theta)
    std_t1,std_n1 = Var_bounds(Data,Bpred,B,theta,P,lamb,0.99)
    r_long = zeros([Data.shape[0],2])
    r_long[:,0] = Data[:,0]
    r_long[:,1] = Data[:,1]-ypred1.flatten()
    

    point = []
    for d in range(r_long.shape[0]):
        if abs(r_long[d,1])>std_t1[d]*thresh1:
            point.append([Data[d,0],Data[d,1]])
            
    point = array(point)
    if len(point)>0:
        Dat_temp = []
        for h in range(Data.shape[0]):
            if Data[h,0] not in point[:,0]:
                Dat_temp.append([Data[h,0],Data[h,1]])
                
        Dat_temp = array(Dat_temp)
        [n,lamb,sigmasq] = full_search_nk(Dat_temp,p,q)
        c = n+p
        U = Kno_pspline_opt(Dat_temp,p,n)
        B = Basis_Pspline(n,p,U,Dat_temp[:,0])
        P = Penalty_p(q,c)
        theta = linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(Dat_temp[:,1].reshape(-1,1)))
        xpred = Dat_temp[:,0]
        Bpred = Basis_Pspline(n,p,U,xpred)
        ypred2 = Bpred.dot(theta)
        std_t2,std_n2 = Var_bounds(Dat_temp,Bpred,B,theta,P,lamb,0.99)
        r_short = zeros([Dat_temp.shape[0],2])
        r_short[:,0] = Dat_temp[:,0]
        r_short[:,1] = ypred2.flatten() - Dat_temp[:,1]
        
        point = point.tolist()
        for d in range(r_short.shape[0]):
            if abs(r_short[d,1])>std_t2[d]*thresh2:
                point.append([Dat_temp[d,0],Dat_temp[d,1]])
                
        point = array(point)

        
    if len(point) == 0:
        Dat_temp = Data
        [n,lamb,sigmasq] = full_search_nk(Dat_temp,p,q)
        c = n+p
        U = Kno_pspline_opt(Dat_temp,p,n)
        B = Basis_Pspline(n,p,U,Dat_temp[:,0])
        P = Penalty_p(q,c)
        theta = linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(Dat_temp[:,1].reshape(-1,1)))
        xpred = Dat_temp[:,0]
        Bpred = Basis_Pspline(n,p,U,xpred)
        ypred2 = Bpred.dot(theta)
        std_t2,std_n2 = Var_bounds(Dat_temp,Bpred,B,theta,P,lamb,0.99)
        r_short = zeros([Dat_temp.shape[0],2])
        r_short[:,0] = Dat_temp[:,0]
        r_short[:,1] = Dat_temp[:,1]-ypred2.flatten()
        
        point = point.tolist()
        for d in range(r_short.shape[0]):
            if abs(r_short[d,1])>std_t2[d]*thresh2:
                point.append([Dat_temp[d,0],Dat_temp[d,1]])
                
        point = array(point)
        
        
    ### Segregating the Dataset and outliers
    Dataa = []
    if len(point)>0:
        for i in range(Data.shape[0]):
            if Data[i,0] not in point[:,0]:
                Dataa.append([Data[i,0],Data[i,1]])
            
        Dataa = array(Dataa)
    if len(point) == 0:
        Dataa = Data
    
    return (Dataa,point)


###########################################################################################################################
###########################################################################################################################
    