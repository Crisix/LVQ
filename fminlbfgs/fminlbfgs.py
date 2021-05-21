import numpy as np


def isempty(x):
    return len(x) == 0


def eps(x=None):
    return np.spacing(1)


def getexitmessage(exitflag):  # returns message
    result = {
            1: "Change in the objective function value was less than the specified tolerance TolFun.",
            2: "Change in x was smaller than the specified tolerance TolX.",
            3: "Magnitude of gradient smaller than the specified tolerance",
            4: "Boundary fminimum reached.",
            0: "Number of iterations exceeded options.MaxIter or number of function evaluations exceeded options.FunEvals.",
            -1: "Algorithm was terminated by the output function.",
            -2: "Line search cannot find an acceptable point along the current search",
    }
    if exitflag != [] and exitflag in result:
        return result[exitflag]
    return 'Undefined exit code'


def call_output_function(data, optim, where):  # returns stopt
    stopt = False
    if "OutputFcn" in optim.keys():
        output = {
                "iteration": data.iteration,
                "funccount": data.funcCount,
                "fval": data.fInitial,
                "stepsize": data.alpha,
                "directionalderivative": data.fPrimeInitial,
                "gradient": data.gradient.reshape(data.xsizes),
                "searchdirection": data.dir,
        }
        stopt = optim["OutputFcn"](data.xInitial.reshape(data.xsizes), output, where)
    return stopt


class Data:

    def __init__(self, x_init: np.ndarray):
        self.fval = 0
        self.gradient = 0
        self.fOld = []
        self.xsizes = len(x_init)
        self.numberOfVariables = x_init.size
        self.xInitial = x_init.copy()
        self.alpha = 1
        self.xOld = self.xInitial
        self.iteration = 0
        self.funcCount = 0
        self.gradCount = 0
        self.exitflag = []
        self.nStored = 0
        self.timeExtern = 0
        self.nStored = 0

        self.deltaX = None
        self.deltaG = None
        self.saveD = None

        self.initialStepLength = None

        self.storefx = None
        self.storepx = None
        self.storex = None
        self.storegx = None


def gradient_function(x, funfcn, data, optim, calc_grad):  # returns [data,fval,grad]
    # Call the error function for error (and gradient)
    if not calc_grad:
        fval, grad = funfcn(x.reshape(data.xsizes, -1))
        data.funcCount = data.funcCount + 1
        return data, fval
    else:
        if optim["GradObj"] == 'on':
            fval, grad = funfcn(x.reshape(data.xsizes))
            # fval, grad = funfcn(x.reshape(data.xsizes, -1))  # TODO
            data.funcCount = data.funcCount + 1
            data.gradCount = data.gradCount + 1
            grad = grad.reshape(-1)
            return data, fval, grad
        else:
            print("gradient function must be given")
            exit(0)
            """
            # Calculate gradient with forward difference if not provided by the function
            grad=zeros(length(x),1);
            fval=funfcn(reshape(x,data.xsizes));
            gstep=data.initialStepLength/1e6; 
            if(gstep>optim.DiffMaxChange), gstep=optim.DiffMaxChange; end
            if(gstep<optim.DiffMinChange), gstep=optim.DiffMinChange; end
            for i in range(len(x)):
                x_temp=x; x_temp(i)=x_temp(i)+gstep;
                [fval_g]=feval(funfcn,reshape(x_temp,data.xsizes)); data.funcCount=data.funcCount+1;
                grad(i)=(fval_g-fval)/gstep;
            grad=grad(:)
            return data, fval, grad
            """


def matlab_fmin_lbfgs(funfcn, x_init, optim: dict = None):  # return x, fval, exitfalg, output, grad
    """
    %FMINLBFGS finds a local minimum of a function of several variables.
    %   This optimizer is developed for image registration methods with large
    %	amounts of unknown variables.
    %
    %   Optimization methods supported:
    %	- Quasi Newton Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    %   - Limited memory BFGS (L-BFGS)
    %   - Steepest Gradient Descent optimization.
    %
    %   [X,FVAL,EXITFLAG,OUTPUT,GRAD] = FMINLBFGS(FUN,X0,OPTIONS)
    %
    %   Inputs,
    %		FUN: Function handle or string which is minimized, returning an
    %				error value and optional the error gradient.
    %		X0: Initial values of unknowns can be a scalar, vector or matrix
    %	 (optional)
    %		OPTIONS: Structure with optimizer options, made by a struct or
    %				optimset. (optimset doesnot support all input options)
    %
    %   Outputs,
    %		X : The found location (values) which minimize the function.
    %		FVAL : The minimum found
    %		EXITFLAG : Gives value, which explain why the minimizer stopt
    %		OUTPUT : Structure with all important ouput values and parameters
    %		GRAD : The gradient at this location
    %
    %   Extended description of input/ouput variables
    %   OPTIONS,
    %		OPTIONS.GoalsExactAchieve : If set to 0, a line search method is
    %               used which uses a few function calls to do a good line
    %               search. When set to 1 a normal line search method with Wolfe
    %				conditions is used (default).
    %		OPTIONS.GradConstr, Set this variable to true if gradient calls are
    %				cpu-expensive (default). If false more gradient calls are
    %				used and less function calls.
    %	    OPTIONS.HessUpdate : If set to 'bfgs', Broyden�Fletcher�Goldfarb�Shanno
    %				optimization is used (default), when the number of unknowns is
    %				larger then 3000 the function will switch to Limited memory BFGS,
    %				or if you set it to 'lbfgs'. When set to 'steepdesc', steepest
    %				decent optimization is used.
    %		OPTIONS.StoreN : Number of itterations used to approximate the Hessian,
    %			 	in L-BFGS, 20 is default. A lower value may work better with
    %				non smooth functions, because than the Hessian is only valid for
    %				a specific position. A higher value is recommend with quadratic equations.
    %		OPTIONS.GradObj : Set to 'on' if gradient available otherwise finited difference
    %				is used.
    %     	OPTIONS.Display : Level of display. 'off' displays no output; 'plot' displays
    %				all linesearch results in figures. 'iter' displays output at  each
    %               iteration; 'final' displays just the final output; 'notify'
    %				displays output only if the function does not converge;
    %	    OPTIONS.TolX : Termination tolerance on x, default 1e-6.
    %	    OPTIONS.TolFun : Termination tolerance on the function value, default 1e-6.
    %		OPTIONS.MaxIter : Maximum number of iterations allowed, default 400.
    % 		OPTIONS.MaxFunEvals : Maximum number of function evaluations allowed,
    %				default 100 times the amount of unknowns.
    %		OPTIONS.DiffMaxChange : Maximum stepsize used for finite difference gradients.
    %		OPTIONS.DiffMinChange : Minimum stepsize used for finite difference gradients.
    %		OPTIONS.OutputFcn : User-defined function that an optimization function calls
    %				at each iteration.
    %		OPTIONS.rho : Wolfe condition on gradient (c1 on wikipedia), default 0.01.
    %		OPTIONS.sigma : Wolfe condition on gradient (c2 on wikipedia), default 0.9.
    %		OPTIONS.tau1 : Bracket expansion if stepsize becomes larger, default 3.
    %		OPTIONS.tau2 : Left bracket reduction used in section phase,
    %		default 0.1.
    %		OPTIONS.tau3 : Right bracket reduction used in section phase, default 0.5.
    %   FUN,
    %		The speed of this optimizer can be improved by also providing
    %   	the gradient at X. Write the FUN function as follows
    %   	function [f,g]=FUN(X)
    %       	f , value calculation at X;
    %   	if ( nargout > 1 )
    %       	g , gradient calculation at X;
    %   	end
    %	EXITFLAG,
    %		Possible values of EXITFLAG, and the corresponding exit conditions
    %		are
    %  		1, 'Change in the objective function value was less than the specified tolerance TolFun.';
    %  		2, 'Change in x was smaller than the specified tolerance TolX.';
    %  		3, 'Magnitude of gradient smaller than the specified tolerance';
    %  		4, 'Boundary fminimum reached.';
    %  		0, 'Number of iterations exceeded options.MaxIter or number of function evaluations exceeded options.FunEvals.';
    %  		-1, 'Algorithm was terminated by the output function.';
    %  		-2, 'Line search cannot find an acceptable point along the current search';
    %
    %   Examples
    %       options = optimset('GradObj','on');
    %       X = fminlbfgs(@myfun,2,options)
    %
    %   	% where myfun is a MATLAB function such as:
    %       function [f,g] = myfun(x)
    %       f = sin(x) + 3;
    %	    if ( nargout > 1 ), g = cos(x); end
    %
    %   See also OPTIMSET, FMINSEARCH, FMINBND, FMINCON, FMINUNC, @, INLINE.
    %
    %   Function is written by D.Kroon University of Twente (March 2009)
    """

    # original
    defaultopt = {
            'Display': 'off',  # 'final', 'iter
            'HessUpdate': 'bfgs',
            'GoalsExactAchieve': 1,
            'GradConstr': True,
            'TolX': 1e-6,
            'TolFun': 1e-6,
            'GradObj': 'on',
            'MaxIter': 400,
            'MaxFunEvals': 100 * x_init.size - 1,
            'DiffMaxChange': 1e-1,
            'DiffMinChange': 1e-8,
            'rho': 0.0100,
            'sigma': 0.900,
            'tau1': 3,
            'tau2': 0.1,
            'tau3': 0.5,
            'StoreN': 10,
    }

    if optim is None:
        optim = defaultopt
    else:
        for f in defaultopt.keys():
            if f not in optim:
                optim[f] = defaultopt[f]

    # Initialize the data structure
    data = Data(x_init)

    # Switch to L-BFGS in case of more than 3000 unknown variables
    if optim["HessUpdate"][0] == 'b':
        if data.numberOfVariables < 3000:
            optim["HessUpdate"] = 'bfgs'
        else:
            optim["HessUpdate"] = 'lbfgs'

    if optim["HessUpdate"][0] == 'l':
        data.deltaX = np.zeros((data.numberOfVariables, optim["StoreN"]))
        data.deltaG = np.zeros((data.numberOfVariables, optim["StoreN"]))
        data.saveD = np.zeros((data.numberOfVariables, optim["StoreN"]))

    exitflag = []

    # Display column headers
    if optim["Display"] == 'iter':
        print('     Iteration  Func-count   Grad-count         f(x)         Step-size')

    # Calculate the initial error and gradient
    data.initialStepLength = 1
    data, fval, grad = gradient_function(data.xInitial, funfcn, data, optim, calc_grad=True)
    data.gradient = grad
    data.dir = -data.gradient
    data.gOld = grad
    data.fInitial = fval
    data.fPrimeInitial = data.gradient.T @ data.dir.reshape(-1)

    gNorm = np.linalg.norm(data.gradient, np.inf)  # Norm of gradient
    data.initialStepLength = min(1 / gNorm, 5)

    # Show the current iteration
    if optim["Display"] == 'iter':
        print(f'     {data.iteration:5.0f}       {data.funcCount:5.0f}       {data.gradCount:5.0f}       {data.fInitial}    ')

    # Hessian intialization
    if optim["HessUpdate"][0] == 'b':
        data.Hessian = np.eye(data.numberOfVariables)

    # Call output function
    if (call_output_function(data, optim, 'init')):
        exitflag = -1

    # start minimizing
    while True:
        data.iteration = data.iteration + 1

        # Set current lineSearch parameters
        data.TolFunLnS = eps(max(1, abs(data.fInitial)))  # TODO py=2.220446049250313e-16 vs matlab=2.842170943040401e-14
        data.fminimum = data.fInitial - 1e16 * (1 + abs(data.fInitial))

        # Make arrays to store linesearch results
        data.storefx = []
        data.storepx = []
        data.storex = []
        data.storegx = []

        # Find a good step size in the direction of the gradient: Linesearch
        if optim["GoalsExactAchieve"] == 1:
            data = linesearch(funfcn, data, optim)
        else:
            data = linesearch_simple(funfcn, data, optim)

        # Check if exitflag is set
        if data.exitflag:
            exitflag = data.exitflag
            data.xInitial = data.xOld
            data.fInitial = data.fOld
            data.gradient = data.gOld
            break

        # Update x with the alpha step
        data.xInitial = data.xInitial + data.alpha * data.dir

        # Set the current error and gradient
        data.fInitial = data.f_alpha
        data.gradient = data.grad

        # Set initial steplength to 1
        data.initialStepLength = 1

        gNorm = np.linalg.norm(data.gradient, np.inf)  # Norm of gradient

        # Set exit flags
        if gNorm < optim["TolFun"]:
            exitflag = 1
        if max(abs(data.xOld - data.xInitial)) < optim["TolX"]:
            exitflag = 2
        if data.iteration >= optim["MaxIter"]:
            exitflag = 0

        # Check if exitflag is set
        if exitflag != []:  # "~isempty(exitflag)"
            break

        # Update the inverse Hessian matrix
        if optim["HessUpdate"][0] != 's':  # TODO wann s wenn lbfgs | bfgs
            # Do the Quasi-Neton Hessian update.
            data = updateQuasiNewtonMatrix_LBFGS(data, optim)
        else:
            data.dir = -data.gradient

        # Derivative of direction
        data.fPrimeInitial = data.gradient.T @ data.dir.reshape(-1)

        # Call output function
        if call_output_function(data, optim, 'iter'):
            exitflag = -1

        # Show the current iteration
        if optim["Display"][0] == 'i' or optim["Display"][0] == 'p':
            print(f'     {data.iteration:5.0f}       {data.funcCount:5.0f}       {data.gradCount:5.0f}       {data.fInitial}   {data.alpha:13.6f}')

        # Keep the variables for next iteration
        data.fOld = data.fInitial
        data.xOld = data.xInitial
        data.gOld = data.gradient

    # Set output parameters
    fval = data.fInitial
    grad = data.gradient
    x = data.xInitial

    # Reshape x to original shape
    x = x.reshape(data.xsizes)

    # Call output function
    if call_output_function(data, optim, 'done'):
        exitflag = -1

    # Make exist output structure
    output = {}
    if optim["HessUpdate"][0] == 'b':
        output["algorithm"] = 'Broyden-Fletcher-Goldfarb-Shanno (BFGS)'
    elif optim["HessUpdate"][0] == 'l':
        output["algorithm"] = 'limited memory BFGS (L-BFGS)'
    else:
        output["algorithm"] = 'Steepest Gradient Descent'

    output["message"] = getexitmessage(exitflag)
    output["iteration"] = data.iteration
    output["funccount"] = data.funcCount
    output["fval"] = data.fInitial
    output["stepsize"] = data.alpha
    output["directionalderivative"] = data.fPrimeInitial
    output["gradient"] = data.gradient.reshape(data.xsizes)
    output["searchdirection"] = data.dir
    output["timeExtern"] = data.timeExtern

    # Display final results
    if optim["Display"] != 'off':
        print('    Optimizer Results')
        print('        Algorithm Used: ', output["algorithm"])
        print('        Exit message : ', output["message"])
        print('        iterations : ', str(data.iteration))
        print('        Function Count : ', str(data.funcCount))
        print('        Minimum found : ', str(fval))

    return x, fval, exitflag, output, grad, data.iteration


###################################################

def linesearch_simple(funfcn, data, optim):
    # Find a bracket of acceptable points
    data = bracketingPhase_simple(funfcn, data, optim)

    if (data.bracket_exitflag == 2):
        # BracketingPhase found a bracket containing acceptable points;
        # now find acceptable point within bracket
        data = sectioningPhase_simple(funfcn, data, optim)
        data.exitflag = data.section_exitflag
    else:
        # Already acceptable point found or MaxFunEvals reached
        data.exitflag = data.bracket_exitflag
    return data


def bracketingPhase_simple(funfcn, data, optim):
    # Number of itterations
    itw = 0

    # Point with smaller value, initial
    data.beta = 0
    data.f_beta = data.fInitial
    data.fPrime_beta = data.fPrimeInitial

    # Initial step is equal to alpha of previous step.
    alpha = data.initialStepLength

    # Going up hill
    hill = False

    # Search for brackets
    while True:
        # Calculate the error registration gradient
        if optim["GradConstr"]:
            data, f_alpha = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
            fPrime_alpha = float("NaN")
            grad = np.array(float("NaN"))
        else:
            data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
            fPrime_alpha = grad.T @ data.dir.reshape(-1)

        # Store values linesearch
        data.storefx.append(f_alpha)
        data.storepx.append(fPrime_alpha)
        data.storex.append(alpha)
        data.storegx.append(grad.reshape(-1))

        # Update step value
        if data.f_beta < f_alpha:
            # Go to smaller stepsize
            alpha = alpha * optim["tau3"]

            # Set hill variable
            hill = True
        else:
            # Save current minium point
            data.beta = alpha
            data.f_beta = f_alpha
            data.fPrime_beta = fPrime_alpha
            data.grad = grad
            if not hill:
                alpha = alpha * optim["tau1"]

        # Update number of loop iterations
        itw = itw + 1

        if (itw > (np.log(optim["TolFun"]) / np.log(optim["tau3"]))):
            # No new optium found, linesearch failed.
            data.bracket_exitflag = -2
            break

        if (data.beta > 0 and hill):
            # Get the brackets around minimum point
            # Pick bracket A from stored trials

            # MATLBLA: t, i = sort(data.storex, 'ascend')
            # t = np.sort(data.storex)
            i = np.argsort(data.storex)

            storefx = np.array(data.storefx)[i]
            storepx = np.array(data.storepx)[i]
            storex = np.array(data.storex)[i]
            # t, i = find(storex > data.beta, 1)
            i = np.where(storex > data.beta)[0]  # returns tuple
            if len(i) == 0:
                i = np.where(storex == data.beta)[0]
            i = i[0]
            alpha = storex[i]
            f_alpha = storefx[i]
            fPrime_alpha = storepx[i]

            # % Pick bracket B from stored trials
            i = np.argsort(data.storex)[::-1]
            storefx = np.array(data.storefx)[i]
            storepx = np.array(data.storepx)[i]
            storex = np.array(data.storex)[i]
            i = np.where(storex < data.beta)[0][:1]
            if isempty(i):
                i = np.where(storex == data.beta)[0]
            i = i[0]
            beta = storex[i]
            f_beta = storefx[i]
            fPrime_beta = storepx[i]

            # % Calculate derivatives if not already calculated
            if (optim["GradConstr"]):
                gstep = data.initialStepLength / 1e6
                if (gstep > optim["DiffMaxChange"]):
                    gstep = optim["DiffMaxChange"]
                if (gstep < optim["DiffMinChange"]):
                    gstep = optim["DiffMinChange"]
                data, f_alpha2 = gradient_function(data.xInitial.reshape(-1) + (alpha + gstep) * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
                data, f_beta2 = gradient_function(data.xInitial.reshape(-1) + (beta + gstep) * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
                fPrime_alpha = (f_alpha2 - f_alpha) / gstep
                fPrime_beta = (f_beta2 - f_beta) / gstep

            # Set the brackets A and B
            data.a = alpha
            data.f_a = f_alpha
            data.fPrime_a = fPrime_alpha
            data.b = beta
            data.f_b = f_beta
            data.fPrime_b = fPrime_beta

            # Finished bracketing phase
            data.bracket_exitflag = 2
            return data

        # Reached max function evaluations
        if data.funcCount >= optim["MaxFunEvals"]:
            data.bracket_exitflag = 0
            return data
    return data


###########################################

def sectioningPhase_simple(funfcn, data, optim):
    # Get the brackets
    brcktEndpntA = data.a
    brcktEndpntB = data.b

    # Calculate minimum between brackets
    alpha, f_alpha_estimated = pickAlphaWithinInterval(brcktEndpntA, brcktEndpntB, data.a, data.b, data.f_a, data.fPrime_a, data.f_b, data.fPrime_b, optim)
    if 'beta' in data.__dict__ and (data.f_beta < f_alpha_estimated):
        alpha = data.beta

    i = np.where(data.storex == alpha)[0]
    if (not isempty(i)) and (not np.all(np.isnan(data.storegx[i[0]]))):
        i = i[0]
        f_alpha = np.array(data.storefx)[i]
        # grad = np.array(data.storegx)[:, i] # vorher
        grad = np.array(data.storegx)[i]
    else:
        # Calculate the error and gradient for the next minimizer itteration
        data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
        if 'beta' in data.__dict__ and (data.f_beta < f_alpha):
            alpha = data.beta
            if (not isempty(i)) and (not np.isnan(data.storegx[i])):
                f_alpha = data.storefx[i]
                grad = data.storegx[:, i]
            else:
                data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)

    # Store values linesearch
    data.storefx.append(f_alpha)
    data.storex.append(alpha)

    fPrime_alpha = grad.T @ data.dir.reshape(-1)
    data.alpha = alpha
    data.fPrime_alpha = fPrime_alpha
    data.f_alpha = f_alpha
    data.grad = grad

    # Set the exit flag to succes
    data.section_exitflag = []
    return data


def linesearch(funfcn, data, optim):
    # Find a bracket of acceptable points
    data = bracketingPhase(funfcn, data, optim)

    if data.bracket_exitflag == 2:
        # BracketingPhase found a bracket containing acceptable points
        # now find acceptable point within bracket
        data = sectioningPhase(funfcn, data, optim)
        data.exitflag = data.section_exitflag
    else:
        # Already acceptable point found or MaxFunEvals reached
        data.exitflag = data.bracket_exitflag
    return data


def sectioningPhase(funfcn, data, optim):
    # sectioningPhase finds an acceptable point alpha within a given bracket [a,b]
    # containing acceptable points. Notice that funcCount counts the total number of
    # function evaluations including those of the bracketing phase.

    while True:

        # Pick alpha in reduced bracket
        brcktEndpntA = data.a + min(optim["tau2"], optim["sigma"]) * (data.b - data.a)
        brcktEndpntB = data.b - optim["tau3"] * (data.b - data.a)

        # Find global minimizer in bracket [brcktEndpntA,brcktEndpntB] of 3rd-degree
        # polynomial that interpolates f() and f'() at "a" and at "b".
        alpha, _ = pickAlphaWithinInterval(brcktEndpntA, brcktEndpntB, data.a, data.b, data.f_a, data.fPrime_a, data.f_b, data.fPrime_b, optim)

        # No acceptable point could be found
        if abs((alpha - data.a) * data.fPrime_a) <= data.TolFunLnS:
            data.section_exitflag = -2
            return data

            # Calculate value (and gradient if no extra time cost) of current alpha
        if (not optim["GradConstr"]):
            data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
            fPrime_alpha = grad.T @ data.dir.reshape(-1)
        else:
            gstep = data.initialStepLength / 1e6
            if gstep > optim["DiffMaxChange"]:
                gstep = optim["DiffMaxChange"]
            if gstep < optim["DiffMinChange"]:
                gstep = optim["DiffMinChange"]
            data, f_alpha = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
            data, f_alpha2 = gradient_function(data.xInitial.reshape(-1) + (alpha + gstep) * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
            fPrime_alpha = (f_alpha2 - f_alpha) / gstep

        # Store values linesearch
        data.storefx.append(f_alpha)
        data.storex.append(alpha)

        # Store current bracket position of A
        aPrev = data.a
        f_aPrev = data.f_a
        fPrime_aPrev = data.fPrime_a

        # Update the current brackets
        if (f_alpha > data.fInitial + alpha * optim["rho"] * data.fPrimeInitial) or (f_alpha >= data.f_a):
            # Update bracket B to current alpha
            data.b = alpha
            data.f_b = f_alpha
            data.fPrime_b = fPrime_alpha
        else:
            # Wolfe conditions, if true then acceptable point found
            if abs(fPrime_alpha) <= -optim["sigma"] * data.fPrimeInitial:
                if optim["GradConstr"]:
                    # Gradient was not yet calculated because of time costs
                    (data, f_alpha, grad) = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
                    fPrime_alpha = grad.T @ data.dir.reshape(-1)

                # Store the found alpha values
                data.alpha = alpha
                data.fPrime_alpha = fPrime_alpha
                data.f_alpha = f_alpha
                data.grad = grad
                data.section_exitflag = []
                return data

                # Update bracket A
            data.a = alpha
            data.f_a = f_alpha
            data.fPrime_a = fPrime_alpha

            if (data.b - data.a) * fPrime_alpha >= 0:
                # B becomes old bracket A;
                data.b = aPrev
                data.f_b = f_aPrev
                data.fPrime_b = fPrime_aPrev

        # No acceptable point could be found
        if abs(data.b - data.a) < eps():
            data.section_exitflag = -2
            return data

        # maxFunEvals reached
        if data.funcCount > optim["MaxFunEvals"]:
            data.section_exitflag = -1
            return data


def bracketingPhase(funfcn, data, optim):
    # bracketingPhase finds a bracket [a,b] that contains acceptable points; a bracket 
    # is the same as a closed interval, except that a > b is allowed.
    #
    # The outputs f_a and fPrime_a are the values of the function and the derivative 
    # evaluated at the bracket endpoint 'a'. Similar notation applies to the endpoint 
    # 'b'. 

    # Parameters of bracket A
    data.a = []
    data.f_a = []
    data.fPrime_a = []

    # Parameters of bracket B
    data.b = []
    data.f_b = []
    data.fPrime_b = []

    # First trial alpha is user-supplied
    # f_alpha will contain f(alpha) for all trial points alpha
    # fPrime_alpha will contain f'(alpha) for all trial points alpha
    alpha = data.initialStepLength
    f_alpha = data.fInitial
    fPrime_alpha = data.fPrimeInitial

    # Set maximum value of alpha (determined by fminimum)
    alphaMax = (data.fminimum - data.fInitial) / (optim["rho"] * data.fPrimeInitial)
    alphaPrev = 0

    while True:
        # Evaluate f(alpha) and f'(alpha)
        fPrev = f_alpha
        fPrimePrev = fPrime_alpha

        # Calculate value (and gradient if no extra time cost) of current alpha
        if not optim["GradConstr"]:
            data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
            fPrime_alpha = grad.T @ data.dir.reshape(-1)
        else:
            gstep = data.initialStepLength / 1e6
            if (gstep > optim["DiffMaxChange"]):
                gstep = optim["DiffMaxChange"]
            if (gstep < optim["DiffMinChange"]):
                gstep = optim["DiffMinChange"]
            data, f_alpha = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
            data, f_alpha2 = gradient_function(data.xInitial.reshape(-1) + (alpha + gstep) * data.dir.reshape(-1), funfcn, data, optim, calc_grad=False)
            fPrime_alpha = (f_alpha2 - f_alpha) / gstep

        # Store values linesearch
        data.storefx.append(f_alpha)
        data.storex.append(alpha)

        # Terminate if f < fminimum
        if f_alpha <= data.fminimum:
            data.bracket_exitflag = 4
            return data

            # Bracket located - case 1 (Wolfe conditions)
        if np.all(f_alpha > (data.fInitial + alpha * optim["rho"] * data.fPrimeInitial)) or (f_alpha >= fPrev):
            # Set the bracket values
            data.a = alphaPrev
            data.f_a = fPrev
            data.fPrime_a = fPrimePrev
            data.b = alpha
            data.f_b = f_alpha
            data.fPrime_b = fPrime_alpha
            # Finished bracketing phase
            data.bracket_exitflag = 2
            return data

            # Acceptable steplength found
        if abs(fPrime_alpha) <= -optim["sigma"] * data.fPrimeInitial:  # TODO hier muss wahrscheinlich np.all ?
            if optim["GradConstr"]:
                # Gradient was not yet calculated because of time costs
                data, f_alpha, grad = gradient_function(data.xInitial.reshape(-1) + alpha * data.dir.reshape(-1), funfcn, data, optim, calc_grad=True)
                fPrime_alpha = grad.T @ data.dir.reshape(-1)
            # Store the found alpha values
            data.alpha = alpha
            data.fPrime_alpha = fPrime_alpha
            data.f_alpha = f_alpha
            data.grad = grad
            # Finished bracketing phase, and no need to call sectioning phase
            data.bracket_exitflag = []
            return data

            # Bracket located - case 2
        if fPrime_alpha >= 0:
            # Set the bracket values
            data.a = alpha
            data.f_a = f_alpha
            data.fPrime_a = fPrime_alpha
            data.b = alphaPrev
            data.f_b = fPrev
            data.fPrime_b = fPrimePrev
            # Finished bracketing phase
            data.bracket_exitflag = 2
            return data

        # Update alpha
        if 2 * alpha - alphaPrev < alphaMax:
            brcktEndpntA = 2 * alpha - alphaPrev
            brcktEndpntB = min(alphaMax, alpha + optim["tau1"] * (alpha - alphaPrev))
            # Find global minimizer in bracket [brcktEndpntA,brcktEndpntB] of 3rd-degree polynomial
            # that interpolates f() and f'() at alphaPrev and at alpha
            alphaNew, _ = pickAlphaWithinInterval(brcktEndpntA, brcktEndpntB, alphaPrev, alpha, fPrev, fPrimePrev, f_alpha, fPrime_alpha, optim)
            alphaPrev = alpha
            alpha = alphaNew
        else:
            alpha = alphaMax

        # maxFunEvals reached
        if data.funcCount > optim["MaxFunEvals"]:
            data.bracket_exitflag = -1
            return data


def pickAlphaWithinInterval(brcktEndpntA, brcktEndpntB, alpha1, alpha2, f1, fPrime1, f2, fPrime2, optim):
    # finds a global minimizer alpha within the bracket [brcktEndpntA,brcktEndpntB] of the cubic polynomial
    # that interpolates f() and f'() at alpha1 and alpha2. Here f(alpha1) = f1, f'(alpha1) = fPrime1,
    # f(alpha2) = f2, f'(alpha2) = fPrime2.

    # determines the coefficients of the cubic polynomial with c(alpha1) = f1,
    # c'(alpha1) = fPrime1, c(alpha2) = f2, c'(alpha2) = fPrime2.
    coeff = np.array([(fPrime1 + fPrime2) * (alpha2 - alpha1) - 2 * (f2 - f1),
                      3 * (f2 - f1) - (2 * fPrime1 + fPrime2) * (alpha2 - alpha1),
                      np.array((alpha2 - alpha1) * fPrime1),
                      f1], dtype=np.float64).squeeze()

    # Convert bounds to the z-space
    lowerBound = (brcktEndpntA - alpha1) / (alpha2 - alpha1)
    upperBound = (brcktEndpntB - alpha1) / (alpha2 - alpha1)

    # Swap if lowerbound is higher than the upperbound
    if (lowerBound > upperBound):
        t = upperBound
        upperBound = lowerBound
        lowerBound = t

    # Find minima and maxima from the roots of the derivative of the polynomial.
    sPoints = np.roots([3 * coeff[0], 2 * coeff[1], coeff[2]])

    # Remove imaginaire and points outside range
    sPoints = sPoints[np.imag(sPoints) != 0]
    sPoints = sPoints[sPoints < lowerBound]
    sPoints = sPoints[sPoints > upperBound]

    # Make vector with all possible solutions
    # sPoints = [lowerBound, sPoints.reshape(-1).T, upperBound]  # TODO ... jetzt macht das ganze .' und (:) keinen Sinn mehr
    sPoints = np.append(np.append(lowerBound, sPoints.reshape(-1)), upperBound)

    # Select the global minimum point
    # f_alpha, index = np.min(np.polyval(coeff, sPoints))
    pv = np.polyval(coeff, sPoints)
    index = np.argmin(pv)
    f_alpha = pv[index]
    z = sPoints[index]

    # Add the offset and scale back from [0..1] to the alpha domain
    alpha = alpha1 + z * (alpha2 - alpha1)

    # Show polynomial search
    # if(optim.Display(1)=='p');
    #     vPoints=polyval(coeff,sPoints);
    #     plot(sPoints*(alpha2 - alpha1)+alpha1,vPoints,'co');
    #     plot([sPoints(1) sPoints(end)]*(alpha2 - alpha1)+alpha1,[vPoints(1) vPoints(end)],'c*');
    #     xPoints=linspace(lowerBound/3, upperBound*1.3, 50);
    #     vPoints=polyval(coeff,xPoints);
    #     plot(xPoints*(alpha2 - alpha1)+alpha1,vPoints,'c');
    # end

    # TODO NP real not in matlab
    return np.real(alpha), f_alpha


def updateQuasiNewtonMatrix_LBFGS(data, optim):
    # updates the quasi-Newton matrix that approximates the inverse to the Hessian.
    # Two methods are support BFGS and L-BFGS, in L-BFGS the hessian is not
    # constructed or stored.
    # Calculate position, and gradient diference between the iterations
    deltaX = data.alpha * data.dir
    deltaG = data.gradient - data.gOld

    if (deltaX.T @ deltaG) >= np.sqrt(eps()) * max(eps(), np.linalg.norm(deltaX) * np.linalg.norm(deltaG)):

        if optim["HessUpdate"][0] == 'b':
            # Default BFGS as described by Nocedal
            p_k = 1 / (deltaG.T @ deltaX)
            Vk = np.eye(data.numberOfVariables) - p_k * deltaG * deltaX.T
            # Set Hessian
            data.Hessian = Vk.T @ data.Hessian @ Vk + p_k * deltaX * deltaX[np.newaxis].T
            # Set new Direction
            data.dir = -data.Hessian @ data.gradient
        else:
            # L-BFGS with scaling as described by Nocedal

            # Update a list with the history of deltaX and deltaG
            data.deltaX[:, 1:optim["StoreN"]] = data.deltaX[:, 0:optim["StoreN"] - 1]
            data.deltaX[:, 0] = deltaX
            data.deltaG[:, 1:optim["StoreN"]] = data.deltaG[:, 0:optim["StoreN"] - 1]
            data.deltaG[:, 0] = deltaG

            data.nStored = data.nStored + 1
            if data.nStored > optim["StoreN"]:
                data.nStored = optim["StoreN"]

            # Initialize variables
            # a = np.zeros((1, data.nStored))
            # p = np.zeros((1, data.nStored))
            a = np.zeros(data.nStored)
            p = np.zeros(data.nStored)

            q = data.gradient
            for i in range(data.nStored):
                p[i] = 1 / (data.deltaG[:, i].T @ data.deltaX[:, i])
                a[i] = p[i] * data.deltaX[:, i].T @ q
                q -= a[i] * data.deltaG[:, i]

            # Scaling of initial Hessian (identity matrix)
            p_k = data.deltaG[:, 0].T @ data.deltaX[:, 0] / sum(data.deltaG[:, 0] ** 2)

            # Make r = - Hessian * gradient
            r = p_k * q
            for i in reversed(range(data.nStored)):
                b = p[i] * data.deltaG[:, i].T @ r
                r = r + data.deltaX[:, i] * (a[i] - b)

            # Set new direction
            data.dir = -r
    return data


from scipy.optimize import rosen, rosen_der


def demo():
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    def proxy(x):
        return rosen(x), rosen_der(x)

    from scipy.optimize import minimize

    result = minimize(proxy, x0, jac=True, options={"disp": False})
    print(f"result scipy x: {result.x}")
    print(f"result scipy fval: {result.fun}")

    x, fval, exitfalg, output, grad, iteration = matlab_fmin_lbfgs(proxy, x0, {"Display": "iter"})
    print(f"result  fmin x={x}")
    print(f"result  fmin fval={fval}")

# demo()
