import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
from IPython.display import clear_output
from timeit import default_timer as timer
import time

# minibatch gradient descent
def gradient_descent(g, g_val, alpha, max_its, w, num_train, num_val, batch_size,**kwargs):    
    # switch for verbose
    verbose = True
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,np.arange(num_train))]
    val_hist = [g_val(w_hist[0],np.arange(num_val))]
   
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))

    # over the line
    for k in range(max_its):                   
        # loop over each minibatch
        start = timer()
        train_cost = 0
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))
            
            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # add to total training cost evaluation
            train_cost += cost_eval*float(len(batch_inds))/float(num_train)
            
            # take descent step with momentum
            w = w - alpha*grad_eval

        end = timer()

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
        val_cost = g_val(w_hist[-1],np.arange(num_val))
        val_hist.append(val_cost)
        
        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1][0],4)) + ', val cost = ' + str(np.round(val_hist[-1],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
        time.sleep(1.5)
        clear_output()
    return w_hist,train_hist,val_hist