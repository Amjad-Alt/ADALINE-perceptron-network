import matplotlib.pyplot as plt
import numpy as np
# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2 ],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1 , 1],
              [-1, -1, 1 ,1, -1, -1, 1, 1]])

target = [1 ,1 ,2 ,2 ,3 ,3 ,4 ,4]
# -------------------------------------------------
# plot 

px = p[:1,:]
py = p[1:,:]
tx = t[:1,:]
ty = t[1:,:]

plt.scatter(px,py)
plt.scatter(tx,ty)
#%%
# -------------------------------------------------
# Use an ADALINE network and the LMS learning rule to classify the patterns.

# Transpose p, t 
p_t = p.T
t_t = t.T

# Set the learning rate, sum square error list, and weight matrix
lr = 0.001
sse_p1 = []
sse_p2 = []

w = np.array([[1.0, 0.0],[0.0, 1.0]])
b = np.array([[1.0],[1.0]])

# Loop over 100 iterations
for epoch in range(100):
    p1_error = 0
    p2_error = 0
    # Loop over the input patterns
    for i in range(p_t.shape[0]):
        # Get the input pattern and its corresponding target
        x = p_t[i,:].reshape(2,1)
        t = t_t[i,:].reshape(2,1)
        # Calculate the output of the network
        a = np.dot(w, x) + b # this should be 2x1 
        # Calculate the error
        e = t - a 
        # Update the weight 
        w += 2 * lr * np.dot(e,x.reshape(1,2)) 
        b += 2 * lr * e
        p1_error = p1_error + e[0]**2
        p2_error = p2_error + e[1]**2
    # Calculate the sum square error and add it to the mse list to plot
    sse_p1.append(np.sum(np.square(p1_error)))
    sse_p2.append(np.sum(np.square(p2_error)))
##%%
# -------------------------------------------------
# plot Sum Square Error.
plt.figure()
plt.plot(sse_p1,label="P1 Error")
plt.plot(sse_p2,label="P2 Error")
plt.xscale("log")
plt.yscale("log")
plt.title("SSE")
plt.grid()
plt.tight_layout()
plt.xlabel("Epoch")
plt.ylabel("SSE Error")
plt.legend()
plt.show()

#%%
# -------------------------------------------------
# Plot the Decision Boundaries and the patterns.

# scatterplot of the inputs
plt.figure()
plt.scatter(px, py, c='blue', label='Input')
plt.grid()
# Plot the weight lines
plt.axline((0, 0.2600874861383471), (0.02232481107900362,0), c='red', label='DB1')
plt.axline((0, 0.2522632466931575), (-0.9998363222965161,0), c='green', label='DB2')
plt.tight_layout()
plt.xlabel("P1")
plt.ylabel("P2")
plt.title("Decision boundary for LMS")
plt.legend()
plt.show()


# -------------------------------------------------
#%%
# Compare the results of part iv. with decision boundary that you will get with perceprtron 
# learning rule. Explain the differences?

# Create a function for the Hardlims 1 0r -1
def hardlims(x):
    if x >= 0:
        return 1
    else:
        return -1

w = np.array([[1.0, 0.0],[0.0, 1.0]])
b = np.array([[1.0],[1.0]])

# Loop over 100 iterations
for epoch in range(1000):
    # Loop over the input patterns
    for i in range(p_t.shape[0]):
        dummy = []
        # Get the input pattern and its corresponding target
        x = p_t[i,:].reshape(2,1)
        t = t_t[i,:].reshape(2,1)
        # Calculate the output of the network
        a = np.dot(w, x) + b 
        # Pass through the hardlims
        dummy.append(hardlims(a[0]))
        dummy.append(hardlims(a[1]))
        a = np.array(dummy).reshape(2,1)
        # Calculate the error
        e = t - a 
        # Update the weight 
        w += np.dot(e,x.reshape(1,2)) 
        b += e

# Plotting the decision boundary
plt.figure()
plt.scatter(px, py, c='blue', label='Input')
plt.grid()
# Plot the weight lines
plt.axline((-0.2,0), (-0.2,2), c='red', label='DB1')
plt.axline((0,-0.14285714285714285), (0.5,0), c='green', label='DB2')
plt.tight_layout()
plt.xlabel("P1")
plt.ylabel("P2")
plt.title("Decision boundary for Perceptron")
plt.legend()
plt.show()
##%%
print("The key difference between ADLINE and perceptron lies in the optimization method used to train the network. While perceptron uses a simple update rule, ADLINE uses the more efficient LMS algorithm. By minimizing the sum of squared errors, ADLINE can find the weights that best fit the training data, leading to faster convergence and more accurate classifications. Therefore, the use of LMS in ADLINE is a significant improvement over the simple perceptron update rule.")
