import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(num_samples):
    #generate random (x, y) coordinates within a square bounding box [-1, 1]
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    #check the Euclidean distance of each point from the coordinate origin
    distance = x**2 + y**2
    
    #track points that successfully fall inside the bounds of the unit circle
    inside_circle = distance <= 1
    
    #calculate the running convergence ratio to estimate the true value of Pi
    running_total = np.cumsum(inside_circle)
    sample_sizes = np.arange(1, num_samples + 1)
    pi_estimates = (running_total / sample_sizes) * 4
    
    return sample_sizes, pi_estimates

#execute the simulation using 10,000 algorithmic sample iterations
total_samples = 10000
samples, estimates = estimate_pi(total_samples)

#pipeline the final simulation arrays into a descriptive Matplotlib convergence plot
plt.figure(figsize=(10, 6))
plt.plot(samples, estimates, label='Monte Carlo Operational Estimate', color='blue', linewidth=1.5)
plt.axhline(y=np.pi, color='red', linestyle='--', label='True Value ($\pi$)')
plt.xlabel('Number of Computational Random Samples')
plt.ylabel('Estimated Numeric Value')
plt.title('Statistical Convergence Matrix: Estimating Pi via Stochastic Randomization')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"Final Execution Convergence Result: {estimates[-1]}")
