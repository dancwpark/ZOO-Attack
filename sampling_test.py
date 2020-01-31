import numpy as np
import scipy
from scipy import special
from scipy.special import softmax

std_out = 0.01
std_log = 1.
samples = 1000

x = np.array([1,2,3,4,5,6,7,8,9])
print("logits", x)
p = softmax(x)
print("probs", p)


# Test 1
## Can we sample away noise from probabilities?
total = np.zeros(x.shape)
for i in range(samples):
    total += (p + np.random.normal(0, std_out, 9))
avg_prob = total/samples
print("average prob", avg_prob)
diff = avg_prob - p
print(diff)
print(np.linalg.norm(diff))

# Test 2
## Can we sample away noise from logits?
total = np.zeros(x.shape)
for i in range(samples):
    temp_prob = softmax(x + np.random.normal(0, std_log, 9))
    total += np.log(temp_prob)
avg_prob = total/samples
avg_prob = softmax(avg_prob)
print()
print(avg_prob - p)
print()
print(avg_prob)
print(p)
print(np.linalg.norm((avg_prob-p)))

# Test 3
## Can we average away both
