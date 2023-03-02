import numpy as np
import matplotlib.pyplot as plt

# Load data
rel_err_vec_test0 = np.load(f"./results/test0/rel_err.npy")
ssim_vec_test0 = np.load(f"./results/test0/ssim.npy")

rel_err_vec_test1 = np.load(f"./results/test1/rel_err.npy")
ssim_vec_test1 = np.load(f"./results/test1/ssim.npy")

rel_err_vec_test2 = np.load(f"./results/test2/rel_err.npy")
ssim_vec_test2 = np.load(f"./results/test2/ssim.npy")

# Visualization
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(rel_err_vec_test0)+1), rel_err_vec_test0)
plt.plot(np.arange(1, len(rel_err_vec_test1)+1), rel_err_vec_test1)
plt.plot(np.arange(1, len(rel_err_vec_test2)+1), rel_err_vec_test2)
plt.title('Relative Error')
plt.xlabel('h')
plt.legend(['Test0', 'Test1', 'Test2'])
plt.grid()


plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(ssim_vec_test0)+1), ssim_vec_test0)
plt.plot(np.arange(1, len(ssim_vec_test1)+1), ssim_vec_test1)
plt.plot(np.arange(1, len(ssim_vec_test2)+1), ssim_vec_test2)
plt.title('SSIM')
plt.xlabel('h')
plt.legend(['Test0', 'Test1', 'Test2'])
plt.grid()
plt.show()