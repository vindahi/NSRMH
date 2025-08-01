from sklearn.model_selection import train_test_split
import scipy.io as sio
import numpy as np
import h5py
from tqdm import tqdm


def add_noise_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    for i in tqdm(noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels[i, j]=0
        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels[i, j]=1
    return labels

def generate_noise_F(noise):
    noise_rate = noise
    data = sio.loadmat('/mir_clip_all.mat')
    for i in noise_rate:
        labels_matrix = np.array(list(data['L_tr']))
        labels_matrix2 = np.array(list(data['L_tr']))
        # labels_matrix = data['L_tr'][:].T
        # labels_matrix2 = data['L_tr'][:].T
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)
        output_file = h5py.File('mir-lalll-noise_{}.h5'.format(i), 'w')
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)
        output_file.close()


def generate_noise_N(noise):
    noise_rate = noise
    data = sio.loadmat('/nus_clip_all.mat')
    for i in noise_rate:
        labels_matrix = np.array(list(data['L_tr']))
        labels_matrix2 = np.array(list(data['L_tr']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)
        output_file = h5py.File('nus-lall-noise_{}.h5'.format(i), 'w')
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)
        output_file.close()


def generate_noise_M(noise):
    noise_rate = noise
    data = sio.loadmat('/coco_clip_all.mat')
    for i in noise_rate:
        labels_matrix = np.array(list(data['L_tr']))
        labels_matrix2 = np.array(list(data['L_tr']))

        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)
        output_file = h5py.File('coco-lall-noise_{}.h5'.format(i), 'w')
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)
        output_file.close()




    
    
if __name__ == "__main__":  
    noise_rate = [0.0]
    generate_noise_F(noise_rate)
    generate_noise_N(noise_rate)
    generate_noise_M(noise_rate)
