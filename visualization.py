import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import cv2
import pickle

## Visualization for Camera Transformation
def camera_transformation(checkerboard):
    # Load Camera parameter
    with open('camera_parameter.pkl', 'rb') as f:
        camera_parameter = pickle.load(f)

    # Create projection matrix from camera parameter
    rotation_matries = []
    translation_matries = []

    for rotation_vector, translation_vector in zip(camera_parameter['rotation'], camera_parameter['translation']):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matries.append(rotation_matrix)
        translation_matries.append(translation_vector.T)
    
    translation_matries_np = np.array(translation_matries)
    # Find the maximum value for each column
    max_values = np.max(translation_matries_np, axis=0)

    # Find the minimum value for each column
    min_values = np.min(translation_matries_np, axis=0)

    # define axis and figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111,projection='3d')
    # set limits
    ax.set(xlim=(int(min_values[0][0])-10, int(max_values[0][0])+10), ylim=(int(min_values[0][1])-10, int(max_values[0][1])+10), zlim=(int(min_values[0][2])-10, int(max_values[0][2])+10))

    # plot the plane
    objp = np.zeros((checkerboard[0]*checkerboard[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[1],0:checkerboard[0]].T.reshape(-1,2)
    objp = objp.reshape(checkerboard[1], checkerboard[0], -1)
    x = objp[:, :, 0]
    y = objp[:, :, 1]
    xx, yy = np.meshgrid(range(int(x.min()), int(x.max())), range(int(y.min()), int(y.max())))
    z = 0
    Z = z + 0 * xx + 0 * yy

    # plot the cameara
    for R, T in zip(rotation_matries, translation_matries):
        ax = pr.plot_basis(ax, R, T)

    # plot the plane
    ax.plot_surface(xx, yy, Z, alpha=0.75)

    ax.set_title("Camera Transformation")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Save the figure
    plt.savefig('Camera Transformation.png')

##Visualization for Plane Transformation
def plane_transformation(checkerboard):
    # Load Camera parameter
    with open('camera_parameter.pkl', 'rb') as f:
        camera_parameter = pickle.load(f)

    # Create projection matrix from camera parameter
    rotation_matries = []
    translation_matries = []

    for rotation_vector, translation_vector in zip(camera_parameter['rotation'], camera_parameter['translation']):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matries.append(rotation_matrix)
        translation_matries.append(translation_vector.T)
    
    translation_matries_np = np.array(translation_matries)
    # Find the maximum value for each column
    max_values = np.max(translation_matries_np, axis=0)

    # Find the minimum value for each column
    min_values = np.min(translation_matries_np, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111,projection='3d')
    # set limits
    ax.set(xlim=(int(min_values[0][0])-10, int(max_values[0][0])+10), ylim=(int(min_values[0][1])-10, int(max_values[0][1])+10), zlim=(int(min_values[0][2])-10, int(max_values[0][2])+10))
    
    # plot the camera (fixed at origin)
    ax = pr.plot_basis(ax)

    for R, T in zip(rotation_matries, translation_matries):
        # construct the transformation matrix
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3] = T.squeeze()

        # apply the transformation to the plane points
        objp = np.zeros((checkerboard[0]*checkerboard[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:checkerboard[1],0:checkerboard[0]].T.reshape(-1,2)

        objp_homo = np.concatenate([objp, np.ones((objp.shape[0], 1))], axis=-1)  # make objp homogeneous
        objp_transformed = (RT @ objp_homo.T).T

        # reshape the transformed object points
        objp_transformed = objp_transformed.reshape(checkerboard[1], checkerboard[0], -1)

        # separate x, y, and z coordinates
        x = objp_transformed[:, :, 0]
        y = objp_transformed[:, :, 1]
        z = objp_transformed[:, :, 2]
        
        xx, yy = np.meshgrid(range(int(x.min()), int(x.max())), range(int(y.min()), int(y.max())))
        Z = z.mean() + 0 * xx + 0 * yy

        # plot the plane
        ax.plot_surface(xx, yy, Z, alpha=0.75)

        ax.set_title("Plane Transformation")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        # Save the figure
        plt.savefig('Plane Transformation.png')