import numpy as np
import cv2



def transform_coordinates_batch(matrix, x_array, y_array):
    """
    Transforms arrays of x, y coordinates using a 3x3 transformation matrix.

    Parameters:
        matrix (numpy.ndarray): A 3x3 transformation matrix.
        x_array (numpy.ndarray or list): Array of x-coordinates.
        y_array (numpy.ndarray or list): Array of y-coordinates.

    Returns:
        tuple: Transformed arrays of (x', y') coordinates.
    """
    # Ensure inputs are numpy arrays
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)

    # Check that x_array and y_array have the same size
    if x_array.shape != y_array.shape:
        raise ValueError("x_array and y_array must have the same shape.")
    
    # Convert x and y arrays to homogeneous coordinates (add a row of 1s for the z-coordinate)
    ones = np.ones_like(x_array)
    input_coords = np.stack((x_array, y_array, ones), axis=0)  # Shape: (3, N)

    # Apply the transformation matrix to the coordinates
    transformed_coords = np.dot(matrix, input_coords)  # Shape: (3, N)


    # Return the x' and y' transformed arrays
    return transformed_coords[0], transformed_coords[1]
def transform_coordinates(matrix, x, y):
    """
    Transforms a single (x, y) coordinate using a 3x3 transformation matrix.

    Parameters:
        matrix (numpy.ndarray): A 3x3 transformation matrix.
        x (float or int): The x-coordinate.
        y (float or int): The y-coordinate.

    Returns:
        tuple: Transformed (x', y') coordinates.
    """
    # Create a homogeneous coordinate for the input (x, y, 1)
    input_coord = np.array([x, y, 1])  # Shape: (3,)

    # Apply the transformation matrix
    transformed_coord = np.dot(matrix, input_coord)  # Shape: (3,)

    # Extract x' and y' (ignore the homogeneous z')
    x_prime = transformed_coord[0]
    y_prime = transformed_coord[1]

    return x_prime, y_prime
def get_affine_transformation_matrix():
    # Define 3 corresponding points for MCP and world coordinates
    mcp_points = np.array([
        [2559240, 1421800],  # Point 1 in MCP coordinates
        [-284360, -1421800],  # Point 2 in MCP coordinates
        [2559240, -1421800],          # Point 3 in MCP coordinates (you can choose this based on your data)
    ], dtype=np.float32)

    world_points = np.array([
        [-25, -25],  # Point 1 in world coordinates
        [25, 25],    # Point 2 in world coordinates
        [-25, 25],       # Point 3 in world coordinates (chosen for simplicity)

    ], dtype=np.float32)

    # Calculate the affine transformation matrix using OpenCV
    transformation_matrix = cv2.getAffineTransform(mcp_points, world_points)
    
    return transformation_matrix

transformation_matrix = get_affine_transformation_matrix()