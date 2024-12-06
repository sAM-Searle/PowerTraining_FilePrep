import numpy as np
import math
import h5py
import os
import tqdm
import Coordinate_Transform as CT
import Bounding_Boxes as BB
import cProfile
frameformat = np.dtype([
    ('packet_version', np.uint32),
    ('packet_counter', np.uint32),
    ('packet_info_spare', np.uint32, (2,)),
    ('frame_counter', np.uint32),
    ('setpoint_x', np.uint32),
    ('setpoint_y', np.uint32),
    ('setpoint_speed', np.uint32),
    ('setpoint_diameter', np.uint32),
    ('setpoint_power', np.uint32),
    ('data_block_index', np.uint32),
    ('sync_counter', np.uint32),
    ('laser_status', np.uint32),
    ('roi_x', np.uint32),
    ('roi_y', np.uint32),
    ('frame_info_spare', np.uint32, (1,)),
    ('frame_offset', np.uint32),
    ('frame_length', np.uint32),
    ('frame_spare', np.uint32, (2,)),
    ('I', np.uint8, (96, 96))
])


part_count = 12
input_path = r"D:\Power_Training\visual_laser0\Storage"
output_destination = r"E:\Power_Training"
imager_type = "Visual"
data_to_save = ['packet_counter','frame_counter','setpoint_x','setpoint_y','setpoint_speed','setpoint_diameter','setpoint_power','data_block_index','sync_counter','laser_status','roi_x','roi_y','I']

def convert_to_signed(data):
    # Convert unsigned number to a signed number in 24-bit
    if isinstance(np.array(data), np.ndarray):
        data = data.astype(np.int32)
        if data.size<=1:
            if data > 2**23:
                data-= 2**24
            return data.astype(np.int32)
        data[data > 2**23] -= 2**24
        return data.astype(np.int32)
def find_bounding_boxes_numpy(x,y, bounding_boxes):
    """
    Find the bounding box each position corresponds to using NumPy.

    Parameters:
        positions (numpy.ndarray): Array of shape (n, 2) for n (x, y) positions.
        bounding_boxes (numpy.ndarray): Array of shape (m, 4) for m bounding boxes,
                                         where each box is defined as [x_min, y_min, x_max, y_max].

    Returns:
        numpy.ndarray: Array of size n with the index of the bounding box for each position,
                       or -1 if no bounding box contains the position.
    """
    bounding_boxes = np.asarray(bounding_boxes)  # Ensure bounding_boxes is a NumPy array
    
    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = bounding_boxes.T
    

    
    # Check conditions for all bounding boxes
    in_x_range = (x[:, None] >= x_min) & (x[:, None] <= x_max)  # Shape (n, m)
    in_y_range = (y[:, None] >= y_min) & (y[:, None] <= y_max)  # Shape (n, m)
    
    # Find all positions that satisfy both conditions
    inside_boxes = in_x_range & in_y_range  # Shape (n, m)
    
    # Find the first bounding box index that contains each position, or -1 if none
    box_indices = np.argmax(inside_boxes, axis=1)  # Get the first True in each row
    box_indices[~inside_boxes.any(axis=1)] = -1  # Set to -1 if no bounding box contains the position
    
    return box_indices
def get_part_pos(frame):
    x = convert_to_signed(frame["setpoint_x"])    
    y = convert_to_signed(frame["setpoint_y"])    
    x,y = CT.transform_coordinates_batch(CT.transformation_matrix,x,y)
    result = find_bounding_boxes_numpy(x,y,BB.bounding_boxes)
    return result
def open_memap(file_path):
    return np.memmap(file_path, dtype=frameformat, mode='r')
def find_part_layer(datablock_index):
    unique_datablock = np.unique(datablock_index)
    layers = []
    for index in unique_datablock:

        part = (index+1)%part_count
        layer = (index+1)//part_count
        
        layers.append([layer,part,index])
    return layers        
def open_or_create_hdf5(file_name):
    # Check if the file exists
    if os.path.exists(file_name):
        # Open the existing HDF5 file
        return h5py.File(file_name, 'r+')
    else:
        # Create a new HDF5 file
        return h5py.File(file_name, 'w') 
def open_or_create_group(hdf,group_name):
    # Open or create the group
    return hdf.require_group(group_name)
def append_or_create_dataset(group,dataset_name,data):
    if dataset_name in group:
        # Retrieve the existing data
        existing_data = group[dataset_name][:]
        
        # Combine existing data with new data
        combined_data = np.concatenate((existing_data, data))
        
        # Delete the old dataset
        del group[dataset_name]
        
        # Recreate the dataset with combined data
        group.create_dataset(dataset_name, data=combined_data,
        compression="gzip",   # Compression type (e.g., "gzip", "lzf", or "szip")
    )
    else:
        # Create a new dataset if it doesn't exist
        group.create_dataset(dataset_name, data=data,
        compression="gzip",   # Compression type (e.g., "gzip", "lzf", or "szip")
    )

def find_layer_transitions(part_numbers):
    transitions = (part_numbers[:-1] == part_count-1) & (part_numbers[1:] == 0)  # Identify transitions
    indices = np.where(transitions)[0]  # Get indices of transitions
    return indices.tolist() if indices.size > 0 else None
def add_data_to_part(part_group,start_index,end_index,mask,memap):          
    for data_name in data_to_save:
        to_save = memap[data_name][start_index:end_index][mask]
        append_or_create_dataset(part_group,data_name,to_save)
        

def add_data_to_layer(layer,memap,part_index,start_index,end_index):
    hdf = open_or_create_hdf5(os.path.join(output_destination,str(layer)+".h5"))
    unique_parts = np.unique(part_index[start_index:end_index])
    for part in unique_parts:
        if part ==-1:
            continue
        hdf_group = open_or_create_group(hdf,str(part))
        hdf_group = open_or_create_group(hdf_group,imager_type)
        mask = part_index[start_index:end_index]==part
        add_data_to_part(hdf_group,start_index,end_index,mask,memap)
    hdf.close()

def process_file(file,layer):
    memap = open_memap(file)
    part_numbers = get_part_pos(memap)
    layer_transitions =find_layer_transitions(part_numbers)
    if layer_transitions == None:
        # add all data to the current layer
        add_data_to_layer(layer,memap,part_numbers,0,-1)
    else:
        layer_transitions.append(-1)
        add_data_to_layer(layer,memap,part_numbers,0,layer_transitions[0])
        for transition_index in range(len(layer_transitions)-1):
            layer += 1
            add_data_to_layer(layer,memap,part_numbers,layer_transitions[transition_index],layer_transitions[transition_index+1])
        
        
    return layer

def sort_files(folder_path):
    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]

    # Sort files based on the timestamp (the part of the filename after the underscore)
    sorted_files = sorted(files, key=lambda x: x.split('_')[1])

    # Create an array of file paths in chronological order
    return [os.path.join(folder_path, file) for file in sorted_files]

if __name__ == '__main__':
    files =sort_files(input_path)
    layer = 0
    for file in tqdm.tqdm(files):
        layer = process_file(file,layer)
        