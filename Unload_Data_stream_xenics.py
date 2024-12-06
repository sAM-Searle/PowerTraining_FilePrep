

#Open a fusion box bin file, extract the dat, use the xy position to split it into layer files
import numpy as np
import matplotlib.pyplot as plt
import Coordinate_Transform as CT
import Bounding_Boxes as BB
import os
import glob
import tqdm
import h5py
import asyncio
from concurrent.futures import ThreadPoolExecutor

def list_files_in_directory(directory):
    # Get a list of all files in the directory
    files = glob.glob(f"{directory}/*")  # Lists all items in the directory
    # Filter to keep only files
    files = [f for f in files if os.path.isfile(f)]
    # Sort files alphabetically
    files = sorted(files)
    return files

class MultiDatFile:
    def __init__(self, folder):
        self.folder = folder
        self.current_file = 0 
        self.current_idx = 0
        self.frameformat = np.dtype([
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
        
        # List all files in the directory (bins)
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.dat')]
        self.files.sort()  # Sort files in lexicographical order (or based on naming convention)
        
        # Initialize variables for batch loading
        self.file_idx = 0
        self.current_idx = 0
        self.mmap = np.memmap(self.files[0], dtype=self.frameformat, mode='r')  # Will be initialized when the first file is accessed

        # Get total number of frames across all files
        self.file_frames = [self._get_frame_count(file) for file in self.files]
        self.total_frames = sum(self.file_frames)
        print(self.total_frames)
    def load_next_frame(self):
        if self.current_idx>=self.file_frames[self.current_file]:
            self.current_file +=1
            self.current_idx = 0
            self.mmap = np.memmap(self.files[self.current_file], dtype=self.frameformat, mode='r')
            
        frame = self.mmap[self.current_idx]
        self.current_idx +=1
        return frame
            
    def _get_frame_count(self, file):
        """Return the number of frames in a given binary file."""
        mmap = np.memmap(file, dtype=self.frameformat, mode='r')
        return len(mmap)
    

        
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
    
    x_min, y_min, x_max, y_max = bounding_boxes.T  # Unpack bounding box coordinates

    # Check conditions for all bounding boxes
    in_x_range = (x >= x_min) & (x <= x_max)  # Shape (n, m)
    in_y_range = (y >= y_min) & (y <= y_max)  # Shape (n, m)

    # Find all positions that satisfy both conditions
    inside_boxes = in_x_range & in_y_range  # Shape (n, m)
    true_indices = np.flatnonzero(inside_boxes)


    return true_indices[0] if true_indices.size > 0 else -1

def get_part_pos(frame):
    x = convert_to_signed(frame["setpoint_x"])    
    y = convert_to_signed(frame["setpoint_y"])    
    x,y = CT.transform_coordinates(CT.transformation_matrix,x,y)
    result = find_bounding_boxes_numpy(x,y,bounding_boxes=BB.bounding_boxes)
    return result,x,y
    
    
def part_change(layer,part):
    part_group = layer.create_group(str(part))
    Frametypr = part_group.create_group("Infra")
    return Frametypr
def layer_change(f,layer):
    layer_group = f.create_group(str(layer))
    return layer_group
def convert_power(powers):
    return (np.array(powers)-805.34)/29964.04
def save_data(part_group,packet_counter ,xx ,yy ,xx_mm,yy_mm,frame_counter ,setpoint_power ,data_block_index ,sync_counter ,roi_x ,roi_y,I):
    part_group.create_dataset(name ='packet_counter',data = packet_counter)
    part_group.create_dataset(name ='setpoint',data = np.vstack((xx,yy)))
    part_group.create_dataset(name ='setpoint_mm',data = np.vstack((xx_mm,yy_mm)))
    part_group.create_dataset(name ='frame_counter',data = frame_counter )
    part_group.create_dataset(name ='setpoint_power',data = convert_power(setpoint_power))
    part_group.create_dataset(name ='data_block_index',data = data_block_index )
    part_group.create_dataset(name ='sync_counter',data = sync_counter )
    part_group.create_dataset(name ='roi',data = np.vstack((roi_x,roi_y)) )
    part_group.create_dataset(name ='I',data =I , compression="gzip")
 
def check_unique(arr):
    # Convert the array to a set to check if all elements are unique
    if len(arr) == len(set(arr)):
        return arr  # Values are unique, return the array
    else:
        raise ValueError("Array contains duplicate values")  # Raise an error for duplicates
    
def process_frame(executor):
    bin_obj = MultiDatFile(r"e:\MultiPlicityextrqStorage\Infrq")
    frame_count =0
    layer = 0
    previous_part = 0
    
    packet_counter = []
    setpointmm_xx = []
    setpointmm_yy = []
    setpoint_xx = []
    setpoint_yy = []
    frame_counter = []
    setpoint_power = []
    data_block_index = []
    sync_counter = []
    roi_x = []
    roi_y = []
    I = []
    with h5py.File(r'e:\MultiPlicityextrqStorage\PowerTraining.h5', 'w') as f:

        layer_group = layer_change(f, 0)


        for ii in tqdm.tqdm(range(bin_obj.total_frames)):
            frame = bin_obj.load_next_frame()
            part,x,y = get_part_pos(frame)
            if part == -1:
                continue
            if previous_part != part and part != -1:
                part_group = part_change(layer_group,previous_part)
                executor.submit(save_data, part_group, packet_counter, setpoint_xx, setpoint_yy,setpointmm_xx, setpointmm_yy, frame_counter, setpoint_power, data_block_index, sync_counter, roi_x, roi_y, I)
                # Reset arrays after the save
                packet_counter.clear()
                setpointmm_xx.clear()
                setpointmm_yy.clear()
                setpoint_xx.clear()
                setpoint_yy.clear()
                frame_counter.clear()
                setpoint_power.clear()
                data_block_index.clear()
                sync_counter.clear()
                roi_x.clear()
                roi_y.clear()
                I.clear()
            
                if previous_part == 11 and part == 0:
                    #save_data 

                    layer +=1
                    
                    layer_group = layer_change(f,layer)
                    
            packet_counter.append(frame['packet_counter'])
            setpoint_xx.append(frame['setpoint_x'])
            setpoint_yy.append(frame['setpoint_y'])
            setpointmm_xx.append(x)
            setpointmm_yy.append(y)
            frame_counter.append(frame['frame_counter'])
            setpoint_power.append(frame['setpoint_power'])
            data_block_index.append(frame['data_block_index'])
            sync_counter.append(frame['sync_counter'])
            roi_x.append(frame['roi_x'])
            roi_y.append(frame['roi_y'])
            I.append(frame['I'])
            previous_part = part
            
def main():
    with ThreadPoolExecutor() as executor:
         process_frame(executor)

# Start the asyncio event loop
if __name__ == '__main__':
    main()