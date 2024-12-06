#Open a fusion box bin file, extract the dat, use the xy position to split it into layer files
import numpy as np
import matplotlib.pyplot as plt
import Coordinate_Transform as CT
import Unload_Data_stream_visual

bounding_boxes = [
    [x - 6, y - 6, x + 6, y + 6] 
    for x, y in [
    [22.5	,54.776],
    [39.959	,32.935],
    [6.318	,43.45],
    [13.249	,23.656],
    [33.493	,11.205],
    [5.988	,1.819],
    [53.682	,-6.835],
    [11.643	,-16.684],
    [30.392	,-23.596],
    [46.149	,-32.685],
    [10.227	,-43.11],
    [29.088	,-56.961],

    ]
]

class ClickCapture:
    def __init__(self, ax, max_clicks=12):
        self.ax = ax

        self.max_clicks = max_clicks
        self.clicked_indices = []
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        print("Hello")
    
    def on_click(self, event):
        # Check if the click is within the axes limits
        if event.inaxes != self.ax:
            return
        
        # Get the x-coordinate of the clicked point
        x_clicked = event.xdata
        y_clicked = event.ydata

        if x_clicked is not None:
            # Find the index of the closest x value in average_x
            self.clicked_indices.append([x_clicked,x_clicked])
            print("Click!")
            # Plot a vertical line at the clicked x position
            self.ax.plot(x_clicked,y_clicked, color='r',marker='+', linestyle='')
            # Redraw the plot to update with the vertical line
            self.ax.figure.canvas.draw()
            # Print the current clicked x index

            # If 4 clicks are received, return the indices and close the plot
            if len(self.clicked_indices) == self.max_clicks:
                self.clicked_indices = np.array(self.clicked_indices)
                self.clicked_indices = np.round(self.clicked_indices)
                
                print(f"Final clicked indices: {self.clicked_indices}")
                plt.close(self.ax.figure)  # Close the plot after 4 clicks
                # Flatten and reshape into 4 columns

                return self.clicked_indices
            
if __name__ == "__main__":
    bin_obj = Unload_Data_stream_visual.MultiDatFile(r"VIL01_20241122162815.dat")

    setpoint_x = Unload_Data_stream_visual.convert_to_signed(bin_obj.mmap["setpoint_x"])
    setpoint_y = Unload_Data_stream_visual.convert_to_signed(bin_obj.mmap["setpoint_y"])
    
    x,y = CT.transform_coordinates_batch(CT.transformation_matrix,setpoint_x,setpoint_y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.plot(x,y)
    click_capture = ClickCapture(plt.gca())
    plt.show()