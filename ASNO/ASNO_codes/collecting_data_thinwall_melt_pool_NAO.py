import subprocess
import os


for laser_power_profile_number in range(1,99):

    print("Running Laser Power Number", laser_power_profile_number)

    def download_with_progress(source_path, target_directory):
        # Define the path for the local download
        local_compressed_file_path = os.path.join(target_directory, os.path.basename(source_path))
        
        # Download the compressed file
        subprocess.run(['rclone', 'copy', source_path, target_directory], check=True)
        print(f"Downloaded {source_path} to {local_compressed_file_path}")

    # Example usage
    download_with_progress(f'DED_DT:/custom_thinwall/compressed_laser_profile_{laser_power_profile_number}.zarr.zarr.tar.gz', '/home/vnk3019/DT_DED/ME441_Projects')

    import tarfile

    def decompress_tar_gz(file_path, target_directory):
        # Check if the target directory exists, create if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        
        # Open the .tar.gz file
        with tarfile.open(file_path, 'r:gz') as tar:
            # Extract all the contents into the target directory
            tar.extractall(path=target_directory)
            print(f"Decompressed {file_path} into {target_directory}")

    # Example usage
    file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr.tar.gz"
    target_directory = "/home/vnk3019/DT_DED/ME441_Projects" # Change this to your actual target directory
    decompress_tar_gz(file_path, target_directory)

    import zarr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Define min and max time steps
    min_time_step = 570
    max_time_step = 1300

    # Correcting the path, if necessary, to directly access the Zarr array
    laser_power_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_laser_power"

    # Open the Zarr array directly (make sure this is the correct path to the array)
    laser_power_array = zarr.open(laser_power_path, mode='r')

    # Depending on the structure of your Zarr array, you might directly convert it to a DataFrame
    # For a simple 2D array, it can be straightforward
    if laser_power_array.ndim == 2:
        # Convert the Zarr array to a NumPy array
        laser_power_numpy_array = np.array(laser_power_array)
        
        # Convert the NumPy array to a DataFrame
        laser_power_df = pd.DataFrame(laser_power_numpy_array)
    else:
        print("The Zarr array is not 2D. Additional processing may be required.")

    # Correcting the path, if necessary, to directly access the Zarr array
    laser_location_x_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_x"
    laser_location_y_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_y"
    laser_location_z_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_z"
    laser_time = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/timestamp"


    # Open the Zarr array directly (make sure this is the correct path to the array)
    laser_location_array_x = zarr.open(laser_location_x_path, mode='r')
    laser_location_array_y = zarr.open(laser_location_y_path, mode='r')
    laser_location_array_z = zarr.open(laser_location_z_path, mode='r')
    laser_time_array = zarr.open(laser_time, mode='r')

    # Depending on the structure of your Zarr array, you might directly convert it to a DataFrame
    # For a simple 2D array, it can be straightforward
    if laser_location_array_x.ndim == 2:
        # Convert the Zarr array to a NumPy array
        laser_location_numpy_array_x = np.array(laser_location_array_x)
        laser_location_numpy_array_y = np.array(laser_location_array_y)
        laser_location_numpy_array_z = np.array(laser_location_array_z)
        laser_time_numpy_array = np.array(laser_time_array)
        laser_time = np.array(laser_time_array)

        
        # Convert the NumPy array to a DataFrame
        laser_location_df_x = pd.DataFrame(laser_location_numpy_array_x)
        laser_location_df_y = pd.DataFrame(laser_location_numpy_array_y)
        laser_location_df_z = pd.DataFrame(laser_location_numpy_array_z)
        laser_time_df = pd.DataFrame(laser_time_numpy_array)

    # Ensure that the time data is sorted
    time_diffs = np.diff(laser_time.flatten())

    # Calculate the Euclidean distance between consecutive points
    distances = np.sqrt(
    np.diff(laser_location_numpy_array_x.flatten())**2 + 
    np.diff(laser_location_numpy_array_y.flatten())**2 + 
    np.diff(laser_location_numpy_array_z.flatten())**2
    )

    # Calculate the scanning speed (distance/time)
    scanning_speed = distances / time_diffs

    import pandas as pd

    import pandas as pd

    # Assuming you have already saved the dataframes as per your ranges
    laser_power_df_save = laser_power_df[min_time_step:max_time_step]
    laser_location_df_x_save = laser_location_df_x[min_time_step:max_time_step]
    laser_location_df_y_save = laser_location_df_y[min_time_step:max_time_step]
    laser_location_df_z_save = laser_location_df_z[min_time_step:max_time_step]
    laser_time_df_save = laser_time_df[min_time_step:max_time_step]
    scanning_speed_save = pd.DataFrame(scanning_speed)[min_time_step:max_time_step]

    # Concatenating all dataframes into one
    combined_df_input = pd.concat([
        laser_time_df_save.reset_index(drop=True),
        laser_power_df_save.reset_index(drop=True),
        laser_location_df_x_save.reset_index(drop=True),
        laser_location_df_y_save.reset_index(drop=True),
        laser_location_df_z_save.reset_index(drop=True),
        scanning_speed_save.reset_index(drop=True)
    ], axis=1)

    # Naming the columns of the combined dataframe for clarity
    combined_df_input.columns = ['Time', 'Laser_Power', 'Laser_Location_X', 'Laser_Location_Y', 'Laser_Location_Z', 'Scanning_Speed']

    import zarr
    import pandas as pd
    import numpy as np

    # Correcting the path, if necessary, to directly access the Zarr array
    active_nodes = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/ff_dt_active_nodes"

    # Open the Zarr array directly (make sure this is the correct path to the array)
    active_nodes = zarr.open(active_nodes, mode='r')

    # Depending on the structure of your Zarr array, you might directly convert it to a DataFrame
    # For a simple 2D array, it can be straightforward
    if active_nodes.ndim == 2:
        # Convert the Zarr array to a NumPy array
        active_nodes_numpy_array = np.array(active_nodes)
        
        # Convert the NumPy array to a DataFrame
        active_nodes_df = pd.DataFrame(active_nodes_numpy_array)
    else:
        print("The Zarr array is not 2D. Additional processing may be required.")

    active_nodes_df

    import zarr
    import pandas as pd
    import numpy as np

    # Correcting the path, if necessary, to directly access the Zarr array
    temperature_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/ff_dt_temperature"

    # Open the Zarr array directly (make sure this is the correct path to the array)
    temperature_array = zarr.open(temperature_path, mode='r')

    # Depending on the structure of your Zarr array, you might directly convert it to a DataFrame
    # For a simple 2D array, it can be straightforward
    if temperature_array.ndim == 2:
        # Convert the Zarr array to a NumPy array
        temperature_numpy_array = np.array(temperature_array)
        
        # Convert the NumPy array to a DataFrame
        temperature_df = pd.DataFrame(temperature_numpy_array)
    else:
        print("The Zarr array is not 2D. Additional processing may be required.")


    domain_nodes = pd.read_csv("/home/vnk3019/DT_DED/ME441_Projects/domain_nodes.csv")
    domain_nodes['node'] = domain_nodes.index
    domain_nodes_df = domain_nodes

    # Check if the number of nodes matches the number of rows in the temperature DataFrame
    if len(domain_nodes_df) != temperature_df.T.shape[0]:
        raise ValueError("The number of nodes does not match the number of rows in the temperature DataFrame.")

    # Combine both DataFrames
    # Set the index of the temperature DataFrame to be the node numbers
    temperature_df.T.index = domain_nodes_df.index

    # Concatenate the domain nodes DataFrame with the temperature DataFrame
    combined_df = pd.concat([domain_nodes_df, temperature_df.T], axis=1)
    #combined_df = combined_df[combined_df["z"]>0]

    # Step 1: Transpose active_nodes_df
    active_nodes_df_transposed = active_nodes_df.T

    # Step 2: Rename the columns of active_nodes_df_transposed to avoid collision
    # For example, we can add a prefix 'active_node_' to each column name
    active_nodes_df_transposed.columns = [f"active_node_{col}" for col in active_nodes_df_transposed.columns]

    # Step 3: Combine the transposed active_nodes_df with combined_df
    # Since the column names are now unique, we can concatenate them
    combined_result = pd.concat([combined_df, active_nodes_df_transposed], axis=1)

    combined_result = combined_result[combined_result["z"]>0]

    # Display the combined result
    combined_result

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    from ipywidgets import interactive
    import ipywidgets as widgets
    import os

    import numpy as np
    from scipy.interpolate import griddata

    # Assuming laser_x, laser_y, and laser_z are given for each timestep
    laser_x = combined_df_input[["Laser_Location_X"]].to_numpy().flatten()
    laser_y = combined_df_input[["Laser_Location_Y"]].to_numpy().flatten()
    laser_z = combined_df_input[["Laser_Location_Z"]].to_numpy().flatten()

    # Initialize a list to store temperature matrices
    temperature_matrices = []

    # Function to calculate the global min and max temperature
    def get_global_temperature_range():
        all_temperatures = []
        for time_step in range(285, 1426):  # Assuming time steps from 285 to 1425
            temperature = combined_result[time_step].copy()
            all_temperatures.extend(temperature)
        global_min = max(300, np.min(all_temperatures))  # Ensure minimum is at least 300
        global_max = np.max(all_temperatures)
        return global_min, global_max

    # Function to interpolate NaN values using linear and nearest methods
    def interpolate_nan(grid_x, grid_z, grid_temperature):
        nan_indices = np.isnan(grid_temperature)
        
        # Interpolate over NaN points with linear method
        if np.any(nan_indices):
            valid_points = np.column_stack((grid_x[~nan_indices], grid_z[~nan_indices]))
            valid_temperature = grid_temperature[~nan_indices]
            
            try:
                grid_temperature[nan_indices] = griddata(
                    valid_points, valid_temperature,
                    (grid_x[nan_indices], grid_z[nan_indices]),
                    method='linear'
                )
            except Exception as e:
                # If linear interpolation fails, fall back to nearest-neighbor
                grid_temperature[nan_indices] = griddata(
                    valid_points, valid_temperature,
                    (grid_x[nan_indices], grid_z[nan_indices]),
                    method='nearest'
                )
        
        # Fill any remaining NaN values with nearest-neighbor interpolation
        if np.any(np.isnan(grid_temperature)):
            grid_temperature[nan_indices] = griddata(
                valid_points, valid_temperature,
                (grid_x[nan_indices], grid_z[nan_indices]),
                method='nearest'
            )
        
        return grid_temperature

    # Function to collect the temperature matrix for each time step without plotting
    def collect_temperature_matrices(start_step, end_step, box_size):
        global_min_temp, global_max_temp = get_global_temperature_range()
        
        grid_x_absolute = None  # Initialize within the function
        grid_z_absolute = None  # Initialize within the function
        
        for time_step in range(start_step, end_step + 1):
            x = combined_result['x']
            z = combined_result['z']
            
            # Get the temperature values for the given time step
            temperature = combined_result[time_step].copy()
            
            # Define the boundaries of the zoomed-in box based on laser location and direction of movement
            laser_loc_x = laser_x[time_step - start_step - 1]
            laser_loc_z = laser_z[time_step - start_step - 1]
            
            # Determine direction of movement (left to right or right to left)
            if time_step > start_step:
                previous_laser_x = laser_x[time_step - start_step - 2]
                if laser_loc_x > previous_laser_x:
                    x_min = laser_loc_x - box_size
                    x_max = laser_loc_x
                else:
                    x_min = laser_loc_x
                    x_max = laser_loc_x + box_size
            else:
                x_min = laser_loc_x - box_size
                x_max = laser_loc_x
            
            z_min = laser_loc_z - box_size
            z_max = laser_loc_z
            
            # Add a small noise to avoid co-planar issues
            noise = 1e-6 * np.random.rand(len(x))
            x_noisy = x + noise
            z_noisy = z + noise
            
            # Create a structured grid within the user-defined box for the contour plot, ensuring origin-based coordinates
            grid_x_box, grid_z_box = np.mgrid[x_min:x_max:30j, z_min:z_max:30j]
            
            # Perform grid interpolation for temperature data using griddata (ensures proper orientation)
            points = np.column_stack((x_noisy, z_noisy))
            grid_temperature_box = griddata(points, temperature, (grid_x_box, grid_z_box), method='linear')
            grid_temperature_box = np.maximum(grid_temperature_box, 300)  # Ensure minimum temperature is 300
            
            # Interpolate any NaN values in the grid_temperature_box
            grid_temperature_box = interpolate_nan(grid_x_box, grid_z_box, grid_temperature_box)
            
            # Ensure that the grid_x and grid_z are consistent and absolute across all time steps
            if grid_x_absolute is None and grid_z_absolute is None:
                grid_x_absolute = grid_x_box
                grid_z_absolute = grid_z_box
            
            # Save the temperature matrix corresponding to this time step
            temperature_matrices.append(grid_temperature_box)

        # Convert the list of temperature matrices to a 3D array
        temperature_3d_array = np.array(temperature_matrices)
        
        return temperature_3d_array, grid_x_absolute, grid_z_absolute

    # Collect temperature matrices for time steps from 570 to 1300
    min_time_step = 570
    max_time_step = 1300
    box_size = 1.0

    # Get the temperature matrix and the grid information
    temperature_3d_array, grid_x_absolute, grid_z_absolute = collect_temperature_matrices(min_time_step, max_time_step, box_size)

    import scipy.io

    # Load the .mat file
    loaded_data = scipy.io.loadmat('/home/vnk3019/DED_melt_pool_thinwall_data_dict.mat')

    # Extract the 'temperature' and 'input_data' arrays
    loaded_temperature_data = loaded_data['temperature']
    loaded_input_data = loaded_data['input_data']

    # Concatenate (duplicate) the data by doubling it
    new_temperature_data = np.concatenate([loaded_temperature_data, temperature_3d_array], axis=0)
    new_doubled_input_data = np.concatenate([loaded_input_data, combined_df_input.to_numpy()], axis=0)

    # Create a dictionary with the key as 'temperature' and value as the matrix
    DED_thinwall_data_dict = {'temperature': new_temperature_data, 'input_data': new_doubled_input_data}


    # Save the dictionary as a .mat file
    scipy.io.savemat('/home/vnk3019/DED_melt_pool_thinwall_data_dict.mat', DED_thinwall_data_dict)

    print("Temperature data has been saved as 'DED_melt_pool_thinwall_data_dict.mat'")

    import os 
    import shutil

    os.remove(f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr.tar.gz")
    shutil.rmtree(f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr")

