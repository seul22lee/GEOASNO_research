import subprocess
import os


for laser_power_profile_number in range(12,26):

    def download_with_progress(source_path, target_directory):
        # Define the path for the local download
        local_compressed_file_path = os.path.join(target_directory, os.path.basename(source_path))
        
        # Download the compressed file
        subprocess.run(['rclone', 'copy', source_path, target_directory], check=True)
        print(f"Downloaded {source_path} to {local_compressed_file_path}")

    # Example usage
    download_with_progress(f'DED_DT:/DED_DT_Data_Zarr_316L_printed_cube/compressed_laser_profile_{laser_power_profile_number}.zarr.zarr.tar.gz', '/home/vnk3019/DT_DED/ME441_Projects')

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


    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from ipywidgets import interact, IntSlider
    import numpy as np

    # Assuming nodes_df is already defined and contains 'x', 'y', 'z' columns
    # If not, you'll need to read the CSV file as before
    filename = '/home/vnk3019/DT_DED/DED_DT_Thermomechanical_Solver/src/domain_nodes.csv'
    nodes_df = pd.read_csv(filename)

    # Add a 'node_number' column starting from 0
    nodes_df['node_number'] = np.arange(len(nodes_df))

    import zarr
    import pandas as pd
    import numpy as np

    def get_nodes_above_solidus_dataframe(file_path, solidus_temperature):
        """
        Collect node numbers that are above the solidus temperature over time and save in a DataFrame.
        
        :param file_path: Path to the Zarr array.
        :param solidus_temperature: The solidus temperature to compare against.
        :return: A pandas DataFrame with time steps and node numbers above solidus temperature.
        """
        # Open the Zarr array directly

        zarr_array = zarr.open(file_path, mode='r')

        # Ensure the Zarr array is 2D (time steps x nodes)
        if zarr_array.ndim != 2:
            print("The Zarr array is not 2D. Additional processing may be required.")
            return
        
        # Convert the Zarr array to a NumPy array
        numpy_array = np.array(zarr_array)
        
        # List to accumulate results
        results = []
        
        # Iterate over each time step (row in the numpy array)
        for time_step, temperatures in enumerate(numpy_array):
            # Find indices (node numbers) where temperature is above the solidus temperature

            nodes = np.where(temperatures > solidus_temperature)[0]
            
            # Append a dictionary for each time step with nodes above the solidus temperature
            results.append({"Time Step": time_step, "Nodes Above Solidus": nodes.tolist()})

        # Convert results to a DataFrame
        df_results = pd.DataFrame(results)

        
        return df_results
    # Example usage
    file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/ff_dt_temperature"
    solidus_temperature = 400  # Example solidus temperature, adjust this value as needed

    #solidus_temperature = 1000  # Example solidus temperature, adjust this value as needed

    df_nodes_above_solidus = get_nodes_above_solidus_dataframe(file_path, solidus_temperature)


    import pandas as pd
    import numpy as np
    import zarr

    # Load toolpath data from a CSV file
    file_path = '/home/vnk3019/DT_DED/ME441_Projects/toolpath_data.csv'
    toolpath_data = pd.read_csv(file_path)

    # Display the first few rows to confirm structure
    print(toolpath_data.head())

    # Desired number of samples for interpolation
    desired_samples = 46002

    # Interpolating the toolpath data to reduce it to the desired number of samples
    df_reduced_laser_location = pd.DataFrame({
        'X': np.interp(np.linspace(0, len(toolpath_data)-1, desired_samples), np.arange(len(toolpath_data)), toolpath_data['X']),
        'Y': np.interp(np.linspace(0, len(toolpath_data)-1, desired_samples), np.arange(len(toolpath_data)), toolpath_data['Y']),
        'Z': np.interp(np.linspace(0, len(toolpath_data)-1, desired_samples), np.arange(len(toolpath_data)), toolpath_data['Z']),
        'Laser State': np.round(np.interp(np.linspace(0, len(toolpath_data)-1, desired_samples), np.arange(len(toolpath_data)), toolpath_data['Laser State']))
    })

    # Ensure Laser State is binary after interpolation
    df_reduced_laser_location['Laser State'] = (df_reduced_laser_location['Laser State'] > 0.5).astype(int)

    # Find indexes where Laser State is 1 (i.e., laser is on)
    indexes_laser_on = df_reduced_laser_location.index[df_reduced_laser_location['Laser State'] == 1].tolist()
    print("Indexes where Laser State is on:", indexes_laser_on)

    import zarr
    import pandas as pd
    import numpy as np

    def load_and_combine_zarr(file_paths, dt):
        """
        Loads Zarr arrays from specified file paths and combines them into a single DataFrame with a time index.
        
        Args:
            file_paths (dict): Dictionary of file paths for 'x', 'y', and 'z' coordinates.
            dt (float): Time interval between each data point in seconds.
        
        Returns:
            DataFrame: Combined DataFrame with columns ['time', 'x', 'y', 'z'].
        """
        # Initialize a dictionary to hold data frames for each coordinate
        data_frames = {}
        for coord, path in file_paths.items():
            zarr_array = zarr.open(path, mode='r')
            if zarr_array.ndim != 2 or zarr_array.shape[1] != 1:
                raise ValueError(f"The Zarr array at {path} is not a single-column 2D array.")
            # Convert the Zarr array to a DataFrame
            data_frames[coord] = pd.DataFrame(zarr_array[:], columns=[coord])
        
        # Combine the DataFrames into a single DataFrame with 'x', 'y', 'z' columns
        combined_df = pd.concat(data_frames.values(), axis=1)
        # Create a time index and add it to the DataFrame
        combined_df['time'] = np.arange(0, len(combined_df) * dt, dt)
        return combined_df

    # Define time step in seconds (e.g., 0.01 seconds between each point)
    time_step = 1  # Adjust this based on your actual data sampling rate

    # Paths to the Zarr arrays for x, y, and z coordinates
    file_paths = {
        'x': f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_x",
        'y': f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_y",
        'z': f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_pos_z"
    }

    # Load and combine Zarr arrays into a DataFrame with a time index
    tool_path_df = load_and_combine_zarr(file_paths, time_step)

    # Calculate differences to find direction changes
    tool_path_df[['dx', 'dy', 'dz']] = tool_path_df[['x', 'y', 'z']].diff()

    # Handle potential NaN values from diff operation
    tool_path_df.fillna(0, inplace=True)

    # Calculate unit vectors
    magnitude = np.sqrt(tool_path_df['dx']**2 + tool_path_df['dy']**2 + tool_path_df['dz']**2)
    # Avoid division by zero; set zero magnitudes to 1
    magnitude[magnitude == 0] = 1
    unit_vectors = tool_path_df[['dx', 'dy', 'dz']].div(magnitude, axis='index')
    unit_vectors.columns = ['unit_dx', 'unit_dy', 'unit_dz']

    # Append unit vectors to the main DataFrame with time index
    tool_path_df[['unit_dx', 'unit_dy', 'unit_dz']] = unit_vectors

    unit_vectors = tool_path_df[['time', 'unit_dx', 'unit_dy', 'unit_dz']]

    laser_on_unit_vectors = unit_vectors.loc[indexes_laser_on]

    direction_changes_df = laser_on_unit_vectors


    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    # Load node locations (assuming this is already loaded as nodes_df with 'x', 'y', 'z', 'node_number')
    # Load node temperature data from the Zarr file
    temp_file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/ff_dt_temperature"
    zarr_temp_array = zarr.open(temp_file_path, mode='r')
    temperature_data = np.array(zarr_temp_array).squeeze()  # Assuming it's a 2D array (time steps x nodes)
    # Pre-define lists to accumulate data
    time_steps = []
    xs = []
    ys = []
    zs = []
    temperatures = []

    for specific_time_step in tqdm(range(len(temperature_data)), desc="Processing Time Steps"):
        specific_temperatures = temperature_data[specific_time_step, :]
        nodes_df['temperature'] = specific_temperatures

        # Filter nodes above solidus temperature and find the max z
        nodes_above_solidus = nodes_df[nodes_df['temperature'] > solidus_temperature]
        max_z = nodes_above_solidus['z'].max()

        # Filter for nodes with max_z
        for _, row in nodes_above_solidus[nodes_above_solidus['z'] == max_z].iterrows():
            time_steps.append(specific_time_step)
            xs.append(row['x'])
            ys.append(row['y'])
            zs.append(row['z'])
            temperatures.append(row['temperature'])

    # Convert lists to DataFrame
    all_nodes_with_max_z = pd.DataFrame({
        'time_step': time_steps,
        'x': xs,
        'y': ys,
        'z': zs,
        'temperature': temperatures
    })

    print(all_nodes_with_max_z.head())

    #all_nodes_with_max_z.to_csv("all_nodes_with_max_z_over_time.csv", index=False)


    # Load the time index from the Zarr file
    file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/timestamp"
    zarr_array = zarr.open(file_path, mode='r')
    time_index_array = np.array(zarr_array).squeeze()  # Assuming it's a 1D array

    def find_intersections(line_positions, temperatures, temp_threshold):
        intersections = []
        for i in range(len(temperatures) - 1):
            if np.isnan(temperatures[i]) or np.isnan(temperatures[i+1]):
                continue  # Skip masked (NaN) regions
            if temperatures[i] < temp_threshold <= temperatures[i+1] or temperatures[i] > temp_threshold >= temperatures[i+1]:
                ratio = (temp_threshold - temperatures[i]) / (temperatures[i+1] - temperatures[i])
                position = line_positions[i] + ratio * (line_positions[i+1] - line_positions[i])
                intersections.append(position)
        return intersections

    import numpy as np
    import pandas as pd
    from scipy.interpolate import Rbf
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm

    # Placeholder for your data loading
    # Assuming all_nodes_with_max_z and direction_changes_df are already loaded.

    solidus_temp = 1648.15
    liquidus_temp = 1673.15

    def select_opposite_direction_points(unit_vector, solidus_intersections, liquidus_intersections):
        if len(solidus_intersections) < 2 or len(liquidus_intersections) < 2:
            return [], []
        direction = 'left' if unit_vector['unit_dx'] <= 0 else 'right'
        if direction == 'right':
            selected_solidus = [min(solidus_intersections)]
            selected_liquidus = [min(liquidus_intersections)]
        else:
            selected_solidus = [max(solidus_intersections)]
            selected_liquidus = [max(liquidus_intersections)]
        return selected_solidus, selected_liquidus

    def calculate_slope(x1, y1, x2, y2):
        if x2 - x1 == 0:
            return np.inf
        return (y2 - y1) / (x2 - x1)

    def fit_temperature_surface(time_step, specific_data):
        if specific_data.empty or len(specific_data) < 2:
            return None, None
        try:
            rbf = Rbf(specific_data['x'], specific_data['y'], specific_data['temperature'], function='linear')
            return rbf, specific_data
        except ZeroDivisionError:
            return None, None

    def calculate_line(unit_vector, max_temp_point, line_length):
        if abs(unit_vector['unit_dx']) > abs(unit_vector['unit_dy']):
            line_x = np.linspace(max_temp_point['x'] - line_length/2, max_temp_point['x'] + line_length/2, num=100)
            line_y = np.full(100, max_temp_point['y'])
        else:
            line_x = np.full(100, max_temp_point['x'])
            line_y = np.linspace(max_temp_point['y'] - line_length/2, max_temp_point['y'] + line_length/2, num=100)
        return line_x, line_y

    def calculate_melt_pool_width(line_x, line_y, line_z, solidus_temp):
        intersections = np.where(np.diff(np.sign(line_z - solidus_temp)) != 0)[0]
        if len(intersections) >= 2:
            x_coords = line_x[intersections]
            y_coords = line_y[intersections]
            width = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
            return width
        return 0

    # Initialize lists to store results
    time_step_with_slopes = []
    melt_pool_widths = []

    # Define the time steps range where the laser is on
    time_steps = np.linspace(0, len(time_index_array), num=len(time_index_array), dtype=int)

    # Process only the time steps where the laser is on
    for time_step in tqdm(indexes_laser_on, desc='Calculating slopes and melt pool widths'):
        specific_data = all_nodes_with_max_z[all_nodes_with_max_z['time_step'] == time_step]
        rbf, _ = fit_temperature_surface(time_step, specific_data)
        if rbf is None:
            time_step_with_slopes.append((time_step, 0))
            melt_pool_widths.append((time_step, 0))
            continue

        unit_vector = direction_changes_df.loc[time_step]
        max_temp_point = specific_data.loc[specific_data['temperature'].idxmax(), ['x', 'y']]
        line_length = 6
        line_x, line_y = calculate_line(unit_vector, max_temp_point, line_length)
        line_z = rbf(line_x, line_y)
        width = calculate_melt_pool_width(line_x, line_y, line_z, solidus_temp)
        melt_pool_widths.append((time_step, width))

        solidus_intersections = find_intersections(line_x, line_z, solidus_temp)
        liquidus_intersections = find_intersections(line_x, line_z, liquidus_temp)
        selected_solidus, selected_liquidus = select_opposite_direction_points(unit_vector, solidus_intersections, liquidus_intersections)

        if len(selected_solidus) == 1 and len(selected_liquidus) == 1:
            slope = calculate_slope(selected_solidus[0], solidus_temp, selected_liquidus[0], liquidus_temp)
            time_step_with_slopes.append((time_step, abs(slope)))
        else:
            time_step_with_slopes.append((time_step, 0))

    # Convert lists of tuples to DataFrames for better manipulation and saving
    slopes_df = pd.DataFrame(time_step_with_slopes, columns=['Time Step', 'Slope'])
    widths_df = pd.DataFrame(melt_pool_widths, columns=['Time Step', 'Melt Pool Width'])

    # Print first few rows to see the results
    print(slopes_df.head())
    print(widths_df.head())

    slopes_array = slopes_df['Slope']

    slopes_array_pd = pd.Series(slopes_array)  # Replace with your data array


    # Apply a rolling window maximum
    window_size = 150  # Window size for the rolling maximum
    rolling_max = slopes_array_pd.rolling(window=window_size, min_periods=1).max()


    # Apply Exponential Moving Average (EMA) to smooth the data
    ema_span = 100  # Define the span for EMA which provides a degree of smoothing
    smoothed_data_ema = rolling_max.ewm(span=ema_span, adjust=False).mean()

    # Extracting widths only and applying cleanup
    melt_pool_widths_only = [width for _, width in melt_pool_widths]

    # Replace None with 0 and change values greater than 4 to 0, then create a DataFrame
    melt_pool_widths_cleaned = pd.DataFrame({'Width': [0 if (width is None or width > 3) else width for width in melt_pool_widths_only]})

    widths_df = pd.DataFrame(melt_pool_widths, columns=['Time Step', 'Melt Pool Width'])

    # Concatenate the cleaned widths DataFrame with the time steps DataFrame
    combined_df_melt_width = pd.concat([widths_df['Time Step'], melt_pool_widths_cleaned], axis=1)

    combined_df_melt_width['Width'] = combined_df_melt_width['Width'].replace(0).ffill()

    import pandas as pd
    import matplotlib.pyplot as plt

    # Assuming 'melt_pool_widths' is a list of melt pool widths obtained from measurements or calculations
    melt_pool_widths_series = pd.Series(combined_df_melt_width['Width'])  # Convert list to a pandas Series

    # Set the window sizes for different stages of rolling operations
    initial_window_size = 5  # Window size for initial rolling minimum
    secondary_window_size = 100  # Window size for secondary rolling maximum and mean

    # Apply a rolling window operation to smooth and derive maximum stable values
    smoothed_min_widths = melt_pool_widths_series.rolling(window=initial_window_size, min_periods=1).min()
    smoothed_max_widths = smoothed_min_widths.rolling(window=secondary_window_size, min_periods=1).max()
    final_smoothed_widths = smoothed_max_widths.ewm(span=ema_span, adjust=False).mean()

    from scipy.spatial.qhull import QhullError
    from scipy.interpolate import griddata
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider
    import pandas as pd
    all_nodes_with_max_z = all_nodes_with_max_z[all_nodes_with_max_z['time_step'].isin(indexes_laser_on)]


    solidus_temp = 1648.15  # Define your solidus temperature

    # Create a grid
    grid_x, grid_y = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))

    # Calculate melt pool temperatures for all timesteps
    def calculate_melt_pool_temperatures(group, solidus_temp):
        points = group[['x', 'y']].values
        temperatures = group['temperature'].values
        grid_x, grid_y = np.meshgrid(np.linspace(-6, 6, 500), np.linspace(-6, 6, 500))

        if points.shape[0] < 4:  # Check if there are enough points for triangulation
            print(f"Not enough points for interpolation. Skipping.")
            return np.nan

        try:
            grid_z = griddata(points, temperatures, (grid_x, grid_y), method='linear')
        except Exception as e:  # Catch any exception related to griddata
            print(f"Interpolation failed with error: {e}. Falling back to 'nearest'.")
            grid_z = griddata(points, temperatures, (grid_x, grid_y), method='nearest')

        melt_pool_mask = grid_z >= solidus_temp
        return np.nanmean(grid_z[melt_pool_mask]) if np.any(melt_pool_mask) else np.nan

    # Group data by time_step and calculate
    melt_pool_temperatures = {}
    for time_step, group in tqdm(all_nodes_with_max_z.groupby('time_step'), desc="Calculating melt pool temperatures"):
        melt_pool_temperatures[time_step] = calculate_melt_pool_temperatures(group, solidus_temp)

    # Convert to DataFrame
    melt_pool_df = pd.DataFrame(list(melt_pool_temperatures.items()), columns=['time_step', 'melt_pool_temperature'])

    # Assuming melt_pool_df is your DataFrame and 'melt_pool_temperature' is the column with NaN values
    melt_pool_df['melt_pool_temperature'] = melt_pool_df['melt_pool_temperature'].fillna(0)

    # Assuming melt_pool_df is your DataFrame and 'melt_pool_temperature' is the column you're working with

    # First, fill NaN values with 0
    melt_pool_df['melt_pool_temperature'].fillna(0, inplace=True)

    # Convert the temperature column to a pandas Series (if it's not already)
    melt_pool_temperature_series = pd.Series(melt_pool_df['melt_pool_temperature'])

    # Apply a rolling window operation
    window_size = 10  # Define the window size for the rolling operation
    smoothed_temperature = melt_pool_temperature_series.rolling(window=window_size, min_periods=1).min()
    smoothed_temperature = smoothed_temperature.rolling(window=50, min_periods=1).max()
    smoothed_temperature = smoothed_temperature.rolling(window=50, min_periods=1).mean()

    import zarr
    import pandas as pd
    import numpy as np

    # Correcting the path, if necessary, to directly access the Zarr array
    file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_laser_power"

    # Open the Zarr array directly (make sure this is the correct path to the array)
    zarr_array = zarr.open(file_path, mode='r')

    # Depending on the structure of your Zarr array, you might directly convert it to a DataFrame
    # For a simple 2D array, it can be straightforward
    if zarr_array.ndim == 2:
        # Convert the Zarr array to a NumPy array
        numpy_array = np.array(zarr_array)
        
        # Convert the NumPy array to a DataFrame
        df = pd.DataFrame(numpy_array)
    else:
        print("The Zarr array is not 2D. Additional processing may be required.")


    import zarr
    import pandas as pd
    import numpy as np

    # Define the file path to the Zarr array that contains the laser power data
    file_path = f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr/dt_laser_power"

    # Open the Zarr array directly (ensure this is the correct path to the array)
    zarr_array = zarr.open(file_path, mode='r')

    # Initialize a DataFrame from the Zarr array if it is 2D
    if zarr_array.ndim == 2:
        # Convert the Zarr array to a NumPy array
        numpy_array = np.array(zarr_array)
        
        # Convert the NumPy array to a DataFrame and rename columns for clarity
        laser_power_df = pd.DataFrame(numpy_array)
    else:
        print("The Zarr array is not 2D. Additional processing may be required.")

    # Assuming indexes_laser_on contains the indices where the laser is on
    # We will mask the DataFrame to only include rows where the laser is on
    laser_power_on_df = laser_power_df.iloc[indexes_laser_on]
    laser_power_on_df = laser_power_on_df*2.5

    # Creating a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comprehensive Data Analysis of Laser and Melt Pool Metrics of laser power {laser_power_profile_number}')

    # Plotting in subplot grid
    axes[0, 0].plot(laser_power_on_df, color='black')
    axes[0, 0].set_title('Laser Power Dynamics Over Time')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Laser Power (W/cm²)')
    axes[0, 0].legend()

    axes[0, 1].plot(smoothed_data_ema, color='green')
    axes[0, 1].set_title('Thermal Gradient')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Thermal Gradient (K/mm)')
    axes[0, 1].legend()

    axes[1, 0].plot(final_smoothed_widths, color='blue')
    axes[1, 0].set_title('Trends in Smoothed Melt Pool Widths')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Melt Pool Width (mm)')
    axes[1, 0].legend()

    axes[1, 1].plot(smoothed_temperature, color='red')
    axes[1, 1].set_title('Melt Pool Temperature Profile')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Temperature (K)')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the global title
    plt.savefig(f'laser_melt_pool_analysis_{laser_power_profile_number}.png', format='png', dpi=300)  # Save as high-resolution PNG file
    plt.show()

    # New index for interpolation
    new_index = np.linspace(start=0, stop=len(smoothed_temperature)-1, num=26268)

    # Interpolating temperature data
    # 'np.interp' function parameters: x-coordinates of the interpolated values, 
    # x-coordinates of the data points, y-coordinates of the data points
    interpolated_temperatures = np.interp(new_index, np.arange(len(smoothed_temperature)), smoothed_temperature)

    # Creating a new DataFrame for the interpolated data
    interpolated_temperature_df = pd.DataFrame({'Temperature': interpolated_temperatures}, index=new_index)

    index_laser_on_df = pd.DataFrame({'laser_state': indexes_laser_on})

    # Using .to_numpy() to ignore DataFrame indices
    combined_df_laser_power_1 = pd.concat([
        pd.DataFrame(index_laser_on_df['laser_state'].to_numpy()),
        pd.DataFrame(laser_power_on_df[0].to_numpy()),
        pd.DataFrame(smoothed_data_ema.to_numpy()),
        pd.DataFrame(final_smoothed_widths.to_numpy()),
        pd.DataFrame(interpolated_temperature_df['Temperature'].to_numpy())
    ], axis=1, ignore_index=True)

    # Optionally, set column names after concatenation
    combined_df_laser_power_1.columns = ['Laser State', 'Laser Power', 'Thermal Gradient', 'Melt Pool Width', 'Temperature']

    # Assuming 'combined_df_laser_power' is your DataFrame
    combined_df_laser_power_1.to_csv(f'combined_df_laser_power_{laser_power_profile_number}.csv', index=False)

    import os 
    import shutil

    os.remove(f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr.tar.gz")
    shutil.rmtree(f"/home/vnk3019/DT_DED/ME441_Projects/laser_profile_{laser_power_profile_number}.zarr")
