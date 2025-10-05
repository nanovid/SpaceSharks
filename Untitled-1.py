import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from netCDF4 import Dataset
import xarray as xr


filename = "./NeurOST_SSH-SST_20241201_20250206.nc"
dataset = Dataset("./NeurOST_SSH-SST_20241201_20250206.nc", "r")
Omega = 7.2921e-5 # Earth's angular velocity in rad/s


#print(dataset)

#print(dataset.variables.keys()) 

#lon_grid = dataset.variables['longitude'][:]
#lat_grid = dataset.variables['latitude'][:]
#sla_to_plot = dataset.variables['sla'][:]



# Use xarray to open the dataset.
print(f"Attempting to load data from: {filename}")
ds = xr.open_dataset(filename, engine="netcdf4")

print("\nSuccessfully loaded dataset. Here is a summary:")
print(ds) # Printing the dataset gives a helpful overview

# Extract the data into numpy arrays. The variable names in the square
# brackets (e.g., 'sla') MUST match the names in your NetCDF file.
lon = ds['longitude'].values
lat = ds['latitude'].values
sla_data_all_times = ds['sla'].values
#print(sla_data_all_times)
ugosa_data_all_times = ds['ugosa'].values
vgosa_data_all_times = ds['vgosa'].values
zeta_data_all_times = ds['zeta'].values
# You might also want to load the time variable
# time_vector = ds['time'].values


def is_eddy(lat, zeta, alpha=0.2):
    f = 2 * Omega * np.sin(np.deg2rad(lat))
    cyclonic_mask = (zeta > alpha * np.abs(f)) ## True where cyclonic
    anticyclonic_mask = (zeta < -alpha * np.abs(f)) ## True where anticyclonic
    if cyclonic_mask:
        return 1
    elif anticyclonic_mask:
        return -1
    else:
        return 0

#zeta_test = map(lambda x, y: is_eddy(x, y), lat, zeta_data_all_times[:,:,0])
#print(list(zeta_test))
# --- 2. Prepare Data for a Single Time Step ---
# NetCDF data is often a time series. We must select one time step to plot.


time_index = 0 # Select the first time step. Change this index to see other times.

print(f"\nPreparing to plot data for time_index = {time_index}")

# Slice the 3D arrays to get a 2D array for the chosen time step.
# This assumes the dimensions are ordered (time, latitude, longitude).
# If your order is different, you may need to adjust the slicing (e.g., sla_data[:,:,time_index]).
sla_to_plot = sla_data_all_times[:, :, time_index]
ugosa_to_plot = ugosa_data_all_times[:, :, time_index]
vgosa_to_plot = vgosa_data_all_times[:, :, time_index]
zeta_to_plot = zeta_data_all_times[:, :, time_index]

#calculations
num_latitudes, num_longitudes = zeta_to_plot.shape
eddy = np.zeros((num_latitudes, num_longitudes))
# Use nested loops to iterate through each index (i, j)
for i in range(num_latitudes):
    for j in range(num_longitudes):
        # Get the values at the specific grid point (i, j)
        current_lat = lat[i]
        current_lon = lon[j]
        current_zeta = zeta_to_plot[i, j]
        eddy_type = is_eddy(current_lat, current_zeta)
        eddy[i,j] = eddy_type



# --- 3. Plot 1: Sea Level Anomaly (SLA) Map ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
# Use pcolormesh for grid-based data. Use shading='auto' or 'gouraud'.
cf = ax1.pcolormesh(lon, lat, sla_to_plot, cmap='jet', shading='auto')

fig1.colorbar(cf, ax=ax1, label='Sea Level Anomaly (m)')
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_title(f'Map of Sea Level Anomaly (sla) at Time Index {time_index}')
# Using 'equal' aspect can cause distortion with global data, 'auto' is often safer.
ax1.set_aspect('auto', adjustable='box')


# --- 4. Plot 2: Zeta with Geostrophic Current Vectors ---
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Plot the SLA as the background
cf2 = ax2.pcolormesh(lon, lat, zeta_to_plot, cmap='jet', shading='auto')
fig2.colorbar(cf2, ax=ax2, label='Zeta (1/s)')

#eddy type
fig1, ax1 = plt.subplots(figsize=(10, 8))
# Use pcolormesh for grid-based data. Use shading='auto' or 'gouraud'.
cf = ax1.pcolormesh(lon, lat, eddy, cmap='jet', shading='auto')

""" # Downsample the vector data for a cleaner plot
skip = 50 # Plot a vector every 50th grid point. Adjust as needed.
lon_sub = lon[::skip, ::skip]
lat_sub = lat[::skip, ::skip]
u_sub = ugosa_to_plot[::skip, ::skip]
v_sub = vgosa_to_plot[::skip, ::skip]

# Overlay the vector plot using quiver
ax2.quiver(lon_sub, lat_sub, u_sub, v_sub, color='black', scale=20, width=0.003)

ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title(f'SLA with Current Vectors at Time Index {time_index}')
ax2.set_aspect('auto', adjustable='box') """

# --- Display the plots ---
plt.tight_layout()
plt.show()

print(zeta_data_all_times)