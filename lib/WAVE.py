#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import deepwave
from deepwave import scalar, scalar_born
import os
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#
import matplotlib
params = {
	'savefig.dpi': 300,
	'figure.dpi' : 300,
	'axes.labelsize':12,
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

class WAVE:
	"""Class for acoustic wave propagation and visualization"""

	@staticmethod
	def clip(model, perc):
		"""Clip model values based on percentile"""
		(ROWs, COLs) = model.shape
		reshape2D_1D = model.reshape(ROWs*COLs)
		reshape2D_1D = np.sort(reshape2D_1D)
		if perc != 100:
			min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
			max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
		elif perc == 100:
			min_num = min(model.flatten())
			max_num = max(model.flatten())
		if min_num > max_num:
			dummy = max_num
			max_num = min_num
			min_num = dummy
		return max_num, min_num

	@staticmethod
	def get_wavefield(vp, dx, dt, freq, nt, device, source_location):
		"""Generate wavefield using deepwave scalar propagation"""
		peak_time = 1.5 / freq # The time at which the Ricker wavelet reaches its peak
		return scalar(vp,            # Velocity model
					  dx,            # Spatial sampling interval
					  dt,            # Temporal sampling interval
					  source_amplitudes=(
					  deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(device)), # Source wavelet
					  source_locations=source_location,  # Location of the source in the grid
					  accuracy=8,    # Accuracy of the finite difference stencil
					  pml_freq=freq) # Perfectly Matched Layer frequency to absorb boundary reflections

	@staticmethod
	def plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, save_path):
		"""Plot wave propagation at multiple time steps"""
		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		plt.figure(figsize=(10, 10))
		wavefields = [WAVE.get_wavefield(vp, dx, dt, freq, nt, device, source_location) for nt in time_steps]
		# Extract source location from the tensor
		pml_thickness = 20
		source_y = (source_location[0, 0, 0] + pml_thickness).item()
		source_x = (source_location[0, 0, 1] + pml_thickness).item()
		for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
			plt.subplot(2, 2, idx)
			wave_data = wavefield[0][0, :, :].cpu().numpy() # extract array from tensor and move to CPU
			max_num, min_num = WAVE.clip(wave_data, 100)
			plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
			plt.scatter(source_x, source_y, c='blue', s=50)  # Plot blue dot at source location
			plt.xlabel('X Distance (m)')
			plt.ylabel('Y Distance (m)')
			plt.title(f"Time Step: {nt} ms")
		plt.subplots_adjust(wspace=0.1, hspace=0.4)
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.show()

	@staticmethod
	def box(vp, dx, dt, freq, time_steps, device, source_location, box_start_x, box_start_y,
			box_end_x, box_end_y, save_path):
		"""Plot wave propagation with a box overlay at multiple time steps"""
		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		plt.figure(figsize=(10, 10))
		wavefields = [WAVE.get_wavefield(vp, dx, dt, freq, nt, device, source_location) for nt in time_steps]
		pml_thickness = 20
		source_y = (source_location[0, 0, 0] + pml_thickness).item()
		source_x = (source_location[0, 0, 1] + pml_thickness).item()
		box_start_x += pml_thickness; box_end_x += pml_thickness
		box_start_y += pml_thickness; box_end_y += pml_thickness
		for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
			plt.subplot(2, 2, idx)
			wave_data = wavefield[0][0, :, :].cpu().numpy() # extract array from tensor and move to CPU
			max_num, min_num = WAVE.clip(wave_data, 98)
			plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
			plt.scatter(source_x, source_y, c='blue', s=50)
			rect = patches.Rectangle((box_start_x, box_start_y),
									box_end_x - box_start_x,
									box_end_y - box_start_y,
									linewidth=1, edgecolor='r', facecolor='none')
			plt.gca().add_patch(rect)
			plt.xlabel('X Distance (m)')
			plt.ylabel('Y Distance (m)')
			plt.title(f"Time Step: {nt} ms")
		plt.subplots_adjust(wspace=0.1, hspace=0.4)
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.show()

	@staticmethod
	def wavefield(freq, nt, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
				  source_locations, receiver_locations):
		"""Generate wavefield with receivers using deepwave scalar propagation"""
		source_amplitudes = (deepwave.wavelets.ricker(freq, nt, dt, peak_time)
							 .repeat(n_shots, n_sources_per_shot, 1).to(device))
		outputs = scalar(vp, dx, dt,
						 source_amplitudes=source_amplitudes,
						 source_locations=source_locations,
						 receiver_locations=receiver_locations,
						 accuracy=8,
						 pml_width=[40, 40, 40, 40],
						 pml_freq=freq)
		wavefields, receiver_amplitudes = outputs[0], outputs[-1]
		return wavefields, receiver_amplitudes

	@staticmethod
	def plot_receivers(freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx, source_locations,
					   receiver_locations, time_steps, limestone_start, save_path):
		"""Plot wave propagation and receiver data at multiple time steps"""
		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		plt.figure()
		for i, nt in enumerate(time_steps):
			wave_propagation, receiver_data = WAVE.wavefield(freq, nt, dt, peak_time, n_shots, n_sources_per_shot,
															device, vp, dx, source_locations, receiver_locations)
			wave_propagation = wave_propagation[0, :, :].cpu().numpy().T
			receiver_data = receiver_data[0].cpu().numpy().T
			# NOTE Wave Propagation
			plt.subplot(2, 2, 2*i + 1)
			y_max_wp = wave_propagation.shape[0] * dx * 0.001  # Convert y index to kilometers for wave propagation
			x_max_wp = wave_propagation.shape[1] * dx * 0.001  # Convert x index to kilometers for wave propagation
			max_wp, min_wp = WAVE.clip(wave_propagation, 100)
			plt.imshow(wave_propagation, aspect='auto', cmap='gray', origin='upper',
					   extent=[0, x_max_wp, y_max_wp, 0], vmin=min_wp, vmax=max_wp)
			pml_thickness = 40
			limestone_depth_km = (limestone_start + pml_thickness) * dx * 0.001
			plt.axhline(limestone_depth_km, color='salmon', linestyle='--', linewidth=2)
			plt.title(f"Wave Propagation: {nt*0.001:.2f} s")
			plt.xlabel('Distance (km)')
			plt.ylabel('Depth (km)')
			# NOTE Receiver Data
			plt.subplot(2, 2, 2*i + 2)
			nt_seconds = nt * 0.001  # Convert nt to seconds
			max_rd, min_rd = WAVE.clip(receiver_data, 99.5)
			plt.imshow(receiver_data, aspect='auto', cmap='gray', origin='upper',
					   extent=[0, x_max_wp, nt_seconds, 0], vmin=min_rd, vmax=max_rd)
			plt.xlabel('Receiver Position (km)')
			plt.ylabel('Time (sec)')
			plt.title('Receiver')
		plt.subplots_adjust(wspace=0.4, hspace=0.6)
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		plt.show()

	@staticmethod
	def rgba_to_grayscale(arr):
		"""Convert RGBA image to grayscale"""
		return 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

	@staticmethod
	def normalize_data(arr, a=0, b=1):
		"""Normalize array data to range [a, b]"""
		min_val = np.min(arr)
		max_val = np.max(arr)
		return a + (b - a) * (arr - min_val) / (max_val - min_val)

	@staticmethod
	def photo2velocity(img_arr, min_velocity, max_velocity, save_path):
		"""Convert image to velocity model"""
		print("...Creating Velocity Model...")
		img_arr = WAVE.rgba_to_grayscale(img_arr)
		img_arr = WAVE.normalize_data(img_arr, min_velocity, max_velocity)

		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		fig = plt.figure(figsize=(10, 8))
		plt.imshow(img_arr, cmap='rainbow')
		plt.title("Velocity Model")
		cbar = plt.colorbar()
		cbar.set_label('Velocity (m/s)')
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		print("Velocity model saved as:", save_path)
		plt.show()
		return img_arr

	@staticmethod
	def get_wavefield_dtype(vp, dx, dt, freq, nt, dtype, device, source_location):
		"""Generate wavefield using deepwave scalar propagation with dtype parameter"""
		peak_time = 1.5 / freq
		wavefields = scalar(vp, dx, dt,
							source_amplitudes=deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=dtype, device=device),
							source_locations=source_location,
							accuracy=8,
							pml_freq=freq)
		return wavefields

	@staticmethod
	def plot_wave_propagation_dtype(vp, dx, dt, freq, time_steps, dtype, device, source_location, save_path):
		"""Plot wave propagation at multiple time steps with dtype parameter"""
		print("...Time Step Wavefield...")

		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		plt.figure()
		wavefields = [WAVE.get_wavefield_dtype(vp, dx, dt, freq, nt, dtype, device, source_location) for nt in time_steps]
		pml_thickness = 20
		source_y = (source_location[0, 0, 0] + pml_thickness).item()
		source_x = (source_location[0, 0, 1] + pml_thickness).item()
		for idx, (wavefield, nt) in enumerate(zip(wavefields, time_steps), 1):
			plt.subplot(2, 2, idx)
			wave_data = wavefield[0][0, :, :].cpu().numpy()
			max_num, min_num = WAVE.clip(wave_data, 100)
			plt.imshow(wave_data, cmap='gray', vmin=min_num, vmax=max_num)
			plt.scatter(source_x, source_y, c='blue', s=50)
			plt.xlabel('X Distance (m)')
			plt.ylabel('Y Distance (m)')
			plt.title(f"Time Step: {nt} ms")
		plt.subplots_adjust(wspace=0.1, hspace=0.6)
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		print("Wave propagation saved as:", save_path)
		plt.show()

	@staticmethod
	def plot_velocity(img_array, sigma, min_velocity, max_velocity):
		"""Convert image to velocity model with Gaussian smoothing and plot comparison"""
		from scipy.ndimage import gaussian_filter

		print("...Creating Velocity Model...")
		img_array = WAVE.rgba_to_grayscale(img_array)
		img_array = WAVE.normalize_data(img_array, min_velocity, max_velocity)
		original_img_array = img_array.copy()
		img_array = gaussian_filter(img_array, sigma=sigma)

		fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw={'wspace': 0.1})
		img1 = axes[0].imshow(original_img_array, cmap='rainbow')
		axes[0].set_xlabel('Distance-X (pixel)')
		axes[0].set_ylabel('Depth (pixel)')
		axes[0].set_title('Original Velocity Model')
		img2 = axes[1].imshow(img_array, cmap='rainbow')
		axes[1].set_xlabel('Distance-X (pixel)')
		axes[1].set_ylabel('Depth (pixel)')
		axes[1].set_title('Smoothed Velocity Model')
		cbar = fig.colorbar(img2, ax=axes.ravel().tolist(), orientation='horizontal', aspect=50)
		cbar.set_label('velocity (m/s)')
		plt.show()
		return img_array

	@staticmethod
	def loop_wavefield(freq, nt, dt, peak_time, n_sources_per_shot, dtype, device, vp, dx,
					   source_locations, receiver_locations):
		"""Generate wavefield for a single shot in a loop (used for multiple shot acquisitions)"""
		source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=dtype, device=device)
		outputs = scalar(vp, dx, dt,
						 source_amplitudes=source_amplitudes,
						 source_locations=source_locations,
						 receiver_locations=receiver_locations,
						 accuracy=8,
						 pml_width=[40, 40, 40, 40],
						 pml_freq=freq)
		wavefields, receiver_amplitudes = outputs[0], outputs[-1]
		return wavefields, receiver_amplitudes

	@staticmethod
	def run_migration(vp, npy_folder, dtype, device, dx, dt, source_amplitudes, receiver_locations, freq,
					  optimizer_name, lr, loss_fn_name, n_epochs, shot_interval, n_shots, save_path):
		"""Run seismic migration inversion and save result"""
		# Initialize scatter model
		scatter = torch.zeros_like(vp, requires_grad=True)

		# Setup optimizer
		if optimizer_name == 'SGD':
			optimizer = torch.optim.SGD([scatter], lr=lr)
		elif optimizer_name == 'RMSprop':
			optimizer = torch.optim.RMSprop([scatter], lr=lr)
		elif optimizer_name == 'Adagrad':
			optimizer = torch.optim.Adagrad([scatter], lr=lr)
		elif optimizer_name == 'AdamW':
			optimizer = torch.optim.AdamW([scatter], lr=lr)
		else:
			optimizer = torch.optim.Adam([scatter], lr=lr)

		# Setup loss function
		if loss_fn_name == 'CrossEntropyLoss':
			loss_fn = torch.nn.CrossEntropyLoss()
		elif loss_fn_name == 'BCEWithLogitsLoss':
			loss_fn = torch.nn.BCEWithLogitsLoss()
		elif loss_fn_name == 'NLLLoss':
			loss_fn = torch.nn.NLLLoss()
		elif loss_fn_name == 'L1Loss':
			loss_fn = torch.nn.L1Loss()
		else:
			loss_fn = torch.nn.MSELoss()

		# Load observed scatter files
		observed_scatter_files = [os.path.join(npy_folder,
								  f'shot_pixel_{i * shot_interval:04d}.npy') for i in range(n_shots)]
		observed_scatter_masked = [torch.tensor(np.load(f),
								   dtype=dtype,
								   device=device) for f in observed_scatter_files]

		# Run inversion
		for epoch in tqdm(range(n_epochs), desc="Epochs"):
			epoch_loss = 0
			for shot_index in tqdm(range(n_shots), desc="Shots", leave=False):
				current_source_position = shot_index * shot_interval
				source_locations = torch.zeros(1, 1, 2, dtype=dtype, device=device)
				source_locations[0, 0, 0] = current_source_position

				out = scalar_born(
					vp, scatter, dx, dt,
					source_amplitudes=source_amplitudes,
					source_locations=source_locations,
					receiver_locations=receiver_locations,
					pml_freq=freq
				)

				observed_scatter_shot = observed_scatter_masked[shot_index]
				loss = loss_fn(out[-1], observed_scatter_shot)
				epoch_loss += loss.item()
				loss.backward()

			optimizer.step()
			optimizer.zero_grad()
			print(f'Epoch {epoch}, Loss: {epoch_loss}')

		# Save migration result
		scatter_numpy = scatter.detach().cpu().numpy()
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		np.save(save_path.replace('.png', '.npy'), scatter_numpy)
		return scatter_numpy

	@staticmethod
	def plot_migration(migration_data_path, clip_percent, save_path):
		"""Plot migration result"""
		migration_data = np.load(migration_data_path)
		migration_data -= np.mean(migration_data)
		migration_data /= np.max(np.abs(migration_data))
		migration_data_clipped = migration_data[:, 52:]  # Clip top image
		print("Shape of migration_data_clipped:", migration_data_clipped.shape)

		vmax_clip, vmin_clip = WAVE.clip(migration_data_clipped, clip_percent)

		# Create output directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		plt.figure()
		plt.imshow(migration_data_clipped.T, aspect='auto', cmap='gray', vmin=vmin_clip, vmax=vmax_clip)
		plt.xlabel('Distance (pixel)')
		plt.ylabel('Depth (m)')
		plt.title('Migration')
		plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
		print("Migration saved as:", save_path)
		plt.show()
