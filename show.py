import numpy as np
import matplotlib.pyplot as plt
import torch

input_tensor = torch.load('data.pt')
ct, bev, mask, dose = input_tensor[:,0].unbind()
wed = torch.cumsum(mask*(ct/1000+1.024), dim=0)

#idx = 120
#img1 = wed[idx].numpy()
#img2 = dose[idx].numpy()
#mask = (bev[idx]==1).numpy()

img1 = torch.load('depth.pth').numpy()
img2 = torch.load('dose.pth').numpy()
mask = torch.load('bev.pth').bool().numpy()


# --- 2. Extract pixel coordinates and values inside the mask ---
# Get 2D indices where mask is True
y_indices, x_indices = np.where(mask)

# Extract the pixel values for the scatter plot
val_img1 = img1[mask]
val_img2 = img2[mask]

# --- 3. Set up the Figure Layout ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax_scatter, ax_img1, ax_img2 = axes

# Plot Image 1 + Mask overlay
ax_img1.imshow(img1, cmap='gray')
ax_img1.imshow(mask, cmap='jet', alpha=0.3)
ax_img1.set_title("Image 1 (with Mask)")

# Plot Image 2 + Mask overlay
ax_img2.imshow(img2, cmap='gray')
ax_img2.imshow(mask, cmap='jet', alpha=0.3)
ax_img2.set_title("Image 2 (with Mask)")

# Plot Scatter
# Use 'picker=True' so matplotlib tracks coordinates on click/hover
scatter = ax_scatter.scatter(val_img1, val_img2, alpha=0.5, s=10, picker=True)
ax_scatter.set_xlabel("Image 1 Pixels")
ax_scatter.set_ylabel("Image 2 Pixels")
ax_scatter.set_title("Scatter Plot (Inside Mask)")
ax_scatter.grid(True)

# Create placeholder visual markers for the highlighted pixel
pixel_marker1, = ax_img1.plot([], [], 'ro', markersize=8)  # Red dot on Image 1
pixel_marker2, = ax_img2.plot([], [], 'ro', markersize=8)  # Red dot on Image 2
scatter_highlight, = ax_scatter.plot([], [], 'ro', markersize=8) # Red dot on Scatter

# --- 4. Interactive Hover/Click Event Handler ---
def on_hover(event):
    # Check if the mouse is over a scatter point
    cont, ind = scatter.contains(event)
    if cont:
        # Get the index of the first point hovered over
        idx = ind['ind'][0]
        
        # Map the scatter index back to the original 2D image coordinates
        pixel_x = x_indices[idx]
        pixel_y = y_indices[idx]
        
        # Get the values
        v1 = val_img1[idx]
        v2 = val_img2[idx]
        
        # Update the positions of the red highlight markers
        pixel_marker1.set_data([pixel_x], [pixel_y])
        pixel_marker2.set_data([pixel_x], [pixel_y])
        scatter_highlight.set_data([v1], [v2])
        
        # Dynamic title update to show coordinates
        ax_scatter.set_title(f"Selected Pixel: X={pixel_x}, Y={pixel_y}")
        
        # Redraw the canvas to show updates immediately
        fig.canvas.draw_idle()

# Connect the hover event to the figure canvas
fig.canvas.mpl_connect('motion_notify_event', on_hover)

plt.tight_layout()
plt.show()

