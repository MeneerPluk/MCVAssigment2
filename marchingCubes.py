import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import voxels as v
from skimage import measure
from skimage.draw import ellipsoid

# Generate voxel coords of frame:
coords = v.initilizeVoxels()

# turing it into a 3d array one hot encoding:
voxelGrid = np.zeros((50,50,100))  
for cord in coords:
    x,y,z = cord[0],-cord[2],cord[1]
    voxelGrid[x,y,z] = 1


# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(voxelGrid, 0)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: 2cm per unit")
ax.set_ylabel("y-axis: 2cm per unit")
ax.set_zlabel("z-axis: 2cm per unit")

ax.set_xlim(0, 65)
ax.set_ylim(0, 65)
ax.set_zlim(0, 65)

plt.tight_layout()
plt.show()