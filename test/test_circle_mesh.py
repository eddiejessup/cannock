from __future__ import print_function, division
import numpy as np
import fipy
import make_circle_mesh

L = 1.0
dx = 0.05
R = 0.1
dt = 0.01
D = 1.0
eq = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=D)
n = 100

mesh = make_circle_mesh.mesh_circle(L, dx, R)
phi = fipy.CellVariable(mesh=mesh, value=np.zeros([mesh.numberOfCells]))

X, Y = mesh.faceCenters
R_f = np.sqrt(X ** 2 + Y ** 2)
V = np.zeros_like(R_f)
V[R_f < 1.2 * R] = 1.0
raw_input()
phi.constrain(V, mesh.exteriorFaces)

viewer = fipy.Matplotlib2DViewer(vars=phi, colorbar=None)
viewer.plotMesh()

for step in range(n):
    eq.solve(var=phi, dt=dt)
    viewer.plot()
raw_input()
