import fipy

gmsh_text = '''
// Define the square that acts as the system boundary.

dx = %(dx)g;
L = %(L)g;
R = %(R)g;

// Define each corner of the square
// Arguments are (x, y, z, dx); dx is the desired cell size near that point.
Point(1) = {L / 2, L / 2, 0, dx};
Point(2) = {-L / 2, L / 2, 0, dx};
Point(3) = {-L / 2, -L / 2, 0, dx};
Point(4) = {L / 2, -L / 2, 0, dx};

// Line is a straight line between points.
// Arguments are indices of points as defined above.
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};

// Loop is a closed loop of lines.
// Arguments are indices of lines as defined above.
Line Loop(1) = {1, 2, 3, 4};

// Define a circle

// Define the center and compass points of the circle.
Point(5) = {0, 0, 0, dx};
Point(6) = {-R, 0, 0, dx};
Point(7) = {0, R, 0, dx};
Point(8) = {R, 0, 0, dx};
Point(9) = {0, -R, 0, dx};

// Circle is confusingly actually an arc line between points.
// Arguments are indices of: starting point; center of curvature; end point.
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Line Loop(2) = {5, 6, 7, 8};

// The first argument is the outer loop boundary.
// The remainder are holes in it.
 Plane Surface(1) = {1, 2};
//Plane Surface(1) = {1};
'''


def make_circle_mesh(L, dx, R):
    return fipy.Gmsh2D(gmsh_text % {'L': L, 'dx': dx, 'R': R})
