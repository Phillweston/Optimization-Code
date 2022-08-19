sym(1/3)

% Create Symbolic Variables
syms x
y = sym('y')
syms a b c
A = sym('a', [1 20]) % Create the variables a1, ..., a20
x = sym('x','real');
y = sym('y','positive');
z = sym('z','integer');
t = sym('t','rational');

% Create Symbolic Expressions
phi = (1 + sqrt(sym(5)))/2
syms a b c x
f = a*x^2 + b*x + c
% Create Symbolic Functions
syms x y
f = x^2*y;

dfx = diff(f,x)
g = gradient(f)
H = hessian(f)

% Create Symbolic Matrices
syms a b c
A = [a b c; c a b; b c a]
sum(A(1,:))
A = sym('A', [2 4]) %create the 2-by-4 matrix A
A = sym('A%d%d', [2 4]) %create the 2-by-4 matrix A
a = sym('x_%d', [1 4])

% Use Assumptions on Symbolic Variables
syms x
assume(x >= 0)
assumeAlso(x,'integer')
syms a b real
syms c positive

% Perform Symbolic Computations
syms x
f = sin(x)^2;
diff(f)
diff(f,x,2)

syms x y
f = sin(x)^2 + cos(y)^2;
diff(diff(f, y), x)

syms x
f = sin(x)^2;
int(f)

syms x y n
f = x^n + y^n;
int(f)

% Definite Integrals
syms x y n
f = x^n + y^n;
int(f,x,[1, 10])

syms a x
assume(a >= 0);
F = int(sin(a*x)*sin(x/a),x,-a,a)


% Solve Algebraic Equations
syms x
solve(x^3 - 6*x^2 == 6 - 11*x)

syms x y
solve(6*x^2 - 6*x^2*y + x*y^2 - x*y + y^3 - y^2 == 0, y)
syms x y z
[x, y, z] = solve(z == 4*x, x == y, z == x^2 + y^2)

% Substitutions in Symbolic Expressions
syms x
f = 2*x^2 - 3*x + 1;
double(subs(f,x,1/3))

syms x y
f = x^2*y + 5*x*sqrt(y);
subs(f, x, 3)

% Create Symbolic Expressions from Function Handles
h_expr = @(x)(sin(x) + cos(x));
sym_expr = sym(h_expr)

h_matrix = @(x)(x*pascal(3));
sym_matrix = sym(h_matrix)

% Plot Symbolic Functions
syms x
f = x^3 - 6*x^2 + 11*x - 6;
fplot(f)

syms t
fplot3(t^2*sin(10*t), t^2*cos(10*t), t)

syms x y
fsurf(x^2 + y^2)


clear
syms x f(x)
f(x) = x*exp(-x)*sin(5*x) -2;
xs = 0:1/3:3;
ys = double(subs(f,xs));
fplot(f,[0,3])
hold on
plot(xs,ys,'*k','DisplayName','Data Points')
fplot(@(x) spline(xs,ys,x),[0 3],'DisplayName','Spline interpolant')
hold off
grid on
legend show

clear
syms x y g(x,y)
g(x,y) = x^3-4*x-y^2
fcontour(g,[-3 3 -4 4],'LevelList',-6:6)
title 'Some Elliptic Curves'

% Convert linear equations to matrix form
syms x y z
eqns = [x+y-2*z == 0,
        x+y+z == 1,
        2*y-z == -5];
[A,b] = equationsToMatrix(eqns)


