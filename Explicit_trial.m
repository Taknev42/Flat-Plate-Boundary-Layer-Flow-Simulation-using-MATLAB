clc;
clear all;
close all;

% Parameters
L = 1;  % Length of the plate (m)
H = 0.2;  % Height of the domain (m)
Re = 1e4;  % Reynolds number
U_inf = 1;  % Free-stream velocity (m/s)
nu = U_inf * L / Re;  % Kinematic viscosity (m^2/s)

% Grid parameters
nx = 10000;  % Number of grid points in the x-direction
ny = 300;  % Number of grid points in the y-direction
dx = L / (nx - 1);  % Grid spacing in x-direction
dy = H / (ny - 1);  % Grid spacing in y-direction

% Initialize u and v velocity fields (zero at the start)
u = zeros(ny, nx);  % x-velocity
v = zeros(ny, nx);  % y-velocity

% Boundary Conditions
u(:, 1) = U_inf;  % Free-stream velocity at the left boundary
u(end, :) = U_inf;  % Free-stream velocity at the top boundary

% Arrays to track boundary layer thickness
boundary_layer_idx = [0];  % Grid index where u = 0.99*U_inf
boundary_layer_x = [1];  % Corresponding x-grid index

% Loop through the grid and update u and v velocities (explicit method)
for j = 2:nx
    flag = false;
    for i = 2:ny-1
        % Discretize x-momentum equation (solve for u at the next point)
        u_next = u(i+1, j-1);  % u(i+1,j-1)
        u_current = u(i, j-1);  % u(i,j-1)
        u_prev = u(i-1, j-1);  % u(i-1,j-1)

        % Update u using the explicit method
        u(i, j) = u_current + nu * (u_next - 2 * u_current + u_prev) / u_current * dx / (dy^2) ...
                   - (u_next - u_prev) / 2 * v(i, j-1) / u_current * dx / dy;

        % Check if u reaches 99% of free-stream velocity
        if ~flag && u(i,j) >= 0.99 * U_inf
            flag = true;
            boundary_layer_idx = [boundary_layer_idx, i];
            boundary_layer_x = [boundary_layer_x, j];
        end

        % Continuity equation to solve for v
        v_prev = v(i-1, j);  % v(i-1,j)
        u_diff_x = (u(i,j) - u(i,j-1) + u(i-1,j) - u(i-1,j-1));  % Difference in u

        % Update v at the next point
        v(i, j) = v_prev - 0.5 * u_diff_x * dy / dx;
    end
end

% x and y grid arrays for plotting
x_arr = linspace(0, L, nx);
y_arr = linspace(0, H, ny);
[X,Y] = meshgrid(x_arr,y_arr);

% Plotting u-velocity contour
figure;
contourf(X, Y, u, 40);
colorbar;
hold on;

% Plot boundary layer thickness (where u = 0.99*U_inf)
boundary_layer_y = (H / ny) .* boundary_layer_idx;
boundary_layer_x_phys = (L / nx) .* boundary_layer_x;
plot(boundary_layer_x_phys, boundary_layer_y, 'r', 'LineWidth', 2);
title('Contour of u-velocity');
xlabel('x');
ylabel('y');
ylim([0 0.1])
hold off;

% Plot v-velocity contour
figure;
contourf(X, Y, v, 50);
colorbar;
title('Contour of v-velocity');
xlabel('x');
ylabel('y');

% Compute and plot normalized x-velocity (F') as a function of similarity variable (eta)
eta = linspace(0, H * sqrt(U_inf / nu / L), ny);
F_prime_numeric = u(:, end) / U_inf;

% Solve the Blasius equation for theoretical comparison
eta_theoretical = linspace(0, 6, 100);
F_prime_theoretical = blasius_solution(eta_theoretical);  % Call Blasius solver

figure;
plot(eta, F_prime_numeric, '.', 'DisplayName', 'Numerical');
hold on;
plot(eta_theoretical, F_prime_theoretical, '-o', 'DisplayName', 'Theoretical');
title('Normalized x-velocity F''(\eta) vs Similarity Variable \eta');
xlabel('Similarity Variable \eta');
ylabel('Normalized x-velocity F''(\eta)');
xlim([0 10])
legend;
grid on;

% Compute and plot boundary layer thickness (delta)
delta_x = 4.91 * sqrt(nu * linspace(0, L, nx) / U_inf);
figure;
plot(linspace(0, L, nx), delta_x, 'b', 'LineWidth', 2, 'DisplayName', 'Theoretical \delta(x)');
hold on;
plot(boundary_layer_x_phys, boundary_layer_y, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical \delta(x)');
title('Boundary Layer Thickness \delta(x) vs x');
xlabel('x');
ylabel('Boundary Layer Thickness \delta(x)');
legend;
grid on;

disp('Simulation complete.');

% Blasius solution for theoretical F'(eta)
function F_prime = blasius_solution(eta)
    % Initial conditions [f(0), f'(0), f''(0)] = [0, 0, 0.33206]
    f0 = [0, 0, 0.33206];
    
    % Solve the Blasius ODE using ode45
    [~, F] = ode45(@blasius_ode, eta, f0);
    
    % F'(eta) corresponds to the second column of F
    F_prime = F(:, 2);
end

% Blasius ODE
function dF = blasius_ode(eta, F)
    dF = zeros(3,1);
    dF(1) = F(2);  % f' = F(2)
    dF(2) = F(3);  % f'' = F(3)
    dF(3) = -0.5 * F(1) * F(3);  % f''' = -0.5*f*f''
end


