clc;
clear all;
close all;

% Parameters
L = 1;  % Length of the plate (m)
H = 0.2;  % Height of the domain (m)
Re = 1e4;  % Reynolds number
U_inf = 1;  % Free-stream velocity (m/s)
nu = U_inf * L / Re;  % Kinematic viscosity (m^2/s)
tol = 1e-6;  % Convergence tolerance
max_iter = 10000;  % Maximum iterations

nx = 3000;  % Number of grid points in x-direction
ny = 3000;  % Number of grid points in y-direction
dx = L / (nx - 1);  % Grid spacing in x-direction
dy = H / (ny - 1);  % Grid spacing in y-direction

% Initialize u and v velocity fields
u = U_inf .* ones(ny, nx);  % x-velocity initialized with U_inf
v = zeros(ny, nx);  % y-velocity initialized with zeros

% Boundary Conditions
u(:, 1) = U_inf;  % Free-stream velocity at left boundary
u(1, :) = 0;  % No-slip condition at the plate (bottom boundary)
u(end, :) = U_inf;  % Free-stream velocity at the top boundary

% Iterating over the x-direction (for each column)
for j = 2:nx
    u(:, j) = u(:, j-1);  % Start with the previous column's solution
    % Gauss-Seidel loop for u-velocity convergence
    no_convergence = true;
    iter = 0;
    while no_convergence
        u_old = u(:, j);
        for i = 2:ny-1
            % Discretized u-velocity update (Crank-Nicolson scheme logic)
            u_prev_col = u(i, j-1);  % u(i,j-1)
            u_above = u(i+1, j);     % u(i+1,j)
            u_below = u(i-1, j);     % u(i-1,j)
            
            % Update u using implicit scheme
            u(i, j) = ((u_prev_col^2 / dx) - u_above * (v(i, j-1) / (2 * dy) - nu / (dy^2)) + ...
                       u_below * (v(i, j-1) / (2 * dy) + nu / (dy^2))) / ...
                      (u_prev_col / dx + 2 * nu / (dy^2));
        end
        % Check convergence for u
        residual = sqrt(sum((u(:, j) - u_old).^2));
        iter = iter + 1;
        if (residual < tol || iter >= max_iter)
            no_convergence = false;
        end
    end

    v(:, j) = v(:, j-1);  % Start with the previous column's v
    % Gauss-Seidel loop for v-velocity convergence
    no_convergence = true;
    iter = 0;
    while no_convergence
        v_old = v(:, j);
        for i = 2:ny-1
            % Discretized v-velocity update (from continuity equation)
            v_above = v(i-1, j);  % v(i-1,j)
            % Update v using implicit scheme
            v(i, j) = v_above - (u(i, j) - u(i, j-1)) * dy / dx;
        end
        % Check convergence for v
        residual = sqrt(sum((v(:, j) - v_old).^2));
        iter = iter + 1;
        if (residual < tol || iter >= max_iter)
            no_convergence = false;
        end
    end
end

% Tracking boundary layer thickness where u >= 0.9999 * U_inf
boundary_layer_idx = [0];  % y-indices for boundary layer
boundary_layer_x = [1];  % x-indices for boundary layer
for j = 2:nx
    for i = 2:ny-1
        if u(i, j) >= 0.99 * U_inf
            boundary_layer_idx = [boundary_layer_idx, i];
            boundary_layer_x = [boundary_layer_x, j];
            break;
        end
    end
end

% Creating x and y arrays for plotting
x_arr = linspace(0, L, nx);
y_arr = linspace(0, H, ny);
[X, Y] = meshgrid(x_arr, y_arr);

% Plotting u-velocity contour
figure;
contourf(X, Y, u, 40);
colorbar;
new_boundary_y = (H / ny) .* boundary_layer_idx;  % Normalize boundary layer y-values
new_boundary_x = (L / nx) .* boundary_layer_x;  % Normalize boundary layer x-values
hold on;
plot(new_boundary_x, new_boundary_y, 'r', 'LineWidth', 2);  % Plot boundary layer
title('Contour of u-velocity');
xlabel('x');
ylabel('y');
ylim([0 0.1])
hold off;

% Plotting v-velocity contour
figure;
contourf(linspace(0, L, nx), linspace(0, H, ny), v, 40);
colorbar;
title('Contour of v-velocity');
xlabel('x');
ylabel('y');

% Compute and plot normalized x-velocity (F') as a function of similarity variable (eta)
eta = linspace(0, H * sqrt(U_inf / nu / L), ny);  % Similarity variable
F_prime_numeric = u(:, end) / U_inf;  % Normalized velocity

% Solve the Blasius equation for theoretical comparison
eta_theoretical = linspace(0, 20, 100);
F_prime_theoretical = blasius_solution(eta_theoretical);  % Call Blasius solver

figure;
plot(eta, F_prime_numeric, '-o', 'DisplayName', 'Numerical');
hold on;
plot(eta_theoretical, F_prime_theoretical, '-o', 'DisplayName', 'Theoretical');
title('Normalized x-velocity F''(\eta) vs Similarity Variable \eta');
xlabel('Similarity Variable \eta');
ylabel('Normalized x-velocity F''(\eta)');
xlim([0 10])
legend;
grid on;

% Compute and plot boundary layer thickness (delta)
delta_x = 4.91 * sqrt(nu * linspace(0, L, nx) / U_inf);  % Analytical boundary layer thickness
figure;
plot(linspace(0, L, nx), delta_x, 'b', 'LineWidth', 2, 'DisplayName', 'Theoretical \delta(x)');
hold on;
plot(new_boundary_x, new_boundary_y, 'r--', 'LineWidth', 2, 'DisplayName', 'Numerical \delta(x)');
title('Boundary Layer Thickness \delta(x) vs x');
xlabel('x');
ylabel('Boundary Layer Thickness \delta(x)');
legend;
grid on;

disp('Simulation completed.');

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

