clc;
clear all;
close all;

% Parameters
L = 1;
H = 0.1;
Re = 1e4;
U_inf = 1;
nu = U_inf * L / Re;
tol = 1e-6;
max_iter = 1000;
nx = 3000;
ny = 3000;
dx = L / (nx - 1);
dy = H / (ny - 1);

% Initialize u and v velocity fields
u = U_inf.*ones(ny, nx);
v = zeros(ny, nx);

% Boundary Conditions
u(:, 1) = U_inf;  % Left boundary: u = U_inf
u(1,:) = 0;       % Bottom boundary: no-slip condition
u(end, :) = U_inf; % Top boundary: free-stream velocity

% Marching in x-direction
for j = 2:nx
    u(:,j) = u(:,j-1);  % Initial guess for u at the new x-station
    no_convergence = true;
    iter = 0;
    while no_convergence
        u_old = u(:,j);
        for i = 2:ny-1
            u_e = u(i,j-1);
            u_c = u(i+1,j);
            u_n = u(i-1,j);
            u_b = u(i+1,j-1);
            u_t = u(i-1,j-1);

            v_e = v(i, j-1);
            
            % Update u using Crank-Nicolson scheme
            u(i,j) = (u_e^2 / dx - v_e * (u_c - u_n + u_b - u_t) / (4 * dy) ...
                     + nu * (u_c + u_n + u_b - 2 * u_e + u_t) / (2 * dy^2)) ...
                     / (u_e / dx + nu / dy^2);
        end
        
        residual = sqrt(sum((u(:,j) - u_old).^2));
        iter = iter + 1;
        if residual < tol || iter >= max_iter
            no_convergence = false;
        end
    end
    
    % Update v using the continuity equation
    for i = 2:ny-1
        v(i,j) = v(i-1,j) - (u(i,j) - u(i,j-1)) * dy / dx;
    end
end

% Boundary Layer Thickness Calculation
boundary_layer_idx = [];
x_bl_coords = [];
for j = 2:nx
    for i = 2:ny-1
        if u(i,j) >= 0.99 * U_inf
            boundary_layer_idx = [boundary_layer_idx, i];
            x_bl_coords = [x_bl_coords, j];
            break;
        end
    end
end

x_arr = linspace(0, L, nx);
y_arr = linspace(0, H, ny);
[X,Y] = meshgrid(x_arr,y_arr);

% Plot u-velocity contour
figure;
contourf(X,Y, u, 40);
colorbar;
hold on;
bl_thickness_y = (H / ny) * boundary_layer_idx;
bl_thickness_x = (L / nx) * x_bl_coords;
plot(bl_thickness_x, bl_thickness_y, 'k--', 'LineWidth', 2);
title('Contour of u-velocity with boundary layer thickness');
xlabel('x');
ylabel('y');

% Plot v-velocity contour
figure;
contourf(X,Y, v, 20);
colorbar;
title('Contour of v-velocity');
xlabel('x');
ylabel('y');

% Compute and plot normalized x-velocity (F') as a function of similarity variable (eta)
eta = linspace(0, H * sqrt(U_inf / (nu * L)), ny);
F_prime_numeric = u(:, end) / U_inf;

% Solve the Blasius equation to get the theoretical F'(eta)
eta_theoretical = linspace(0, 6, 100);
F_prime_theoretical = blasius_solution(eta_theoretical);  % Call Blasius solver

figure;
plot(eta, F_prime_numeric, '-o', 'DisplayName', 'Numerical');
hold on;
plot(eta_theoretical, F_prime_theoretical, '-r', 'DisplayName', 'Theoretical');
title('Normalized x-velocity F''(\eta) vs Similarity Variable \eta');
xlabel('Similarity Variable \eta');
ylabel('Normalized x-velocity F''(\eta)');
legend;
grid on;

% Compute and plot boundary layer thickness (delta)
delta_x = 4.91 * sqrt(nu * linspace(0, L, nx) / U_inf);
figure;
plot(linspace(0, L, nx), delta_x, 'r', 'DisplayName', 'Theoretical \delta(x)');
hold on;
plot(bl_thickness_x, bl_thickness_y, 'b--', 'DisplayName', 'Numerical \delta(x)');
title('Boundary Layer Thickness \delta(x) vs x');
xlabel('x');
ylabel('Boundary Layer Thickness \delta(x)');
legend;
grid on;

disp('Calculation complete.');

% Solve the Blasius ODE using ode45 to get the theoretical F'(eta)
function F_prime = blasius_solution(eta)
    % Blasius equation: 2*f''' + f*f'' = 0
    % Boundary conditions: f(0) = 0, f'(0) = 0, f'(\infty) = 1
    % We'll solve it numerically using ode45

    % Initial conditions [f(0), f'(0), f''(0)] = [0, 0, some value (f''(0) is guessed)]
    f0 = [0, 0, 0.33206];  % f''(0) = 0.33206 is the known value for the Blasius equation
    
    % Solve the ODE using ode45
    [~, F] = ode45(@blasius_ode, eta, f0);
    
    % F'(eta) is the second column of the result
    F_prime = F(:, 2);
end

% Blasius ODE function (system of first-order ODEs)
function dF = blasius_ode(eta, F)
    % F(1) = f, F(2) = f', F(3) = f''
    dF = zeros(3,1);
    dF(1) = F(2);  % f' = F(2)
    dF(2) = F(3);  % f'' = F(3)
    dF(3) = -0.5 * F(1) * F(3);  % f''' = -0.5*f*f''
end

