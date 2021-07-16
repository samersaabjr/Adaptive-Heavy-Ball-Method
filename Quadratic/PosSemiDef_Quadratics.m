clc, clear, close all

cnt = 0;

gamma = 1.2;

for n = 2: 100
    
    cnt = cnt + 1;
    
    % Generation of random quadratic functions
    A = zeros(n);
    
    for kk = 1:n/2
        
        yc = rand(n,1);
        
        A = A + yc*yc';
        
    end
    
    % Calculate max eigenvalue of quadratic function to be used for the
    % hyper-parameters of the time-varying and time-invariant HB methods
    L = max(eig(A));
    L_track(cnt) = L;
    
    b = 0*ones(n,1);
    
    % Choose initial conditions for x (same for all methods)
    x = randn(n,1);
    xm = randn(n,1);
    
    xHB = x;
    xHBm = xm;
    
    xHBv = x;
    xHBvm = xm;
    
    % Save previous gradient of cost function for our method
    gm = A*xm+b;
    
    % Difference in current and previous x
    dx = x-xm;
    
    % Hyper-parameters of the time-varying and time-invariant HB methods
    beta = 0.5;
    alpha = 1/L;
    
    % To be used for cesaro average of iterates
    sumx = 0;
    sumHBx = 0;
    sumHBvx = 0;
    
    % Term added into denominators to avoid singularities
    ee=0;
    
    for k = 1:20*n 
        
        % Gradient of our method
        g = A*x+b;
        
        % Difference in current and previous gradient of our method
        dg = g - gm;
        % Our previous gradient
        gm = g;
        
        % Difference in current and previous x for our method
        dx = x - xm;
        % Our update of previous x
        xm = x;
        
        % Our approximation of largest eigenvalue
        Lh = gamma*sqrt((dg'*dg)/(dx'*dx+ee));
        
        dgm = dg - Lh*dx;
        
        % Our approximation of smallest eigenvalue
        lh = sqrt((dgm'*dgm)/(dx'*dx+ee));
        
        % Our approximated Polyak hyper-parameters
        alphah = 4/((sqrt(Lh) + sqrt(lh))^2);
        betah = ((sqrt(Lh) - sqrt(lh))/(sqrt(Lh) + sqrt(lh)))^2;
        
        % Our update
        x = x - alphah*g + betah*dx;
        
        % Our calculation of cesaro average of iterates
        sumx = (sumx + x)/(k+1);
        
        % Our value of f(x)
        fxT(k) = 0.5*sumx'*A*sumx + b'*sumx;
        
        % Our norm of gradient
        normg(k) = norm(g-b);
        
        % Gradient of time-invariant HB
        gHB = A*xHB+b;
        
        % Difference in current and previous x of time-invariant HB
        dxHB = xHB - xHBm;
        % Update previous x of time-invariant HB
        xHBm = xHB;
        
        % Parameter update of time-invariant HB
        xHB = xHB - alpha*gHB + beta*dxHB;
        
        % Calculation of Cesar average od iterates of time-invariant HB
        sumHBx = (sumHBx + xHB)/(k+1);
        
        % f(x) of time-invariant HB
        fxTHB(k) = 0.5*sumHBx'*A*sumHBx + b'*sumHBx;
        
        % Norm of gradient of time-invariant HB
        normgHB(k) = norm(gHB-b);
        
        % Norm of gradient of time-varying HB
        gHBv = A*xHBv+b;
        
        % Hyper-parameters of time-varying HB
        alphav = alpha/(k+2);
        betav = k/(k+2);
        
        % Difference in current and previous x of time-varying HB
        dxHBv = xHBv - xHBvm;
        % Update previous x of time-varying HB
        xHBvm = xHBv;
        
        % Parameter update of time-varying HB
        xHBv = xHBv - alphav*gHBv + betav*dxHBv;
        
        % Calculation of Cesaro average of iterates of time-varying HB
        sumHBvx = (sumHBvx + xHBv)/(k+1);
        
        % f(x) of time-varying HB
        fxTHBv(k) = 0.5*sumHBvx'*A*sumHBvx + b'*sumHBvx;
        
        % Norm of gradient of time-varying HB
        normgHBv(k) = norm(gHBv-b);
        
    end
    
    if n == 50
        
        % Comparison of the progress of the objective values
        % with A for d = 50 evaluated at the Cesaro average of the
        % iterates of the three heavy-ball methods under study.
        
        figure
        semilogy(fxT,'LineWidth', 1.5, 'MarkerSize', 10),hold
        semilogy(fxTHBv,'k--','LineWidth', 1.5, 'MarkerSize', 10),
        semilogy(fxTHB,'k','LineWidth', 1.5, 'MarkerSize', 10),
        semilogy(100./[1:k],'r--'),grid,
        ylabel({'$f(\bar{x}_T) - f(x^*)$'},'Interpreter','latex'),
        legend({'AHB','HB: time-varying','HB','$\mathcal{O}(1/k)$'},'Interpreter','latex')
        xlabel('Iterations, $k$','Interpreter','latex')
        set(gca,'Fontsize',12);
        
    end
    
    N(cnt) = n;
    HBf(cnt) = abs(fxTHB(k));
    HBfv(cnt) = abs(fxTHBv(k));
    AHBf(cnt) = abs(fxT(k));
    HBg(cnt) =  normgHB(k);
    AHBg(cnt) = normg(k);
    
end

% Comparison of the objective values with A for d = 2, ..., 100 evaluated at 
% the Cesaro average at the iterate k = 20d of the three heavy-ball methods 
% under study.

figure
semilogy(N,AHBf,'LineWidth', 1.5, 'MarkerSize', 10),hold,
semilogy(N,HBfv,'k--','LineWidth', 1.5, 'MarkerSize', 10),
semilogy(N,HBf,'k','LineWidth', 1.5, 'MarkerSize', 10),grid,
ylabel({'$f(\bar{x}_T) - f(x^*)$'},'Interpreter','latex'),
legend('AHB','HB: time-varying','HB'),
xlabel('Dimenssion of $A, d$','Interpreter','latex')
set(gca,'Fontsize',12);

%% Performance of AHB with varying gamma

% Comparison of the objective values with A for d = 50 evaluated at the
% Cesaro average at the iterate k = 1,000 of the proposed adaptive heavy-ball 
% method for different values of gamma.

randn('seed',1)
rand('seed',1)

n = 50;

cnt = 0;

gammas = 1: 0.1: 2;

for gamma = gammas
    
    cnt = cnt + 1;
    
    % Generation of random quadratic functions
    A = zeros(n);
    
    for kk = 1:n/2
        
        yc = rand(n,1);
        
        A = A + yc*yc';
        
    end
    
    b = 0*ones(n,1);
    
    % Choose initial conditions for x
    x = randn(n,1);
    xm = randn(n,1);
    
    % Save previous gradient of cost function
    gm = A*xm+b;
    
    % Difference in current and previous x
    dx = x-xm;
    
    % To be used for cesaro average of iterates
    sumx = 0;
    
    % Term added into denominators to avoid singularities
    ee=0;
    
    for k = 1:20*n 
        
        % Calculate gradient
        g = A*x+b;
        
        % Difference in current and previous gradient
        dg = g - gm;
        % Update of previous gradient
        gm = g;
        
        % Difference in current and previous x
        dx = x - xm;
        % Update of previous x
        xm = x;
        
        % Approximation of largest eigenvalue
        Lh = gamma*sqrt((dg'*dg)/(dx'*dx+ee));
        
        dgm = dg - Lh*dx;
        
        % Approximation of smallest eigenvalue
        lh = sqrt((dgm'*dgm)/(dx'*dx+ee));
        
        % Approximated Polyak hyper-parameters
        alphah = 4/((sqrt(Lh) + sqrt(lh))^2);
        betah = ((sqrt(Lh) - sqrt(lh))/(sqrt(Lh) + sqrt(lh)))^2;
        
        % Update model parameters
        x = x - alphah*g + betah*dx;
        
        % Calculation of cesaro average of iterates
        sumx = (sumx + x)/(k+1);
        
        % calculate f(x)
        fxT(k) = 0.5*sumx'*A*sumx + b'*sumx;
        
        % Norm of gradient
        normg(k) = norm(g-b);
        
    end
    
    AHBf_gammas(cnt) = abs(fxT(k));
    
end

figure
semilogy(gammas,AHBf_gammas,'LineWidth', 1.5, 'MarkerSize', 10),hold,
ylabel({'$f(\bar{x}_T) - f(x^*)$'},'Interpreter','latex'),
legend('AHB','HB: time-varying','HB'),
xlabel('{$\gamma$}','Interpreter','latex')
set(gca,'Fontsize',12);
grid
