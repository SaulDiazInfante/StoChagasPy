"""
    This class solves numerically the model for Chagas disease infection
    considering two groups of animals and vectors.
    \begin{align}
        dx_1(t) &=
            \left(
                \lambda -\delta x_1(t) - (1 - \gamma) \beta x_1(t) x_3(t)
            \right)dt
                -\sigma_1 x_1(t) dB_1(t), \notag \\
        dx_2(t) &=
            \left(
                    (1- \gamma) \beta x_1(t) x_3(t) - a x_2(t)
            \right)dt
                -\sigma_1 x_2(t) dB_1(t), 	\\
        dx_3(t) & =
            \left(
                (1 - \eta) N a x_2(t)
                    -ux_3(t)
                    -(1 - \gamma ) \beta x_1(t) x_3(t)
            \right)dt
            - \sigma_2 x_3(t) dB_2(t) \notag
        dx_4(t)
        dx_5(t)
        dx_6(t)
        dx_7(t)
    \end{align}
"""
import yaml
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint


class StochasticChagasDynamics:
    """
        Set the parameters and terms of the Stochastic Duffin Van der Pol
        equation.
    """

    def __init__(self):
        self.H_0 = 500
        self.A_1 = 700
        self.A_2 = 800
        self.K_1 = 42000
        self.K_2 = 120000
        # Set Dimensional parameters
        self.mu_H = 0.000039139
        self.mu_1 = 0.00045662
        self.mu_2 = 0.0011
        self.z_1 = 0.04
        self.z_1_tilde = 0.1
        self.z_2_tilde = 0.11
        self.pi_h_tilde = 0.0009
        self.pi_h = 0.0009
        self.pi_V = 0.03
        self.pi_V_tilde = 0.49
        self.a_1 = 0.7714
        self.a_2 = 0.7
        self.e_1 = 0.0067
        self.e_2 = 0.007
        # Dimensionless parameters
        self.alpha_1 = 1.0
        self.alpha_2 = 1.0
        self.alpha_1_tilde = 1.0
        self.alpha_2_tilde = 1.0
        self.r_1 = 1.0
        self.r_2 = 1.0
        self.rr = 1.0
        self.e_1_tilde = 1.0
        self.e_2_tilde = 1.0
        self.omega_1 = 1.0
        self.omega_2 = 1.0
        self.eta_1 = 1.0
        self.delta_1 = 1.0
        self.delta_2 = 1.0
        self.mu_1_tilde = 1.0
        self.mu_2_tilde = 1.0
        self.sigma_H = 1.0
        self.sigma_1 = 1.0
        self.sigma_1_tilde = 1.0
        self.sigma_2_tilde = 1.0
        self.X = np.zeros([7, 1])
        # Initial Conditions
        self.initial_conditions = np.random.rand(7)
        # Stencil Parameters
        self.k = 3
        self.p = 2
        self.r = 1
        self.T0 = 0.0
        self.N = 10.0 ** self.k
        self.P = 10.0 ** self.p
        self.R = 10.0 ** self.r
        self.N = 2.0 ** self.k
        self.P = 2.0 ** self.p
        self.R = 2.0 ** self.r
        self.L = self.N / self.R
        self.T = 10000.0
        self.t = np.linspace(0, 1, 1000)
        self.dt = self.T / np.float(self.N)
        self.IndexN = np.arange(self.N + 1)
        #
        # set of index to Ito integral
        self.tau = self.IndexN[0:np.int(self.N) + 1:np.int(self.P)]
        self.t = np.linspace(0, self.T, self.N + 1)
        #
        self.Dt = np.float(self.R) * self.dt
        self.L = self.N / self.R
        # diffusion part
        self.DistNormal1 = np.random.randn(np.int(self.N))
        self.DistNormal2 = np.random.randn(np.int(self.N))
        #
        self.dW1 = np.sqrt(self.dt) * self.DistNormal1
        self.dW2 = np.sqrt(self.dt) * self.DistNormal2
        self.x_stkm = np.zeros([np.int(self.L)+ 1, 7])
        self.W1 = np.cumsum(self.dW1)
        self.W1 = np.concatenate(([0], self.W1))
        self.W2 = np.cumsum(self.dW2)
        self.W2 = np.concatenate(([0], self.W2))

    def initialize_mesh(self, k, p, r, T0=0.0, T=1.0):
        """
            Set stencil parameters
        """
        # Stencil of the mesh
        self.k = k
        self.p = p
        self.r = r
        self.T0 = T0
        self.N = 10.0 ** k
        self.P = 10.0 ** p
        self.R = 10.0 ** r
        """
        self.N = 2.0 ** k
        self.P = 2.0 ** p
        self.R = 2.0 ** r
        """
        self.L = self.N / self.R
        self.T = T
        #
        self.dt = self.T / np.float(self.N)
        self.IndexN = np.arange(self.N + 1)

        # set of index to Ito integral
        self.tau = self.IndexN[0:np.int(self.N) + 1:np.int(self.P)]
        self.t = np.linspace(0, self.T, self.N + 1)
        #
        self.Dt = np.float(self.R) * self.dt
        self.L = self.N / self.R
        # diffusion part
        self.DistNormal1 = np.random.randn(np.int(self.N))
        self.DistNormal2 = np.random.randn(np.int(self.N))
        #
        self.dW1 = np.sqrt(self.dt) * self.DistNormal1
        self.dW2 = np.sqrt(self.dt) * self.DistNormal2
        # self.dZ = 0.5 * (self.dt ** 1.5) * (self.DistNormal1 + (3.0 ** (
        # -.5)) * self.DistNormal2)
        # self.Z = np.cumsum(self.dZ)
        # self.Z = np.concatenate(([0], self.Z))
        self.W1 = np.cumsum(self.dW1)
        self.W1 = np.concatenate(([0], self.W1))
        self.W2 = np.cumsum(self.dW2)
        self.W2 = np.concatenate(([0], self.W2))

    def set_dimensional_parameters(self,
                                   file_name='dimensional_parameters.txt'):
        """
            Set dimensional parameters of the Chagas Model.
        """
        with open(file_name, 'r') as f:
            parameter_data = yaml.load(f)
        # Set initial conditions
        self.H_0 = parameter_data.get('H_0')
        self.A_1 = parameter_data.get('A_1')
        self.A_2 = parameter_data.get('A_2')
        self.K_1 = parameter_data.get('K_1')
        self.K_2 = parameter_data.get('K_2')
        # Set Dimensional parameters
        self.mu_H = parameter_data.get('mu_H')
        self.mu_1 = parameter_data.get('mu_1')
        self.mu_2 = parameter_data.get('mu_2')
        self.z_1 = parameter_data.get('z_1')
        self.z_1_tilde = parameter_data.get('z_1_tilde')
        self.z_2_tilde = parameter_data.get('z_2_tilde')
        self.pi_h_tilde = parameter_data.get('pi_h_tilde')
        self.pi_h = parameter_data.get('pi_h')
        self.pi_V = parameter_data.get('pi_V')
        self.pi_V_tilde = parameter_data.get('pi_V_tilde')
        self.a_1 = parameter_data.get('a_1')
        self.a_2 = parameter_data.get('a_2')
        self.e_1 = parameter_data.get('e_1')
        self.e_2 = parameter_data.get('e_2')
        self.omega_1 = parameter_data.get('omega_1')
        self.omega_2 = parameter_data.get('omega_2')
        self.eta_1 = parameter_data.get('eta_1')
        self.initial_conditions = parameter_data.get('initial_conditions')
        # Initial conditions

    def set_dimensionless_parameters(self):
        self.r_1 = self.a_1 - self.e_1
        self.r_2 = self.a_2 - self.e_2
        self.rr = self.r_2 / self.r_1
        self.e_1_tilde = self.e_1 / self.r_1
        self.e_2_tilde = self.e_2 / self.r_1
        #
        # alpha, alpha_tilde
        #
        self.alpha_1 = (self.z_1 * self.pi_h_tilde * self.K_1) / \
                       (self.r_1 * self.H_0)
        #
        num_alpha_2 = self.z_2_tilde * self.pi_h * self.K_2
        den_alpha_2 = self.r_1 * (1.0 - self.eta_1) * self.omega_1 * self.A_1 \
                      + self.r_1 * self.A_2
        self.alpha_2 = num_alpha_2 / den_alpha_2
        #
        num_alpha_1_tilde = self.z_1_tilde * self.pi_h * self.eta_1 * \
                            self.omega_1 \
                            + self.z_2_tilde * self.pi_h \
                              * (1.0 - self.omega_1) * self.omega_2
        dem_alpha_1_tilde = (1.0 - self.omega_1) * self.A_1 \
                            + self.eta_1 * self.omega_1 * self.A_1
        self.alpha_1_tilde = (self.K_1 / self.r_1) \
                             * num_alpha_1_tilde / dem_alpha_1_tilde
        #
        num_alpha_2_tilde = self.z_1_tilde * self.pi_h * (1.0 - self.eta_1) * \
                            self.omega_1
        den_alpha_2_tilde = (1.0 - self.eta_1) * self.omega_1 * self.A_1 \
                            + self.A_2
        self.alpha_2_tilde = (self.K_2 / self.r_1) \
                             * (num_alpha_2_tilde / den_alpha_2_tilde)
        # delta_i
        self.delta_1 = self.a_1 / self.r_1
        self.delta_2 = self.a_2 / self.r_1
        # mu_i_tilde
        self.mu_1_tilde = self.mu_1 / self.r_1
        self.mu_2_tilde = self.mu_2 / self.r_1
        # sigma
        self.sigma_H = self.z_1 * self.pi_V / self.r_1
        num_sigma_1 = self.z_1_tilde * self.pi_V_tilde \
                      * self.eta_1 * self.omega_1 \
                      + self.z_2_tilde * self.pi_V_tilde * (1.0 - self.omega_1)
        #
        den_sigma_1 = self.r_1 * (1.0 - self.omega_1) \
                      + self.eta_1 * self.omega_1 * self.r_1
        #
        self.sigma_1 = num_sigma_1 / den_sigma_1
        #
        num_sigma_1_tilde = (1.0 - self.eta_1) * self.omega_1 * self.A_1
        den_sigma_1_tilde = num_sigma_1_tilde + self.A_2
        self.sigma_1_tilde = (self.z_1_tilde * self.pi_V_tilde) / self.r_1 \
                             * num_sigma_1_tilde / den_sigma_1_tilde

        self.sigma_2_tilde = (self.z_2_tilde * self.pi_V_tilde) / self.r_1 * \
                             self.A_2 / den_sigma_1_tilde

    def a(self, x, t0):
        """"
            The drifft term of the SDE.
        """
        dummy = 0.0 * t0
        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]
        x_4 = x[3]
        x_5 = x[4]
        x_6 = x[5]
        x_7 = x[6]
        # Load parameters.
        e_1_tilde = self.e_1_tilde
        e_2_tilde = self.e_2_tilde
        r = self.rr
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        alpha_1_tilde = self.alpha_1_tilde
        alpha_2_tilde = self.alpha_2_tilde
        mu_H_tilde = self.mu_H / self.r_1
        mu_1_tilde = self.mu_1_tilde
        mu_2_tilde = self.mu_2_tilde
        sigma_H = self.sigma_H
        sigma_1 = self.sigma_1
        sigma_1_tilde = self.sigma_1_tilde
        sigma_2_tilde = self.sigma_2_tilde
        # Right hand side of model.
        f_1 = alpha_1 * x_3 * (1.0 - x_1) - mu_H_tilde * x_1
        f_2 = (alpha_1_tilde * x_3 + alpha_2_tilde * x_6) * (1.0 - x_2) - \
              mu_1_tilde * x_2
        f_3 = (sigma_H * x_1 + sigma_1 * x_2) * (x_4 - x_3) \
              - (e_1_tilde + x_4) * x_3
        f_4 = x_4 * (1 - x_4)
        f_5 = alpha_2 * x_6 * (1.0 - x_5) - mu_2_tilde * x_5
        f_6 = (sigma_1_tilde * x_2 + sigma_2_tilde * x_5) * (x_7 - x_6) \
              - (e_2_tilde + r * x_7) * x_6
        f_7 = r * x_7 * (1.0 - x_7)
        r = np.array([f_1, f_2, f_3, f_4, f_5, f_6, f_7])
        return r

    def b(self, X):
        """
            The diffusion term.
        """
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        x = X[0]
        y = X[1]
        z = X[2]
        B = np.zeros([3, 3])
        x1 = -sigma1 * x
        x2 = -sigma1 * y
        x3 = -sigma2 * z
        B[0, 0] = x1
        B[1, 1] = x2
        B[2, 2] = x3
        return B


class NumericsStochasticChagasDynamics(StochasticChagasDynamics):
    def deterministic_integration(self):
        """
        This function integrats the model via scipy.odeint
        :return: the solution of the chagas infection model.
        """
        y0 = self.initial_conditions
        t = self.t
        sol = odeint(self.a, y0, t)
        return sol

    def linear_steklov(self):
        h = self.Dt
        L = self.L
        R = self.R
        self.x_stkm = np.zeros([np.int(self.L) + 1, 7])
        # model parameters
        e_1_tilde = self.e_1_tilde
        e_2_tilde = self.e_2_tilde
        r = self.rr
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        alpha_1_tilde = self.alpha_1_tilde
        alpha_2_tilde = self.alpha_2_tilde
        mu_H_tilde = self.mu_H / self.r_1
        mu_1_tilde = self.mu_1_tilde
        mu_2_tilde = self.mu_2_tilde
        sigma_H = self.sigma_H
        sigma_1 = self.sigma_1
        sigma_1_tilde = self.sigma_1_tilde
        sigma_2_tilde = self.sigma_2_tilde
        # a_j finctions for the LS-method.

        def a_j(x):
            a1 = - (alpha_1 * x[2] + mu_H_tilde)
            a2 = - (alpha_1_tilde * x[2] + alpha_2_tilde * x[5] + mu_1_tilde)
            a3 = - (sigma_H * x[0] + sigma_1 * x[1] + x[3] + e_1_tilde)
            a4 = 1.0 - x[3]
            a5 = - (alpha_2 * x[5] + mu_2_tilde)
            a6 = - (sigma_1_tilde * x[1]
                    + sigma_2_tilde * x[4] + r * x[6] + e_2_tilde)
            a7 = r * (1.0 - x[6])
            a = np.array([a1, a2, a3, a4, a5, a6, a7])
            return a

        def phi_j(a):
            phi = []
            for aj in a:
                phij = h
                if not(aj == 0.0):
                    phij = (np.exp(h * aj) - 1.0)/aj
                phi.append(phij)
            return np.array(phi)

        def b_j(x):
            b1 = alpha_1 * x[2]
            b2 = alpha_1_tilde * x[2] + alpha_2_tilde * x[5]
            b3 = (sigma_H * x[0] + sigma_1 * x[1]) * x[3]
            b4 = 0.0
            b5 = alpha_2 * x[5]
            b6 = (sigma_1_tilde * x[1] + sigma_2_tilde * x[4]) * x[6]
            b7 = 0.0
            return np.array([b1, b2, b3, b4, b5, b6, b7])

        self.x_stkm[0] = self.initial_conditions
        for j in np.arange(np.int(L)):
            # self.Winc1 = np.sum(self.dW1[R * (j):R * (j + 1)])
            # self.Winc2 = np.sum(self.dW2[R * (j):R * (j + 1)])
            # self.Winc = np.array([[self.Winc1], [self.Winc1], [self.Winc2]])
            xj = self.x_stkm[j, :]
            aj = a_j(xj)
            bj = b_j(xj)
            phij = phi_j(aj)
            A1 = np.diag(np.exp(h * aj))
            A2 = np.diag(phij)
            self.x_stkm[j+1, :] = np.dot(A1, xj) + np.dot(A2, bj)
        xstkm = self.x_stkm
        return xstkm

    def save_data(self):
        """"
            Method to save numerical solutions and parameters of model
        """
        #
        # t=self.t[0:-1:self.R].reshape([self.t[0:-1:self.R].shape[0],1])

        def deterministic_data():
            t = self.dt * self.tau
            Ueem1 = self.Xeem[:, 0]
            Ueem2 = self.Xeem[:, 1]
            Ueem3 = self.Xeem[:, 2]
            Uem1 = self.Xem[:, 0]
            Uem2 = self.Xem[:, 1]
            Uem3 = self.Xem[:, 2]
            Ustk1 = self.Xstkm[:, 0]
            Ustk2 = self.Xstkm[:, 1]
            Ustk3 = self.Xstkm[:, 2]
            tagPar = np.array([
                'k = ',
                'r = ',
                'T0 = ',
                'N = ',
                'R = ',
                'T = ',
                'dt = ',
                'Dt = ',
                'L = ',
                'gamma = ',
                'eta = ',
                'lambda =',
                'delta = ',
                'beta = ',
                'a = ',
                'N0 = ',
                'u = ',
                'sigma1 = ',
                'sigma2 = ',
                'x01 = ',
                'x02 = ',
                'x03 = ',
            ])
            ParValues = np.array([
                self.k,
                self.r,
                self.T0,
                self.N,
                self.R,
                self.T,
                self.dt,
                self.Dt,
                self.L,
                self.gamma,
                self.eta,
                self.Lambda,
                self.delta,
                self.beta,
                self.a,
                self.NN,
                self.u,
                self.sigma1,
                self.sigma2,
                self.Xzero[0, 0],
                self.Xzero[0, 1],
                self.Xzero[0, 2]
            ])
            strPrefix = str(self.Dt)
            name1 = 'DetParameters' + strPrefix + '.txt'
            name2 = 'DetSolution' + strPrefix + '.txt'
            name3 = 'DetRefSolution' + str(self.dt) + '.txt'

            PARAMETERS = np.column_stack((tagPar, ParValues))
            np.savetxt(name1, PARAMETERS, delimiter=" ", fmt="%s")
            np.savetxt(name2,
                       np.transpose(
                           (
                               t, Uem1, Uem2, Uem3, Ustk1, Ustk2, Ustk3,
                           )
                       ), fmt='%1.8f', delimiter='\t')
            np.savetxt(name3,
                       np.transpose(
                           (
                               self.t, Ueem1, Ueem2, Ueem3,
                           )
                       ), fmt='%1.8f', delimiter='\t')

        def stochastic_data():

            """
            t = self.dt * self.tau
            Ueem1 = self.Xeem[:, 0]
            Ueem2 = self.Xeem[:, 1]
            Ueem3 = self.Xeem[:, 2]
            Uem1 = self.Xem[:, 0]
            Uem2 = self.Xem[:, 1]
            Uem3 = self.Xem[:, 2]
            Ustk1 = self.Xstkm[:, 0]
            Ustk2  = self.Xstkm[:, 1]
            Ustk3 = self.Xstkm[:, 2]
            Utem1 = self.Xtem[:, 0]
            Utem2  = self.Xtem[:, 1]
            Utem3 = self.Xtem[:, 2]
            """

            tagPar = np.array([
                'k = ',
                'r = ',
                'T0 = ',
                'N = ',
                'R = ',
                'T = ',
                'dt = ',
                'Dt = ',
                'L = ',
                'gamma = ',
                'eta = ',
                'lambda =',
                'delta = ',
                'beta = ',
                'a = ',
                'N0 = ',
                'u = ',
                'sigma1 = ',
                'sigma2 = ',
                'x01 = ',
                'x02 = ',
                'x03 = ',
            ])
            ParValues = np.array([
                self.k,
                self.r,
                self.T0,
                self.N,
                self.R,
                self.T,
                self.dt,
                self.Dt,
                self.L,
                self.gamma,
                self.eta,
                self.Lambda,
                self.delta,
                self.beta,
                self.a,
                self.NN,
                self.u,
                self.sigma1,
                self.sigma2,
                self.Xzero[0, 0],
                self.Xzero[0, 1],
                self.Xzero[0, 2]
            ])
            strPrefix = str(self.Dt)
            name1 = 'StoParameters' + strPrefix + '.txt'
            '''
            name2 = 'StoSolution' + strPrefix + '.txt'
            name3 = 'StoRefSolution' + str(self.dt) + '.txt'
            '''
            PARAMETERS = np.column_stack((tagPar, ParValues))
            np.savetxt(name1, PARAMETERS, delimiter=" ", fmt="%s")
            '''
            np.save(name2,
                np.transpose(
                    (
                        t, Uem1, Uem2, Uem3, Ustk1, Ustk2, Ustk3, Utem1,
                        Utem2, Utem3
                    )
                ))
            np.savetxt(name3,
                np.transpose(
                    (
                        self.t, Ueem1, Ueem2, Ueem3
                    )
                ))
            if self.sigma1 == 0.0:
                if self.sigma2 == 0.0:
                    DeterministicData()
                    return
            StochasticData()
            '''
        return
