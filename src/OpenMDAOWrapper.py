import openmdao.api as om
import numpy as np

def MakeSymmetric(v):
    return np.array([
        [v[0], v[1], v[2]],
        [v[1], v[3], v[4]],
        [v[2], v[4], v[5]],
    ])

def MakeSymmetricTraceless(v):
    a11, a12, a13, a22, a23 = v
    a33 = - (a11 + a22)
    return np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33],
    ])

def PackInputs(Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T):
    vDelta_Sigma = [Delta_Sigma[0, 0], Delta_Sigma[0, 1], Delta_Sigma[0, 2], Delta_Sigma[1, 1], Delta_Sigma[1, 2], Delta_Sigma[2, 2]]
    vDelta_f = Delta_f
    vDelta_epsilon_bar_T = [Delta_epsilon_bar_T[0, 0], Delta_epsilon_bar_T[0, 1], Delta_epsilon_bar_T[0, 2], Delta_epsilon_bar_T[1, 1], Delta_epsilon_bar_T[1, 2]]
    vDelta_lambda_epsilon_T = Delta_lambda_epsilon_T
    return np.concatenate([vDelta_Sigma, [vDelta_f], vDelta_epsilon_bar_T, [vDelta_lambda_epsilon_T]])

def UnpackInputs(v):
    vDelta_Sigma = v[0:6]
    Delta_f = v[6]
    vDelta_epsilon_bar_T = v[7:12]
    Delta_lambda_epsilon_T = v[12]

    Delta_Sigma = MakeSymmetric(vDelta_Sigma)
    Delta_epsilon_bar_T = MakeSymmetricTraceless(vDelta_epsilon_bar_T)
    return Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T

class SystemResiduals(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_input", default=13)
        self.options.declare("functions", types=dict)

    def setup(self):
        n_input = self.options["n_input"]
        self.add_input("x", shape=(n_input,))
        self.add_output("residuals", shape=(20,))  # 2x3 vector + 2 scalars = 8 residuals

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = outputs["x"]
        Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T = self.UnpackInputs(x)

        r1 = self.options["functions"]["f1"](Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T)
        r2 = self.options["functions"]["f2"](Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T)
        r3 = self.options["functions"]["f3"](Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T)
        r4 = self.options["functions"]["f4"](Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T)

        # Stack residuals into a flat array
        resvec = np.concatenate([[r1], [r2], r3.ravel(), r4.ravel()])
        residuals["x"][:] = resvec[:len(x)]
        outputs["residuals"][:] = resvec

    def solve_nonlinear(self, inputs, outputs):
        outputs["x"][:] = np.zeros_like(outputs["x"])

    def UnpackInputs(self, x):
        vDelta_Sigma = x[0:6]
        Delta_f = x[6]
        vDelta_epsilon_bar_T = x[7:12]
        Delta_lambda_epsilon_T = x[12]

        Delta_Sigma = MakeSymmetric(vDelta_Sigma)
        Delta_epsilon_bar_T = MakeSymmetricTraceless(vDelta_epsilon_bar_T)
        return Delta_Sigma, Delta_f, Delta_epsilon_bar_T, Delta_lambda_epsilon_T

def solve_increment(functions, x0=None):
    prob = om.Problem()

    # Add the residual component
    prob.model.add_subsystem("sys", SystemResiduals(n_input=13, functions=functions), promotes=["*"])


    # Nonlinear solver to drive residuals to zero
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.linear_solver = om.DirectSolver()  # for solving linear system in Newton

    prob.model.nonlinear_solver.options["atol"] = 1e-10  # or stricter
    prob.model.nonlinear_solver.options["rtol"] = 1e-10
    prob.model.nonlinear_solver.options["maxiter"] = 5000
    prob.model.nonlinear_solver.options["iprint"] = 2


    prob.setup()

    prob.set_val("x", x0)

    prob.run_model()

    xopt = prob.get_val("x")
    print("Residuals:", prob.get_val("residuals"))
    return UnpackInputs(xopt)