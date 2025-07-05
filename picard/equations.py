import abc

import numpy as np
import torch
from picard.utils import (
    get_device_prefer_cpu,
    GaussianMixtureDiagonalCovariance,
    GaussianDiagonalCovariance,
)


class ParametersMixin:
    def __init__(self):
        self._attr_names = set()
        self._device = None

    @property
    def device(self):
        return self._device

    def to(self, *, device):
        self._device = get_device_prefer_cpu(device)
        for attr_name in self._attr_names:
            setattr(self, attr_name, getattr(self, attr_name).to(device=self._device))
        # if has gmm, we need to update the device of the gmm
        if hasattr(self, "gaussian_init"):
            mean_vector = 0 * torch.ones(self.nx, device=self._device)
            covariance_matrix = self.alpha_init * torch.eye(self.nx, device=self._device)
            self.gaussian_init = GaussianDiagonalCovariance(mean_vector, covariance_matrix)
        if hasattr(self, "gmm_calc"):
            self.gmm_calc = self.gmm_calc.to(device=self._device)
        if hasattr(self, "gmm_init"):
            self.gmm_init = self.get_gmm_t(self.T)
        return self

    def move_gmm_to_device(self, gmm, device):
        gmm.component_distribution.mean = gmm.component_distribution.mean.to(device)
        gmm.component_distribution.covariance_matrix = gmm.component_distribution.covariance_matrix.to(device)
        gmm.mixture_distribution.logits = gmm.mixture_distribution.logits.to(device)
        return gmm

    def __register_attr(self, attr_name):
        self._attr_names.add(attr_name)

    def __setattr__(self, key, value):
        if isinstance(value, torch.Tensor) and key not in self._attr_names:
            self.__register_attr(key)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        if item in self._attr_names:
            self._attr_names.remove(item)
        super().__delattr__(item)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"{', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_'))}"
            ")"
        )


class Equation(ParametersMixin):
    r"""
    The general form of  an equation is:
        u_t + 1/2 Trace(Sigma Sigma^T) u_xx + <mu, u_x> + ff(t, x, u, u_x) = 0
        u(T, x) = g(x)
    Here we note that in the literature, it is more common to have:
        u_t + 1/2 Trace(Sigma Sigma^T) u_xx + <mu, u_x> + fff(t, x, u, u_x^T Sigma) = 0
    with ff(t, x, y, w) = fff(t, x, y, z=Sigma w).
    We introduce `ff` since `Sigma` is a matrix that should be implemented in the equation.
    We keep the `z` in `fff=fff(t, x, y, z)`.
    We also introduce `f` when `fff` does not depend on `z`.
    ---
    The PDE corresponds to two SDEs:
        dX = mu dt + dW
        D = I + \int \mu_x D dt + \int <\Sigma_x,D> dW
    """

    has_gradient_term = None
    has_laplacian_term = None
    has_hessian_term = None
    num_v_samples = None
    supported_approximate_methods = tuple()

    def __init__(self, T: float = 1, nx: int = 1):
        super().__init__()
        self.T = T
        self.nx = nx
        self.nu = 1

    def ffl(self, t, x, y, z, laplacian):
        pass

    @abc.abstractmethod
    def fff(self, t, x, y, z):
        pass

    @abc.abstractmethod
    def ff(self, t, x, y, w):
        pass

    @abc.abstractmethod
    def f(self, t, x, y):
        pass

    @abc.abstractmethod
    def sample_x_ts(self, t, s, x: torch.Tensor, return_dW=False):
        """
        sample the dX = mu dt + sigma dW starting from x at time t and ending at time s
        :param t:
        :param s:
        :param x:
        :return:
        """
        pass

    def sample_x(self, t: torch.Tensor):
        return self.sample_x_ts(torch.zeros_like(t), t, self.sample_x0(len(t)))

    # --- below are specifications: terminal condition and initial distributions ---

    def sample_x0(self, n: int):
        return torch.randn(n, self.nx, device=self._device)

    @abc.abstractmethod
    def g(self, x):
        pass

    def exact_solution(self, t, x):
        """
        compute the exact solution u(t, x)
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError

    def numerical_solution(self, t, x):
        """
        compute the numerical solution u(t, x), might be expansive
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError

    def laplacian(self, t, x):
        r"""
        compute the laplacian of the exact solution \Delta u(t, x)
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError

    def u_t(self, t, x):
        r"""
        compute the time derivative of the exact solution u_t(t, x)
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError

    def u_x(self, t, x):
        r"""
        compute the gradient of the exact solution u_x(t, x)
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError

    def u_u_x(self, t, x):
        r"""
        compute u(t, x) and u_x(t, x):
            this function is introduced since the formula of u_x might include u
            it can accelerate the computation by computing u and u_x at the same time
        :param t:
        :param x:
        :return:
        """
        raise NotImplementedError


class DiffusionEquation(Equation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I`.
        u_t + alpha/2 u_xx + <mu, u_x> + ff(t, x, u, u_x) = 0
        u_t + alpha/2 u_xx + <mu, u_x> + fff(t, x, u, sqrt(alpha) u_x) = 0
    """

    def __init__(self, *, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.scalar_tensor(alpha)
        self.alpha_sqrt = torch.sqrt(self.alpha)

    def ff(self, t, x, y, w):
        return self.fff(t, x, y, self.alpha_sqrt * w)

    def pinn_function(self, t, x, u, u_t, u_x, u_xx):
        return u_t + self.alpha / 2 * u_xx + self.ff(t, x, u, u_x)


class SimpleDiffusionEquation(DiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I` and `mu = 0`.
        u_t + alpha/2 u_xx + ff(t, x, u, u_x) = 0
        u_t + alpha/2 u_xx + fff(t, x, u, sqrt(alpha) u_x) = 0
    ---
    In this case:
        dX = sqrt(alpha) dW
        D = I
    """

    def sample_x_ts(self, t, s, x: torch.Tensor, return_dW=False):
        """
        sample the dX = mu dt + sigma dW starting from x at time t and ending at time s
        :param t: shape (n_batch, 1)
        :param s: shape (n_batch, 1)
        :param x: shape (n_batch, nx)
        :return:
        """
        dW = torch.randn_like(x)
        x_next = x + torch.sqrt(s - t) * self.alpha_sqrt * dW
        if return_dW:
            return x_next, dW
        else:
            return x_next


class SimpleDiffusionEquationWithoutZ(SimpleDiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I`, `mu = 0` and `ff` does not depend on `z`.
        u_t + alpha/2 u_xx + f(t, x, u) = 0
    """

    has_gradient_term = False
    has_laplacian_term = False
    has_hessian_term = False

    def fff(self, t, x, y, z):
        return self.f(t, x, y)

    # Though `ff` has implementation that calls `fff`,
    # we override to save computation since we know that `z` is not used
    def ff(self, t, x, y, w):
        return self.f(t, x, y)


class SimpleDiffusionEquationWithZ(SimpleDiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I`, `mu = 0` and `ff` depends on `z`.
        u_t + alpha/2 u_xx + fff(t, x, u, sqrt(alpha)u_x) = 0
    """

    has_gradient_term = True
    has_laplacian_term = False
    has_hessian_term = False

    def f(self, t, x, y):
        raise NotImplementedError("The equation has dependence on z, use fff or ff instead.")


class Cha(SimpleDiffusionEquationWithZ):
    r"""
    Two forms of the equation:
        u_t + sigma^2/2 u_xx + [sigma^2 k u-1/(k d)-sigma^2 k/2]\sum_i u_{x_i} = 0,
        u_t + alpha/2 u_xx + [alpha k u-1/(k d)-alpha k/2]\sum_i u_{x_i} = 0.
    It is in the standard form:
        u_t + alpha/2 u_xx + fff(t, x, u, sqrt(alpha) u_x) = 0,
    with fff(t, x, y, z) = [sqrt(alpha) k y - 1/(k sqrt(alpha) d) - sqrt(alpha) k/2] \sum_i z_i
                         = sigma[ k y - 2 / (2k sigma^2 d) - k^2 sigma^2 d / 2(k sigma^2 d)] \sum_i z_i
                         = sigma [ky - (2+k^2 sigma^2 d) / (2k sigma^2 d)] \sum_i z_i.
                         = sqrt(alpha) [ky - (2+k^2 alpha d) / (2k alpha d)] \sum_i z_i.
    ---
    we have exact solution:
        u(t, x) = exp(t + k\sum x_i) / [1 + exp(t + k\sum x_i)] = sigmoid(t + k\sum x_i)
    letting g(x) = exp(T + k\sum x_i) / [1 + exp(T + k\sum x_i)] = sigmoid(T + k\sum x_i).
    """

    def __init__(self, nx: int, alpha: float, k=1.0, T: float = 1.0):
        super().__init__(nx=nx, alpha=alpha, T=T)
        self.k = torch.scalar_tensor(k / np.sqrt(self.nx), device=self._device)
        self.alpha_d = self.alpha * self.nx
        self.k_alpha_d = self.k * self.alpha_d
        self.k_alpha_d_2 = self.k_alpha_d * 2
        self.k2_alpha_d = self.k * self.k_alpha_d

    def __str__(self):
        return (
            r"u_t + alpha/2 u_xx + [alpha k u-1/(k d)-alpha k/2]\sum_i u_{x_i} = 0"
            f" with nx={self.nx}, alpha={self.alpha:.5f}, k={self.k:.5f}"
        )

    def fff(self, t, x, y, z):
        return (
                self.alpha_sqrt
                * (self.k * y - (2 + self.k2_alpha_d) / self.k_alpha_d_2)
                * torch.sum(z, dim=-1, keepdim=True)
        )

    def g(self, x):
        return torch.sigmoid(self.T + self.k * torch.sum(x, dim=-1, keepdim=True))

    def g_x(self, x):
        g_x = self.g(x)

        # Step 2: 计算 sigmoid(T + k * sum(x)) 和 1 - sigmoid(T + k * sum(x))
        sigmoid_value = g_x
        sigmoid_derivative = sigmoid_value * (1 - sigmoid_value)

        grad = self.k * sigmoid_derivative

        return grad

    def exact_solution(self, t, x):
        return torch.sigmoid(t + self.k * torch.sum(x, dim=-1, keepdim=True))

    def u_t(self, t, x):
        uu = torch.sigmoid(t + self.k * torch.sum(x, dim=-1, keepdim=True))
        return (1 - uu) * uu

    def u_x(self, t, x):
        uu = torch.sigmoid(t + self.k * torch.sum(x, dim=-1, keepdim=True))
        return torch.ones_like(x) * (self.k * uu * (1 - uu))

    def u_u_x(self, t, x):
        u = self.exact_solution(t, x)
        u_x = torch.ones_like(x) * (self.k * u * (1 - u))
        return u, u_x

    def sample_x0(self, n: int):
        return torch.zeros(n, self.nx, device=self._device)

    def ffh(self, t, x, u, u_x, hess_u):
        return self.ff(t, x, u, u_x)


class SimpleDiffusionEquationWithLaplacian(SimpleDiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I`, `mu = 0` and `ff` depends on `z` and `laplacian u`.
        u_t + alpha/2 u_xx + ffl(t, x, u, u_x, Tr(Hess u)) = 0
        u_t + alpha/2 u_xx + fffl(t, x, u, sqrt(alpha)u_x, u_xx) = 0
    """

    has_gradient_term = True
    has_laplacian_term = True
    has_hessian_term = False

    def f(self, t, x, y):
        raise NotImplementedError("The equation has dependence on z and laplacian, use ffl instead.")

    def ff(self, t, x, y, z):
        raise NotImplementedError("The equation has dependence on z and laplacian, use ffl instead.")


class SimpleDiffusionEquationWithHessian(SimpleDiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I`, `mu = 0` and `ff` depends on `z` and `hessian u`.
        u_t + alpha/2 u_xx + ffh(t, x, u, u_x, Hess u) = 0
    """

    has_gradient_term = True
    has_laplacian_term = False
    has_hessian_term = True

    def f(self, t, x, y):
        raise NotImplementedError("The equation has dependence on z and hessian, use ffh instead.")

    def ff(self, t, x, y, z):
        raise NotImplementedError("The equation has dependence on z and hessian, use ffh instead.")

    def u_hessian(self, t, x):
        raise NotImplementedError("u hessian is not implemented.")

    def g_x(self, x):
        raise NotImplementedError("g_x is not implemented.")

    def u_u_x_u_hessian(self, t, x):
        u = self.exact_solution(t, x)
        u_x = self.u_x(t, x)
        u_hessian = self.u_hessian(t, x)
        return u, u_x, u_hessian


class GBMEquationComplexExact(SimpleDiffusionEquationWithHessian, abc.ABC):
    r"""
    u_t + 1/2 u_xx + 1/4 \sum abs(u_ii) - f = 0
    g(x) = \sum_k v^k sin(w_0^k * T + \sum w_i^k * x_i)
    ---
    Exact solution: u(t, x) = \sum_k v^k sin(w_0^k *t + \sum w_i^k * x_i)
    u_t = \sum_k v^k w_0^k cos(w_0^k * t + \sum w_i^k * x_i)
    u_x = \sum_k v^k w_i^k cos(w_0^k * t + \sum w_i^k * x_i)
    u_hessian = -\sum_k v^k w_i^k w_j^k sin(w_0^k * t + \sum w_i^k * x_i)
    """

    supported_approximate_methods = ("SDGD",)

    def __init__(self, nx: int, alpha: float = 1.0, T: float = 1.0):
        super().__init__(nx=nx, alpha=alpha, T=T)
        self.inverse_d_sqrt = torch.scalar_tensor(1 / np.sqrt(self.nx), device=self._device)
        self.d = torch.scalar_tensor(self.nx, device=self._device)
        self.d_sqrt = torch.sqrt(self.d)
        self.T_tensor = torch.scalar_tensor(self.T, device=self._device)

        try:
            print(f"Loading w and v for GBM equation with {nx} dimensions")
            self.w = torch.load(f"gbm_2nodes_w_{nx}d.pt")
            self.v = torch.load(f"gbm_2nodes_v_{nx}d.pt")
        except FileNotFoundError:
            print(f"Generating w and v for GBM equation with {nx} dimensions")
            num_neurons = 2
            self.w = torch.randn(num_neurons, 1 + self.nx, device=self._device) * self.inverse_d_sqrt
            self.w[:, 0] = torch.ones(num_neurons, device=self._device)
            self.v = torch.randn(num_neurons, 1, device=self._device)
            torch.save(self.w, f"gbm_2nodes_w_{nx}d.pt")
            torch.save(self.v, f"gbm_2nodes_v_{nx}d.pt")

    def g(self, x):
        return self.exact_solution(self.T_tensor, x)

    def g_x(self, x):
        return self.u_x(self.T_tensor, x)

    def exact_solution(self, t, x):
        # check t dtype and x dtype
        tx = torch.cat([t * torch.ones(x.shape[0], 1, device=self._device), x], dim=-1)
        return torch.sin(tx @ self.w.t()) @ self.v

    def u_t(self, t, x):
        tx = torch.cat([t * torch.ones(x.shape[0], 1, device=self._device), x], dim=-1)
        return torch.cos(tx @ self.w.t()) @ (self.v * self.w[:, 0:1])

    def u_x(self, t, x):
        tx = torch.cat([t * torch.ones(x.shape[0], 1, device=self._device), x], dim=-1)
        return torch.cos(tx @ self.w.t()) @ (self.v * self.w[:, 1:])

    def u_u_x(self, t, x):
        u = self.exact_solution(t, x)
        u_x = self.u_x(t, x)
        return u, u_x

    def u_hessian(self, t, x):
        tx = torch.cat([t * torch.ones(x.shape[0], 1, device=self._device), x], dim=-1)
        sin_term = -torch.sin(tx @ self.w.t())  # N*m
        outer_products = self.w[:, 1:].unsqueeze(2) * self.w[:, 1:].unsqueeze(1)  # m*D*D
        weights = self.v.unsqueeze(-1) * outer_products  # m*D*D
        return torch.einsum("ij,jkl->ikl", sin_term, weights)

    def laplacian(self, t, x):
        tx = torch.cat([t * torch.ones(x.shape[0], 1, device=self._device), x], dim=-1)
        sin_term = torch.sin(tx @ self.w.t())
        return -sin_term @ (self.v * torch.sum(self.w[:, 1:] ** 2, dim=-1, keepdim=True))

    def ffi(self, t, x, u, u_ii):
        laplacian = self.d * torch.mean(u_ii, dim=-1, keepdim=True)
        nonlinear = self.d * torch.mean(torch.abs(u_ii), dim=-1, keepdim=True)
        return (
                0.5 * (1.0 - self.alpha) * laplacian
                + 1 / 4 * nonlinear
                - self.u_t(t, x)
                - 1 / 2 * self.laplacian(t, x)
                - 1 / 4 * torch.abs(torch.diagonal(self.u_hessian(t, x), dim1=1, dim2=2)).sum(-1, keepdim=True)
        )

    def ffh(self, t, x, u, u_x, hess_u):
        u_ii = torch.diagonal(hess_u, dim1=1, dim2=2)
        return self.ffi(t, x, u, u_ii)

    def pinn_function(self, t, x, u, u_t, u_x, u_ii):
        # Only in SDGD
        laplacian = self.d * torch.mean(u_ii, dim=-1, keepdim=True)
        nonlinear = self.d * torch.mean(torch.abs(u_ii), dim=-1, keepdim=True)
        return (
                u_t
                + 1 / 2 * laplacian
                + 1 / 4 * nonlinear
                - self.u_t(t, x)
                - 1 / 2 * self.laplacian(t, x)
                - 1 / 4 * torch.abs(torch.diagonal(self.u_hessian(t, x), dim1=1, dim2=2)).sum(-1, keepdim=True)
        )

    def sample_x0(self, n: int):
        return torch.zeros(n, self.nx, device=self._device)


class ComplexDiffusionEquation(DiffusionEquation, abc.ABC):
    """
    This equation has `Sigma = sqrt(alpha) I` and `F`.
        u_t + alpha/2 u_xx + ff(t, x, u, u_x) = 0
        u_t + alpha/2 u_xx + fff(t, x, u, sqrt(alpha) u_x) = 0
    ---
    Default FWD:
        dX = F dt + sqrt(alpha) dW
        D = I
    """

    has_gradient_term = True
    has_laplacian_term = False
    has_hessian_term = False

    def __init__(
            self,
            nx,
            T,
            theta: float = 1.0,
            mu: float = 0.0,
            alpha: float = 1.0,
            num_components=2,
            mean_scale=1.0,
            var_scale=2.0,
            alpha_scale=4.0,
            **kwargs,
    ):
        super().__init__(nx=nx, T=T, alpha=alpha, **kwargs)
        self.alpha = torch.scalar_tensor(alpha, device=self._device)
        self.alpha_sqrt = torch.sqrt(self.alpha)
        self.theta = torch.scalar_tensor(theta, device=self._device)
        self.mu = torch.scalar_tensor(mu, device=self._device)
        self.d = torch.scalar_tensor(nx, device=self._device)

        self.num_components = num_components
        try:
            # load the means and variances
            print(
                f"Loading mean_{self.nx}d_{self.num_components}.pt, var_{self.nx}d_{self.num_components}.pt and pi_{self.nx}d_{self.num_components}.pt"
            )
            self.mean = torch.load(f"mean_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")
            self.var = torch.load(f"var_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")
            self.pi = torch.load(f"pi_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")
        except FileNotFoundError:
            # means are between [-1, 1]
            print(f"Creating new means and variances for {self.num_components} components")
            self.mean = torch.stack(
                [mean_scale * (torch.rand(nx, device=self._device) * 2 - 1) for _ in range(num_components)]
            )
            self.var = torch.stack([var_scale * torch.eye(nx, device=self._device) for _ in range(num_components)])
            self.pi = torch.rand(self.num_components, device=self._device)
            self.pi = self.pi / self.pi.sum()
            torch.save(self.mean, f"mean_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")
            torch.save(self.var, f"var_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")
            torch.save(self.pi, f"pi_{self.nx}d_ms={mean_scale}_vs={var_scale}_{self.num_components}.pt")

        print(f"mean: {self.mean}")
        print(f"var: {self.var}")
        print(f"pi: {self.pi}")

        # turn mean, var and pi into float64, if needed
        # self.mean = self.mean.double()
        # self.var = self.var.double()
        # self.pi = self.pi.double()

        self.gmm_calc = GaussianMixtureDiagonalCovariance(self.mean, self.var, self.pi)

        self.alpha_init = alpha_scale * self.alpha
        self.batch_size = 100000  # hard code the batch size for GMM in case of memory error

    def sample_x_ts(self, t, s, x: torch.Tensor, return_dW=False):
        """
        sample the dX = mu dt + sigma dW starting from x at time t and ending at time s
        :param t: shape (n_batch, 1)
        :param s: shape (n_batch, 1)
        :param x: shape (n_batch, nx)
        :return:
        """
        dW = torch.randn_like(x)
        x_next = x + torch.sqrt(s - t) * self.alpha_sqrt * dW
        if return_dW:
            return x_next, dW
        else:
            return x_next

    def f(self, t, x, y):
        raise NotImplementedError("The equation has dependence on z, use fff or ff instead.")

    def __str__(self):
        return (
            r"u_t + alpha/2 u_xx - F u_x - |u_x|^2 + alpha/2 div(F) = 0" f" with nx={self.nx}, alpha={self.alpha:.5f}"
        )

    def F(self, t, x, y, z):
        raise NotImplementedError("Implement force term.")

    def ff(self, t, x, y, z):
        raise NotImplementedError("Implement ff term.")

    def fff(self, t, x, y, z):
        return self.ff(t, x, y, self.alpha_sqrt * z)

    def g(self, x):
        return -self.gmm_calc.log_prob(x)

    def g_x(self, x):
        return -self.gmm_calc.grad_log_prob(x)


class OUProcessEquation(ComplexDiffusionEquation):
    """
    This equation has `Sigma = sqrt(alpha) I` and `mu = -x`.
        u_t + alpha/2 u_xx + ff(t, x, u, u_x) = 0
        u_t + alpha/2 u_xx + fff(t, x, u, sqrt(alpha) u_x) = 0
    ---
    Default FWD:
        dX = theta * (mu - X) dt + sqrt(alpha) dW
        D = I
    """

    def __init__(self, nx, theta: float = 1.0, mu: float = 0.0, alpha: float = 1.0, num_components=2, **kwargs):
        super().__init__(nx=nx, theta=theta, mu=mu, alpha=alpha, num_components=num_components, **kwargs)
        mean_vector = 0 * torch.ones(self.nx, device=self._device)
        covariance_matrix = self.alpha_init * torch.eye(self.nx, device=self._device)
        self.gaussian_init = GaussianDiagonalCovariance(mean_vector, covariance_matrix)
        # the limit of the OU process
        mean_vector = self.mu * torch.ones(self.nx, device=self._device)
        covariance_matrix = (self.alpha / (2 * self.theta)) * torch.eye(self.nx, device=self._device)
        self.ou_gaussian_limit = GaussianDiagonalCovariance(mean_vector, covariance_matrix)
        # true distribution at time 0
        self.gmm_init = self.get_gmm_t(self.T)

    def __str__(self):
        return (
            r"u_t + alpha/2 u_xx - theta (mu - x) u_x - |u_x|^2 + alpha/2 div(theta (mu - x)) = 0"
            f" with nx={self.nx}, alpha={self.alpha:.5f}, theta={self.theta:.5f}, mu={self.mu:.5f}"
        )

    def fff(self, t, x, y, z):
        return self.ff(t, x, y, self.alpha_sqrt * z)

    def ou_process_params(self, x0, var0, t):
        mean = self.mu + (x0 - self.mu) * torch.exp(-self.theta * t)
        variance = var0 * torch.exp(-2 * self.theta * t) + (self.alpha / (2 * self.theta)) * (
                1 - torch.exp(-2 * self.theta * t)
        ) * torch.eye(self.nx, device=self._device)
        return mean, variance

    def get_gmm_t(self, t):
        means_t = []
        vars_t = []
        for i in range(self.num_components):
            mean_t, var_t = self.ou_process_params(self.mean[i], self.var[i], t)
            means_t.append(mean_t)
            vars_t.append(var_t)
        means_t_all = torch.stack(means_t)
        vars_t_all = torch.stack(vars_t)
        new_gmm = GaussianMixtureDiagonalCovariance(means_t_all, vars_t_all, self.pi)
        return new_gmm

    def exact_solution(self, t, x):
        results = [-self.get_gmm_t(self.T - t[n]).log_prob(x[n]).reshape(-1, 1) for n in range(t.shape[0])]
        return torch.stack(results).reshape(*t.shape)

    def exact_solution_same_t(self, t, x):
        return -self.get_gmm_t(self.T - t).log_prob(x).reshape(-1, 1)

    def F(self, t, x, y, z):
        return self.theta * (self.mu - x)

    def ff(self, t, x, y, z):
        result = (
                -(self.F(t, x, y, z) * z).sum(dim=-1, keepdim=True)
                - self.alpha / 2 * torch.sum(z ** 2, dim=-1, keepdim=True)
                - self.d * self.theta * torch.ones_like(y)
        )
        return result

    def u_t(self, t, x):
        # cal u_t by auto-diff
        t.requires_grad = True
        u = self.exact_solution(t, x)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        return u_t

    def u_x(self, t, x, retain_graph=False):
        # cal u_x by auto-diff
        x.requires_grad = True
        with torch.enable_grad():
            if (t.shape[0] == 1) and (x.shape[0] > 1):
                u = self.exact_solution_same_t(t, x)
                u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=retain_graph)[0]
            else:
                assert t.shape[0] == x.shape[0]
                u = self.exact_solution(t, x)
                u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=retain_graph)[0]
        return u_x

    def u_x_same_t(self, t, x, retain_graph=False):
        # cal u_x by auto-diff
        x.requires_grad = True
        t = t[0]
        with torch.enable_grad():
            u = self.exact_solution_same_t(t, x)
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=retain_graph)[0]
        return u_x

    def u_u_x(self, t, x):
        u = self.exact_solution(t, x)
        u_x = self.u_x(t, x)
        return u, u_x

    def laplacian(self, t, x):
        # cal u_xx by auto-diff
        from picard.utils import get_laplacian

        x.requires_grad = True
        u_x = self.u_x(t, x, retain_graph=True)
        return get_laplacian(x, self.exact_solution(t, x), u_x)

    def sample_x0(self, n: int):
        return self.gaussian_init.sample((n,))

    def ffh(self, t, x, u, u_x, hess_u):
        return self.ff(t, x, u, u_x)
