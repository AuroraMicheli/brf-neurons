import torch
from .. import functional
from .linear_layer import LinearMask
from ..functional import step

################################################################
# Neuron update functional
################################################################

DEFAULT_MASK_PROB = 0.

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_B_offset_a = 0.
DEFAULT_RF_ADAPTIVE_B_offset_b = 3.

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_OMEGA_a = 5.
DEFAULT_RF_ADAPTIVE_OMEGA_b = 10.

DEFAULT_RF_THETA = 1.  # 1.0 # * 0.1

DEFAULT_DT = 0.01
FACTOR = 1 / (DEFAULT_DT * 2)

DEFAULT_GAMMA = 0.9


def rf_update(
        x: torch.Tensor,  # injected current: input x weight
        z: torch.Tensor,
        u: torch.Tensor,  # membrane potential (real part)
        v: torch.Tensor,  # membrane potential (complex part)
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # # membrane update (complex)
    # u = u + u.mul(torch.complex(b, omega)).mul(dt) + x.mul(dt)
    u_ = u + b * u * dt - omega * v * dt + x * dt
    v = v + omega * u * dt + b * v * dt

    # generate spike
    # z = functional.StepDoubleGaussianGrad.apply(u.real - theta)
    z = functional.FGI_DGaussian(u_ - theta)

    # no reset or
    # soft reset # hard reset
    # u_ = u_ - z * theta  # * u_
    # v = v - z * theta # * v

    return z, u_, v


def brf_I_update(
        x: torch.Tensor,  # injected current: input x weight
        z_: torch.Tensor,
        u_: torch.Tensor,  # membrane potential (real part)
        v_: torch.Tensor,  # membrane potential (complex part)
        q_: torch.Tensor,  # refractory period
        b_0: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # membrane update u dim = (batch_size, hidden_size)
    u = u_ + b_0 * u_ * dt - omega * v_ * dt + x * dt
    v = v_ + omega * u_ * dt + b_0 * v_ * dt
    q = q_.mul(0.9) + z_

    z = functional.StepDoubleGaussianGrad.apply(u - theta - q)

    return z, u, v, q


def brf_II_update(
        x: torch.Tensor,  # injected current: input x weight
        z_: torch.Tensor,
        u_: torch.Tensor,  # membrane potential (real part)
        v_: torch.Tensor,  # membrane potential (complex part)
        q_: torch.Tensor,  # refractory period
        b_0: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    b = b_0 - q_

    exp_bdt = torch.exp(b * dt)
    cos_omega_dt = torch.cos(omega * dt)
    sin_omega_dt = torch.sin(omega * dt)

    u_cos_sin = u_ * cos_omega_dt - v_ * sin_omega_dt
    v_cos_sin = u_ * sin_omega_dt + v_ * cos_omega_dt

    u = exp_bdt * u_cos_sin + x * dt
    v = exp_bdt * v_cos_sin
    q = 0.9 * q_ + z_

    z = functional.StepDoubleGaussianGrad.apply(u - theta - q)

    return z, u, v, q


def adbrf_II_update(
        x: torch.Tensor,  # injected current: input x weight
        z_: torch.Tensor,
        u_: torch.Tensor,  # membrane potential (real part)
        v_: torch.Tensor,  # membrane potential (complex part)
        q_: torch.Tensor,  # refractory period
        b_0: torch.Tensor,  # attraction to resting state
        cos_omega_dt: torch.Tensor,  # eigen ang. frequency of the neuron
        sin_omega_dt: torch.Tensor,
        gamma: torch.Tensor,
        dt: torch.Tensor,  # 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    b = b_0 - q_

    exp_bdt = torch.exp(b * dt)

    u_cos_sin = u_ * cos_omega_dt - v_ * sin_omega_dt
    v_cos_sin = u_ * sin_omega_dt + v_ * cos_omega_dt

    u = exp_bdt * u_cos_sin + x * dt
    v = exp_bdt * v_cos_sin
    q = gamma * q_ + z_

    z = functional.FGI_DGaussian(u_ - theta - q)

    return z, u, v, q


def izhikevich_update(
        x: torch.Tensor,  # injected current: input x weight
        u: torch.Tensor,  # membrane potential (complex value)
        q: torch.Tensor,
        b: torch.Tensor,  # attraction to resting state
        omega: torch.Tensor,  # eigen ang. frequency of the neuron
        dt: float = DEFAULT_DT,  # torch.Tensor 0.01
        theta: float = DEFAULT_RF_THETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # membrane update u dim = (batch_size, hidden_size)
    u = u + u.mul(torch.complex(b, omega)).mul(dt) + x.mul(dt)

    # generate spike
    z = functional.StepDoubleGaussianGrad.apply(u.imag - theta)

    # reset membrane potential
    u = u - u.mul(z) + z.mul(1.j)

    return z, u, q


def sustain_osc(omega: torch.Tensor, dt: float = DEFAULT_DT) -> torch.Tensor:
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt


################################################################
# Layer classes
################################################################


class RFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            mask_prob: float = DEFAULT_MASK_PROB,
            b_offset: float = DEFAULT_RF_B_offset,
            adaptive_b_offset: bool = TRAIN_B_offset,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt: float = DEFAULT_DT,
            bias: bool = False,
            pruning: bool = False,
    ) -> None:
        super(RFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        if pruning:

            # LinearMask: prunes only hidden recurrent weights in forward pass
            self.linear = LinearMask(
                in_features=input_size,
                out_features=layer_size,
                bias=bias,
                mask_prob=mask_prob,
                lbd=input_size-layer_size,
                ubd=input_size,
            )

        else:

            self.linear = torch.nn.Linear(
                in_features=input_size,
                out_features=layer_size,
                bias=bias
            )

            torch.nn.init.xavier_uniform_(self.linear.weight)

        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)


        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x)

        z, u, v = state

        omega = torch.abs(self.omega)

        b = -torch.abs(self.b_offset)

        z, u, v = rf_update(
            x=in_sum,
            z=z,
            u=u,
            v=v,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v


class BRFCell(RFCell):
    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x)

        z, u, v, q = state

        omega = torch.abs(self.omega)

        p_omega = sustain_osc(omega)

        b_offset = torch.abs(self.b_offset)

        # divergence boundary
        b = p_omega - b_offset - q

        z, u, v, q = brf_II_update( #II for better stability 
            x=in_sum,
            z_=z,
            u_=u,
            v_=v,
            q_=q,
            b_0=b,
            omega=omega,
            dt=self.dt,
        )

        return z, u, v, q


class AdBRFIICell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            b_offset: float = DEFAULT_RF_B_offset,
            adaptive_b_offset: bool = TRAIN_B_offset,
            adaptive_b_offset_a: float = DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b: float = DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega_neg: bool = False,
            omega: float = DEFAULT_RF_OMEGA,
            adaptive_omega: bool = TRAIN_OMEGA,
            adaptive_omega_a: float = DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b: float = DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt: float = DEFAULT_DT,
            adaptive_dt: bool = False,
            adaptive_dt_const: float = 0.01,
            gamma: float = DEFAULT_GAMMA,
            adaptive_gamma: bool = True,
            adaptive_gamma_mu: float = DEFAULT_GAMMA,
            adaptive_gamma_std: float = 0.001,
            bias: bool = False,
    ) -> None:
        super(AdBRFIICell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        # define input, recurrent weights
        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        # both input and recurrent weights initialized with xavier uniform
        torch.nn.init.xavier_uniform_(self.linear.weight)

        # define omega (angular frequency of each neuron)
        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b
        self.omega_neg = omega_neg

        omega = torch.fft.fftfreq(layer_size*2, d=dt)[:layer_size] # torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            # torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)

        # define b_offset (dampening term for each neuron)
        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        # time constant, also trainable
        dt = dt * torch.ones(layer_size)

        self.adaptive_dt = adaptive_dt

        if adaptive_dt:
            self.dt = torch.nn.Parameter(dt)
            torch.nn.init.constant_(self.dt, adaptive_dt_const)
        else:
            self.register_buffer('dt', dt)

        # refractory period decay parameter
        gamma = gamma * torch.ones(layer_size)

        self.adaptive_gamma = adaptive_gamma

        if adaptive_gamma:
            self.gamma = torch.nn.Parameter(gamma)
            torch.nn.init.normal_(self.gamma, adaptive_gamma_mu, adaptive_gamma_std)
        # else:
        #     self.register_buffer('gamma', gamma)


    def forward(
            self, x: torch.Tensor, cos_omega_dt: torch.Tensor, sin_omega_dt: torch.Tensor, gamma: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        in_sum = self.linear(x)

        z, u, v, q = state

        b = -torch.abs(self.b_offset)

        z, u, v, q = adbrf_II_update(
            x=in_sum,
            z_=z,
            u_=u,
            v_=v,
            q_=q,
            b_0=b,
            cos_omega_dt=cos_omega_dt,
            sin_omega_dt=sin_omega_dt,
            gamma=gamma,
            dt=self.dt,
        )

        return z, u, v, q
