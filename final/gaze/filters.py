import math

class OneEuroFilter:
    """
    Adaptive smoothing filter.
    - When gaze is slow/still: smooths aggressively (reduces jitter)
    - When gaze moves fast: lets signal through quickly (reduces lag)
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = x0
        self.dx_prev    = dx0
        self.t_prev     = t0

    def alpha(self, t_e, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        a_d    = self.alpha(t_e, self.d_cutoff)
        dx     = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self.alpha(t_e, cutoff)
        x_hat  = a * x + (1 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat