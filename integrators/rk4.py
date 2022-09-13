
def rk4(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + dt/2 * k1, t + dt/2)
    k3 = f(x + dt/2 * k2, t + dt/2)
    k4 = f(x + dt * k3, t + dt)

    out = x + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return out