import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

rs = 1
y_px, x_px = (1000, 1000)

SKYBOX = np.asarray(Image.open("skybox.png").convert("RGB")) / 255.0
SKY_H, SKY_W = SKYBOX.shape[:2]


def sample_skybox(theta, phi):
    u = ((phi % (2 * np.pi)) / (2 * np.pi)) * SKY_W
    v = (theta / np.pi) * SKY_H
    i = int(np.clip(v, 0, SKY_H - 1))
    j = int(np.clip(u, 0, SKY_W - 1))
    return tuple(SKYBOX[i, j])


def tetrad_gen(r, theta):
    return np.diag([
        1 / np.sqrt(1 - rs / r),
        np.sqrt(1 - rs / r),
        1 / r,
        1 / (r * np.sin(theta)),
    ])


def initial_conditions(x_px, y_px, x_lim, y_lim):
    r = 40
    theta = np.pi / 2 + 1e-1
    psi = np.pi / 2 + 1e-1
    x, y = np.meshgrid(
        np.linspace(-x_lim / 2, x_lim / 2, x_px),
        np.linspace(-y_lim / 2, y_lim / 2, y_px),
    )
    pixels = np.stack([-np.ones_like(x), -y, x], axis=-1)
    d = pixels / np.linalg.norm(pixels, axis=-1, keepdims=True)
    d = d.reshape(-1, 3)
    d = np.hstack([-np.ones((d.shape[0], 1)), d])
    tetrad = tetrad_gen(r, theta)
    vels = d @ tetrad
    return np.array([0, r, theta, psi]), vels


def christoffel(r, theta, rs=1):
    Gamma = np.zeros((4, 4, 4))
    sin_t = np.sin(theta); cos_t = np.cos(theta)
    Gamma[0, 0, 1] = rs / (2 * r * (r - rs)); Gamma[0, 1, 0] = Gamma[0, 0, 1]
    Gamma[1, 0, 0] = rs * (r - rs) / (2 * r**3)
    Gamma[1, 1, 1] = -rs / (2 * r * (r - rs))
    Gamma[1, 2, 2] = -(r - rs)
    Gamma[1, 3, 3] = -(r - rs) * sin_t**2
    Gamma[2, 1, 2] = 1 / r; Gamma[2, 2, 1] = Gamma[2, 1, 2]
    Gamma[2, 3, 3] = -sin_t * cos_t
    Gamma[3, 1, 3] = 1 / r; Gamma[3, 3, 1] = Gamma[3, 1, 3]
    Gamma[3, 2, 3] = cos_t / sin_t; Gamma[3, 3, 2] = Gamma[3, 2, 3]
    return Gamma


def cross_horizon(t, QP): return QP[1] - rs * 1.5
def disk_hit(t, QP): return QP[2] - np.pi / 2
def escape(t, QP): return QP[1] - 200

cross_horizon.terminal = True
disk_hit.terminal = False
escape.terminal = True

R_ISCO = 6
R_OUTER = 20


def dynamics(t, QP):
    QP = QP.reshape(2, 4)
    a = -np.einsum("mab,a,b->m", christoffel(*QP[0][[1, 2]]), QP[1], QP[1])
    return np.concatenate([QP[1], a])


def trace_ray(args):
    pos, v = args
    sol = solve_ivp(dynamics, (0, 500), [*pos, *v],
                    max_step=1.0,
                    events=[cross_horizon, disk_hit, escape])
    disk_crossings = sol.y_events[1]
    disk_times = sol.t_events[1]
    valid_mask = [R_ISCO <= y[1] <= R_OUTER for y in disk_crossings]
    valid_disk_times = disk_times[valid_mask] if len(disk_times) else disk_times
    horizon_time = sol.t_events[0][0] if len(sol.t_events[0]) else float("inf")
    escape_time = sol.t_events[2][0] if len(sol.t_events[2]) else float("inf")
    disk_time = valid_disk_times[0] if len(valid_disk_times) else float("inf")
    first = int(np.argmin([horizon_time, disk_time, escape_time]))
    if first == 0:
        return (0.0, 0.0, 0.0)
    if first == 1:
        return (0.0, 1.0, 0.0)
    if first == 2 and len(sol.t_events[2]):
        final_state = sol.y_events[2][0]
        _, _, theta_f, phi_f = final_state[:4]
        return sample_skybox(theta_f, phi_f)
    # ray stalled or timed out — treat as captured
    return (0.0, 0.0, 0.0)


if __name__ == "__main__":
    from tqdm import tqdm
    pos, vel = initial_conditions(x_px, y_px, 2, 2)
    args = [(pos, v) for v in vel]
    with Pool(cpu_count()) as pool:
        colors = list(tqdm(pool.imap(trace_ray, args), total=len(args)))
    image = np.array(colors).reshape(y_px, x_px, 3)
    img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    img.save("blackhole.png")