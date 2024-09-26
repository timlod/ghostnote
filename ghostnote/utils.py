import numpy as np

DIAMETER = 14 * 2.54


def cartesian_to_polar(x: float, y: float, r: float = None):
    """Convert 2D cartesian coordinates to polar coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param r: radius unit-normalize returned radius
    """
    if r is None:
        r = np.sqrt(x**2 + y**2)
    else:
        r = np.sqrt(x**2 + y**2) / r

    phi_radians = np.arctan2(y, x)

    # Adjust theta to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)

    return r, np.degrees(phi_radians)


def polar_to_cartesian(r: float, phi: float):
    """Convert 2D polar coordinates to cartesian coordinates.

    :param r: radius
    :param phi: angle in degrees
    """
    phi_radians = np.radians(phi)

    x = r * np.cos(phi_radians)
    y = r * np.sin(phi_radians)
    return x, y


def spherical_to_cartesian(
    r: float,
    phi: float,
    theta: float,
) -> (float, float, float):
    """Convert 3D spherical coordinates to Cartesian coordinates.

    By default, x-y rotation moves clockwise and starts at y=0 (East); and x-z
    rotation starts at x=0 moving counter-clockwise (up).

    :param r: radius
    :param phi: angle in the x-y plane in degrees
    :param theta: angle in the x-z plane in degrees

    :return: Cartesian coordinates as (x, y, z)
    """
    phi_radians = np.radians(phi)
    if theta < 0:
        theta = -theta
    else:
        theta = 90 - theta
    theta_radians = np.radians(theta)

    x = r * np.cos(phi_radians) * np.sin(theta_radians)
    y = r * np.sin(phi_radians) * np.sin(theta_radians)
    z = r * np.cos(theta_radians)

    return x, y, z


def cartesian_to_spherical(x: float, y: float, z: float):
    """Convert 3D cartesian coordinates to spherical/polar coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi_radians = np.arctan2(y, x)
    theta_radians = np.arccos(z / r)

    # Adjust phi to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)
    theta = np.degrees(theta_radians)
    if theta < 0:
        theta = -theta
    else:
        theta = 90 - theta
    return r, np.degrees(phi_radians), theta


def lag_map_2d(
    mic_a: tuple[int, int],
    mic_b: tuple[int, int],
    d: int = DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    c: float = 343.0,
    tol: int = 1,
):
    """Compute lag map for 2D microphone locations.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param c: the speed of sound through the medium of interest, in meters per
              second.  default is speed of sound through air (343m/s)
    :param tol: lags outside the drum are replaced with np.nan - within some
        tolerance (in centimeters) at the edges.  Note that the
        top/bottom/left/right/edges are naturally at the edge of the matrix,
        tolerance doesn't increase legality there
    """
    c *= scale * 100

    # This will give us a diameter to use which we can sample at millimeter
    # precision
    r = int(np.round(d * scale / 2))
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    circular_mask = i**2 + j**2 > ((r + tol * scale) ** 2)

    # compute lag in seconds from each potential location to microphones
    lag_a = np.sqrt((i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2) / c
    lag_b = np.sqrt((i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2) / c
    lag_map = np.round((lag_a - lag_b) * sr).astype(np.float32)
    lag_map[circular_mask] = np.nan
    return lag_map
