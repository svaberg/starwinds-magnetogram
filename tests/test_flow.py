import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib
import pytest
from itertools import chain

from contextlib import contextmanager
from time import time


from tests import context  # Test context


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    elapsed_time = time() - start

    log.info(f"{description}: {elapsed_time}")


# Think this is a version of Bresenham's algorithm.
def lic_flow(vectors, len_pix=10):
    vectors = np.asarray(vectors)
    m, n, two = vectors.shape
    if two != 2:
        raise ValueError

    result = np.zeros((2 * len_pix + 1, m, n, 2), dtype=np.int32)  # FIXME: int16?
    center = len_pix
    result[center, :, :, 0] = np.arange(m)[:, np.newaxis]
    result[center, :, :, 1] = np.arange(n)[np.newaxis, :]

    for i in range(m):
        for j in range(n):
            yid = i
            xid = j
            fx = .5
            fy = .5
            for k in range(len_pix):
                vx, vy = vectors[yid, xid]
                if vx >= 0:
                    tx = (1 - fx) / vx
                else:
                    tx = -fx / vx
                if vy >= 0:
                    ty = (1 - fy) / vy
                else:
                    ty = -fy / vy

                # assert tx >= 0
                # assert ty >= 0

                # Determine step direction.
                if tx < ty:
                    # X step
                    fy += vy * tx
                    if vx > 0:
                        xid += 1
                        fx = 0.
                    else:
                        xid -= 1
                        fx = 1.
                else:
                    # Y step
                    fx += vx * ty
                    if vy > 0:
                        yid += 1
                        fy = 0.
                    else:
                        yid -= 1
                        fy = 1.

                del tx, ty

                # Don't go outside the domain
                if xid < 0:
                    xid = 0
                if yid < 0:
                    yid = 0
                if xid >= n:
                    xid = n - 1
                if yid >= m:
                    yid = m - 1

                result[center + k + 1, i, j, :] = yid, xid

    result = result[center:, ...]
    return result


def lic_flow_numpy(vectors, len_pix=10, direction="both"):
    """

    """
    vectors = np.asarray(vectors)
    m, n, two = vectors.shape
    if two != 2:
        raise ValueError("Last dimension must be 2")

    origin = len_pix

    if direction == "both":
        step_id_range = chain(range(origin + 1, origin + 1 + len_pix),
                              reversed(range(0, origin)))
    elif direction == "forward":
        step_id_range = range(origin + 1, origin + 1 + len_pix)
    elif direction == "backward":
        step_id_range = reversed(range(0, origin))
    step_id_range = list(step_id_range)

    yxids = np.zeros((2 * len_pix + 1, m, n, 2), dtype=int)
    yxids.fill(-9999)
    yxids[origin, :, :, 0] = np.arange(m)[:, np.newaxis]
    yxids[origin, :, :, 1] = np.arange(n)[np.newaxis, :]

    fxys = 0.5 * np.ones_like(yxids, dtype=float)
    fxs = fxys[..., 0]
    fys = fxys[..., 1]

    for step_1 in step_id_range:

        sgn = np.sign(step_1-origin)
        step_0 = step_1 - sgn
        log.debug("From %d to %d" % (step_0, step_1))

        yids_0 = yxids[step_0, :, :, 0]
        xids_0 = yxids[step_0, :, :, 1]

        _t = vectors[yids_0, xids_0, :] * sgn
        txys = ((_t >= 0) - fxys[step_0]) / _t


        yxids[step_1] = yxids[step_0]
        # X step indices ix, jx
        ix, jx = np.where(txys[..., 0] < txys[..., 1])
        _vx = vectors[yids_0[ix, jx], xids_0[ix, jx], :] * sgn

        # Y step indices iy, jy
        iy, jy = np.where(txys[..., 0] >= txys[..., 1])
        _vy = vectors[yids_0[iy, jy], xids_0[iy, jy], :] * sgn

        # These are not merged as ix and iy do not generally have the same length
        fxys[step_1, ix, jx, 0] = _vx[..., 0] <= 0
        fxys[step_1, ix, jx, 1] = fxys[step_0, ix, jx, 1] + _vx[..., 1] * txys[ix, jx, 0]
        yxids[step_1, ix, jx, 1] += np.sign(_vx[..., 0]).astype(int)

        fxys[step_1, iy, jy, 0] = fxys[step_0, iy, jy, 0] + _vy[..., 0] * txys[iy, jy, 1]
        fxys[step_1, iy, jy, 1] = _vy[..., 1] <= 0
        yxids[step_1, iy, jy, 0] += np.sign(_vy[..., 1]).astype(int)

        # Don't go outside the domain
        yxids[step_1, :, :, 0] = np.clip(yxids[step_1, :, :, 0], 0, m-1)
        yxids[step_1, :, :, 1] = np.clip(yxids[step_1, :, :, 1], 0, n-1)

    # import pdb; pdb.set_trace()
    # if direction == "forward":
    return yxids[origin:]
    # else:
    #     return yxids[(*step_id_range), ...]


def plot_lic(ax, X, Y, U, V, **kwargs):
    S = (U ** 2 + V ** 2) ** .5
    ax.pcolormesh(X, Y, S, cmap='RdBu_r')
    result = add_lic(ax, X, Y, U, V, **kwargs)


def add_lic(ax, X, Y, U, V, length=None, alpha=.1, img=None, seed=None):
    """

    """
    if length is None:
        length = np.min(X.shape)//10
        log.debug('Using length %d' % length)

    if seed is not None:
        np.random.seed(seed)

    if img is None:
        img = np.random.random(size=U.shape)

    if not matplotlib.rcParams['image.composite_image']:
        log.warning("Please enable image.composite_image")

    with timing("Vectorised"):
        flow = lic_flow_numpy(np.stack((U, V), axis=-1), length)

    smear = np.average(img[flow[..., 0], flow[..., 1]], axis=0)

    img = ax.imshow(smear,
                    alpha=alpha,
                    extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)),
                    origin='lower',
                    # origin='upper',
                    interpolation='bicubic',
                    cmap='gray',
                    zorder=10)
    return img, flow


def test_reference(request):

    x = np.linspace(-2, 2, 72)
    y = np.linspace(-1, 1, 62)

    X, Y = np.meshgrid(x, y)

    U = -Y
    V = X

    np.random.seed(0)
    noise = np.random.random(size=U.shape)

    C = np.zeros_like(U)
    UV = np.stack((U, V), axis=-1)

    with timing("Initial"):
        result = lic_flow(UV, 30)
    with timing("Vectorised"):
        result1 = lic_flow_numpy(UV, 30, direction="forward")

    tx = result[..., 0]
    ty = result[..., 1]
    C = np.average(noise[tx, ty], axis=0)

    tx1 = result1[..., 0]
    ty1 = result1[..., 1]
    C1 = np.average(noise[tx1, ty1], axis=0)

    # Save and plot result
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax1 = plt.subplots()
        S = (U**2 + V**2)**.5
        strm = ax1.streamplot(X, Y, U, V, color=S)
        fig.colorbar(strm.lines)

        plt.savefig(pn.get())

        fig, axs = plt.subplots(1, 2)
        for ax, c in zip(axs, (C, C1)):
            ax.imshow(S, extent=(-1, 1, -1, 1), origin='lower', interpolation='bicubic', cmap='RdBu_r')
            ax.imshow(c, alpha=.1, extent=(-1, 1, -1, 1), origin='lower', interpolation='bicubic', cmap='gray')
            ax.grid(False)

        plt.savefig(pn.get())

    assert np.allclose(result, result1)



def test_lic(request):

    x = np.linspace(-2, 2, 272)
    y = np.linspace(-1, 1, 262)
    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(1, 1)
        plot_lic(ax, X, Y, U, V, length=30)
        plt.savefig(pn.get())
