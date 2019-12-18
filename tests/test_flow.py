import logging
log = logging.getLogger(__name__)
import numpy as np
import pytest

from tests import context  # Test context


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


def numpy_lic_flow(vectors, len_pix=10):
    vectors = np.asarray(vectors)
    m, n, two = vectors.shape
    if two != 2:
        raise ValueError

    yxids = np.zeros((2 * len_pix + 1, m, n, 2), dtype=np.int32)  # FIXME: int16?
    center = len_pix
    yxids[center, :, :, 0] = np.arange(m)[:, np.newaxis]
    yxids[center, :, :, 1] = np.arange(n)[np.newaxis, :]

    fxs = 0.5 * np.ones_like(yxids[:, :, :, 0])
    fys = 0.5 * np.ones_like(yxids[:, :, :, 1])
    txs = np.zeros_like(yxids[:, :, :, 0])
    tys = np.zeros_like(yxids[:, :, :, 1])


    for i in range(m):
        for j in range(n):

            # fx = fxs[center, i, j]
            # fy = fys[center, i, j]

            for k in range(center+1, center+1+len_pix):
                yid = yxids[k-1, i, j, 0]
                xid = yxids[k-1, i, j, 1]

                if vectors[yid, xid, 0] >= 0:
                    txs[k, i, j] = (1 - fxs[k-1, i, j]) / vectors[yid, xid, 0]
                else:
                    txs[k, i, j] = -fxs[k-1, i, j] / vectors[yid, xid, 0]
                if vectors[yid, xid, 1] >= 0:
                    tys[k, i, j] = (1 - fys[k-1, i, j]) / vectors[yid, xid, 1]
                else:
                    tys[k, i, j] = -fys[k-1, i, j] / vectors[yid, xid, 1]

                # Determine step direction.
                # THis updates xid, yid
                if txs[k, i, j] < tys[k, i, j]:
                    # X step
                    fys[k, i, j] = fys[k-1, i, j] + vectors[yid, xid, 1] * txs[k, i, j]
                    if vectors[yid, xid, 0] > 0:
                        xid += 1
                        fxs[k, i, j] = 0.
                    else:
                        xid -= 1
                        fxs[k, i, j] = 1.
                else:
                    # Y step
                    fxs[k, i, j] = fxs[k-1, i, j] + vectors[yid, xid, 0] * tys[k, i, j]
                    if vectors[yid, xid, 1] > 0:
                        yid += 1
                        fys[k, i, j] = 0.
                    else:
                        yid -= 1
                        fys[k, i, j] = 1.

                # del tx, ty

                # Don't go outside the domain
                if xid < 0:
                    xid = 0
                if yid < 0:
                    yid = 0
                if xid >= n:
                    xid = n - 1
                if yid >= m:
                    yid = m - 1

                yxids[k, i, j, :] = yid, xid

    yxids = yxids[center:, ...]
    return yxids



def test_lic(request):

    xo = np.linspace(-1, 1, 231)
    yo = np.linspace(-1, 1, 231)
    xc = 0.5 * (xo[:-1] + xo[1:])
    yc = 0.5 * (yo[:-1] + yo[1:])

    Xo, Yo = np.meshgrid(xo, yo)
    Xc, Yc = np.meshgrid(xc, yc)

    # print(Xo)
    U = -Yc
    V = Xc
    speed = np.sqrt(U ** 2 + V ** 2)

    noise = np.random.random(size=U.shape)

    C = np.zeros_like(U)
    UV = np.stack((U, V), axis=-1)

    result = numpy_lic_flow(UV, 30)

    for j in range(U.shape[0]):
        for k in range(U.shape[1]):
            tx = result[:, j, k, 0]
            ty = result[:, j, k, 1]
            C[j, k] = np.average(noise[tx, ty])

    #         if len(tx) > 0:
    #             plt.plot(Xc[ty, tx], Yc[ty, tx], '-')
    #             plt.plot(Xc[ty[0], tx[0]], Yc[ty[0], tx[0]], '-x')



    # Save and plot result
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax1 = plt.subplots()

        S = (U**2 + V**2)**.5
        # Varying color along a streamline
        # ax1.pcolormesh(Xo, Yo, U,)
        strm = ax1.streamplot(Xc, Yc, U, V, color=S)
        fig.colorbar(strm.lines)

        plt.savefig(pn.get())

        fig, [ax1, ax2] = plt.subplots(1, 2)
        # ax1.pcolormesh(C, alpha=.1, edgecolor=(1.0, 1.0, 1.0, 0.1), linewidth=0.0015625)
        ax1.pcolormesh(Xo, Yo, S, cmap='RdBu_r')
        ax1.set_aspect('equal')
        ax2.imshow(S, extent=(-1, 1, -1, 1), origin='lower', interpolation='bicubic', cmap='RdBu_r')
        ax2.imshow(C, alpha=.1, extent=(-1, 1, -1, 1), origin='lower', interpolation='bicubic', cmap='gray')
        ax2.grid(False)
        plt.savefig(pn.get())
