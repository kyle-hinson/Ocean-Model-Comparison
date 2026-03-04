
import time
import numpy as np
from scipy.sparse import csr_matrix, spdiags

from _fp import flt32_t, flt64_t
from _fp import reals_t, index_t

from mat import inv_3x3

""" Staggered spatial discretisation on unstructured meshes.
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

def operators(mesh):
    """
    Implements various "TRiSK-style" numerical
    operators as sparse matrices:

    CELL-FLUX-SUMS: div. integration (* area)
    CELL-KITE-SUMS: dual-to-cell remapping
    CELL-WING-SUMS: edge-to-cell remapping
    CELL-EDGE-SUMS: edge-to-cell summation
    CELL-VERT-SUMS: vert-to-cell summation
    CELL-CURL-SUMS: curl integration (* area)
    CELL-DEL2-SUMS: cell del-squared (* area)

    EDGE-TAIL-SUMS: dual-to-edge remapping
    EDGE-WING-SUMS: cell-to-edge remapping
    EDGE-VERT-SUMS: vert-to-edge summation
    EDGE-CELL-SUMS: cell-to-edge summation
    EDGE-GRAD-NORM: edge gradient (normal)
    EDGE-GRAD-PERP: edge gradient (perpendicular)

    DUAL-FLUX-SUMS: div. integration (* area)
    DUAL-KITE-SUMS: cell-to-dual remapping
    DUAL-TAIL-SUMS: edge-to-dual remapping
    DUAL-CELL-SUMS: cell-to-dual summation
    DUAL-EDGE-SUMS: edge-to-dual summation
    DUAL-CURL-SUMS: curl integration (* area)    
    DUAL-DEL2-SUMS: dual del-squared (* area)

    QUAD-CURL-SUMS: curl integration (* area)

    EDGE-FLUX-PERP: reconstruct v (perpendicular)
    EDGE-LSQR-PERP: reconstruct v (perpendicular)

    (from norm. components)
    DUAL-LSQR-XNRM: reconstruct U (cartesian)
    DUAL-LSQR-YNRM: reconstruct Y (cartesian)
    DUAL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from perp. components)
    DUAL-LSQR-XPRP: reconstruct U (cartesian)
    DUAL-LSQR-YPRP: reconstruct Y (cartesian)
    DUAL-LSQR-ZPRP: reconstruct Z (cartesian)

    (from norm. components)
    CELL-LSQR-XNRM: reconstruct U (cartesian)
    CELL-LSQR-YNRM: reconstruct Y (cartesian)
    CELL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from norm. components)
    EDGE-LSQR-XNRM: reconstruct U (cartesian)
    EDGE-LSQR-YNRM: reconstruct Y (cartesian)
    EDGE-LSQR-ZNRM: reconstruct Z (cartesian)

    """

    class base: pass
    
    mats = base()

    ttic = time.time()

    # vector-calc. as sparse matrix operators
    mats.cell_flux_sums = cell_flux_sums(mesh)
    mats.cell_kite_sums = cell_kite_sums(mesh)
    mats.cell_wing_sums = cell_wing_sums(mesh)
    mats.cell_edge_sums = cell_edge_sums(mesh)
    mats.cell_vert_sums = cell_vert_sums(mesh)
   #mats.cell_curl_sums = cell_curl_sums(mesh)
    mats.cell_curl_sums = mats.cell_flux_sums  # equiv.

    mats.edge_tail_sums = edge_tail_sums(mesh)
    mats.edge_wing_sums = edge_wing_sums(mesh)
    mats.edge_vert_sums = edge_vert_sums(mesh)
    mats.edge_cell_sums = edge_cell_sums(mesh)
    mats.edge_grad_norm = edge_grad_norm(mesh)
    mats.edge_grad_perp = edge_grad_perp(mesh)

   #mats.cell_del2_sums = mats.cell_flux_sums \
   #                    * mats.edge_grad_norm

    mats.dual_flux_sums = dual_flux_sums(mesh)
    mats.dual_kite_sums = dual_kite_sums(mesh)
    mats.dual_tail_sums = dual_tail_sums(mesh)
    mats.dual_cell_sums = dual_cell_sums(mesh)
    mats.dual_edge_sums = dual_edge_sums(mesh)
   #mats.dual_curl_sums = dual_curl_sums(mesh)
    mats.dual_curl_sums = mats.dual_flux_sums  # equiv.

   #mats.dual_del2_sums = mats.dual_flux_sums \
   #                    * mats.edge_grad_perp

    # take curl on rhombi, a'la Gassmann
    mats.quad_curl_sums = mats.edge_vert_sums \
                        * mats.dual_curl_sums

    ttoc = time.time()
    print("-MATS done (sec):", round(ttoc - ttic, 2))
    
    ttic = time.time()

    # ensure flux reconstruction operator is exactly
    # skew-symmetric. Per Ringler et al, 2010, W_prp
    # is required to be anti-symmetric to ensure
    # energetically neutral PV fluxes: W_ij = -W_ji.
    # Due to floating-point round-off!
    mats.edge_flux_perp = edge_flux_perp(mesh)

    lmat = spdiags(
        1./mesh.edge.vlen, 
        0, mesh.edge.size, mesh.edge.size)

    dmat = spdiags(
        1.*mesh.edge.dlen, 
        0, mesh.edge.size, mesh.edge.size)

    wmat = dmat * mats.edge_flux_perp * lmat
    
    wmat = 0.5 * (wmat - wmat.transpose())

    lmat = spdiags(
        1.*mesh.edge.vlen,
        0, mesh.edge.size, mesh.edge.size)

    dmat = spdiags(
        1./mesh.edge.clen, 
        0, mesh.edge.size, mesh.edge.size)

    mats.edge_flux_perp = dmat * wmat * lmat
    
    ttoc = time.time()
    print("-WSYM done (sec):", round(ttoc - ttic, 2))

    ttic = time.time()

    # ensure remapping is always at worst dissipative
    # due to floating-point round-off!
    # this modifies the mesh data-structure in-place.
    crhs = np.ones(mesh.cell.size, 
                   dtype=flt64_t)
    erhs = np.ones(mesh.edge.size, 
                   dtype=flt64_t)
    vrhs = np.ones(mesh.vert.size, 
                   dtype=flt64_t)

    mesh.vert.area = (
        0.5 * mats.dual_kite_sums * crhs +
        0.5 * mats.dual_tail_sums * erhs
        )
        
    mesh.vert.area = reals_t(mesh.vert.area)
    
    mesh.edge.area = (
        0.5 * mats.edge_wing_sums * crhs +
        0.5 * mats.edge_tail_sums * vrhs
        )

    mesh.edge.area = reals_t(mesh.edge.area)
    
    mesh.quad = base()
    mesh.quad.area = mats.edge_vert_sums \
                   * mesh.vert.area

    mesh.quad.area = reals_t(mesh.quad.area)
       
    mesh.cell.area = (
        0.5 * mats.cell_wing_sums * erhs +
        0.5 * mats.cell_kite_sums * vrhs
        )

    mesh.cell.area = reals_t(mesh.cell.area)
 
    ttoc = time.time()
    print("-AREA done (sec):", round(ttoc - ttic, 2))
    
    ttic = time.time()
    
    # least-squares vector reconst. operators
    mats.dual_lsqr_xnrm, \
    mats.dual_lsqr_ynrm, \
    mats.dual_lsqr_znrm, \
    mats.dual_lsqr_xprp, \
    mats.dual_lsqr_yprp, \
    mats.dual_lsqr_zprp = dual_lsqr_fxyz(mesh)

    mats.cell_lsqr_xnrm, \
    mats.cell_lsqr_ynrm, \
    mats.cell_lsqr_znrm = cell_lsqr_fxyz(mesh)
    
    mats.edge_lsqr_xnrm, \
    mats.edge_lsqr_ynrm, \
    mats.edge_lsqr_znrm = edge_lsqr_fxyz(mesh)

    ttoc = time.time()
    print("-LSQR done (sec):", round(ttoc - ttic, 2))

    ttic = time.time()
    
    # build LSQR-<OP> from edge-wise reconstructions
    mats.edge_lsqr_perp = edge_lsqr_perp(mesh, mats)
    
    # operators for piecewise linear reconstructions
    # fe = fi + (xe - xi) * grad(f)
   #mats.edge_dual_reco = edge_dual_reco(mesh, mats)
   #mats.edge_cell_reco = edge_cell_reco(mesh, mats)
   
    ttoc = time.time()
    print("-RECO done (sec):", round(ttoc - ttic, 2))
   
    return mats


def cell_flux_sums(mesh):

#-- CELL-FLUX-SUMS: returns SUM(l_e * F_e) via sparse matrix
#-- operator OP. Use DIV(F) = OP * F, where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, edge] - 1
        eidx = mesh.cell.edge[mask, edge] - 1

        vlen = mesh.edge.vlen[eidx]

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        ivec = np.hstack((
            ivec, +cidx[flip], cidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -vlen[flip], vlen[okay]))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size))


def cell_curl_sums(mesh):

#-- CELL-CURL-SUMS: returns SUM(f_e * P_e) via sparse matrix
#-- operator OP. Use CURL(P) = OP * P where P is a vector of
#-- (perpendicular) fluxes for edges in the mesh.

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, edge] - 1
        eidx = mesh.cell.edge[mask, edge] - 1

        vlen = mesh.edge.vlen[eidx]

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        ivec = np.hstack((
            ivec, +cidx[flip], cidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -vlen[flip], vlen[okay]))
        
    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size))


def cell_kite_sums(mesh):

    return dual_kite_sums(mesh).transpose(copy=True).tocsr()


def cell_wing_sums(mesh):

    return edge_wing_sums(mesh).transpose(copy=True).tocsr()


def dual_tail_sums(mesh):

    return edge_tail_sums(mesh).transpose(copy=True).tocsr()


def cell_edge_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((
            xvec, np.ones(eidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size))


def cell_vert_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for vert in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > vert

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, vert] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.vert.size))


def edge_tail_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for vert in range(2):

        eidx = np.arange(0, mesh.edge.size)

        vidx = mesh.edge.vert[:, vert] - 1
        tail = mesh.edge.tail[:, vert]

        mask = vidx >= 0
        eidx = eidx[mask]
        vidx = vidx[mask]
        tail = tail[mask]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((xvec, tail))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.vert.size))


def edge_wing_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for cell in range(2):

        eidx = np.arange(0, mesh.edge.size)

        cidx = mesh.edge.cell[:, cell] - 1
        wing = mesh.edge.wing[:, cell]

        mask = cidx >= 0
        eidx = eidx[mask]
        cidx = cidx[mask]
        wing = wing[mask]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, wing))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.cell.size))


def edge_vert_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for vert in range(2):

        eidx = np.arange(0, mesh.edge.size)

        vidx = mesh.edge.vert[:, vert] - 1
        
        mask = vidx >= 0
        eidx = eidx[mask]
        vidx = vidx[mask]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.vert.size))


def edge_cell_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for cell in range(2):

        eidx = np.arange(0, mesh.edge.size)

        cidx = mesh.edge.cell[:, cell] - 1

        mask = cidx >= 0
        eidx = eidx[mask]
        cidx = cidx[mask]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((
            xvec, np.ones(cidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.cell.size))


def edge_grad_norm(mesh):

#-- EDGE-GRAD-NORM: returns (Q(j)-Q(i))/lij as sparse matrix
#-- operator OP. Use GRAD(Q) = OP * Q where Q is a vector of
#-- cell-centred scalars for all cells in the mesh.

    icel = mesh.edge.cell[:, 0] - 1
    jcel = mesh.edge.cell[:, 1] - 1

    eidx = np.arange(+0, mesh.edge.size)

    clen = mesh.edge.clen

    mask = np.logical_and.reduce((icel >= 0, 
                                  jcel >= 0)
        )
    icel = icel[mask]
    jcel = jcel[mask]
    eidx = eidx[mask]
    clen = clen[mask]

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((icel, jcel))
    xvec = np.concatenate(
        (-1.E+0 / clen, +1.E+0 / clen))
    
    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.cell.size))


def edge_grad_perp(mesh):

#-- EDGE-GRAD-PERP: returns (V(j)-V(i))/vij as sparse matrix
#-- operator OP. Use GRAD(V) = OP * V where V is a vector of
#-- node-centred scalars for all nodes in the mesh.

    ivrt = mesh.edge.vert[:, 0] - 1
    jvrt = mesh.edge.vert[:, 1] - 1

    eidx = np.arange(+0, mesh.edge.size)

    vlen = mesh.edge.vlen

    mask = np.logical_and.reduce((ivrt >= 0, 
                                  jvrt >= 0)
        )
    ivrt = ivrt[mask]
    jvrt = jvrt[mask]
    eidx = eidx[mask]
    vlen = vlen[mask]

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((ivrt, jvrt))
    xvec = np.concatenate(
        (-1.E+0 / vlen, +1.E+0 / vlen))
    
    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.vert.size))


def edge_flux_perp(mesh):

#-- EDGE-FLUX-PERP: returns f_perp, via the TRSK-type scheme
#-- for edges sandwiched between cells.
#-- Use f_perp = OP * f_nrm to reconstruct the perpendicular
#-- component of a vector field F. 

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(np.max(mesh.edge.topo)):

        mask = mesh.edge.topo > edge

        eidx = np.argwhere(mask).ravel()

        edsh = mesh.edge.edge[mask, edge] - 1

        wmul = mesh.edge.wmul[mask, edge]

        mask = edsh >= 0
        eidx = eidx[mask]
        edsh = edsh[mask]
        wmul = wmul[mask]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, edsh))
        xvec = np.hstack((xvec, wmul))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.edge.size))


def dual_flux_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1
        cidx = mesh.vert.cell[:, edge] - 1

        mask = eidx >= 0
        vidx = vidx[mask]
        eidx = eidx[mask]
        cidx = cidx[mask]

        clen = mesh.edge.clen[eidx]

        c1st = mesh.edge.cell[eidx, 0] - 1

        okay = cidx != c1st
        flip = cidx == c1st

        ivec = np.hstack((
            ivec, +vidx[flip], vidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -clen[flip], clen[okay]))
        
    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size))


def dual_curl_sums(mesh):

#-- DUAL-CURL-SUMS: returns SUM(lij * F_e) via sparse matrix
#-- operator OP. Use CURL(F) = OP * F where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1
        cidx = mesh.vert.cell[:, edge] - 1

        mask = eidx >= 0
        vidx = vidx[mask]
        eidx = eidx[mask]
        cidx = cidx[mask]

        clen = mesh.edge.clen[eidx]

        c1st = mesh.edge.cell[eidx, 0] - 1

        okay = cidx != c1st
        flip = cidx == c1st

        ivec = np.hstack((
            ivec, +vidx[flip], vidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -clen[flip], clen[okay]))
        
    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size))


def dual_kite_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1
        kite = mesh.vert.kite[:, cell]

        mask = cidx >= 0
        vidx = vidx[mask]
        cidx = cidx[mask]
        kite = kite[mask]

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, kite))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.cell.size))


def dual_cell_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1
        
        mask = cidx >= 0
        vidx = vidx[mask]
        cidx = cidx[mask]

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)),
        shape=(mesh.vert.size, mesh.cell.size))


def dual_edge_sums(mesh):

    xvec = np.array([], dtype=reals_t)
    ivec = np.array([], dtype=index_t)
    jvec = np.array([], dtype=index_t)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        mask = eidx >= 0
        vidx = vidx[mask]
        eidx = eidx[mask]

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=reals_t)))

    return csr_matrix((xvec, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size))


def dual_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. cell reconstructions

    ndir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T
    
    pdir = np.vstack((
        mesh.edge.xprp,
        mesh.edge.yprp, mesh.edge.zprp)).T

    dnrm = np.vstack((
        mesh.vert.xpos, 
        mesh.vert.ypos, mesh.vert.zpos)).T
    
    if (mesh.rsph is not None):
        dnrm = dnrm / mesh.rsph
    else:
        dnrm = np.zeros(
        (mesh.vert.size, 3), dtype=flt64_t)
        dnrm[:, 2] = flt64_t(1)
    
    Amat = np.zeros(
        (4, 3, mesh.vert.size), dtype=flt64_t)
    Bmat = np.zeros(
        (4, 3, mesh.vert.size), dtype=flt64_t)
        
    for edge in range(3):
    
        eidx = mesh.vert.edge[:, edge] - 1

        mask = eidx >= 0;
        eidx = eidx[mask]

        Amat[edge, :, mask] = ndir[eidx]
        Bmat[edge, :, mask] = pdir[eidx]
            
    Amat[3, :, :] = np.transpose(dnrm)
    Bmat[3, :, :] = np.transpose(dnrm)

    del ndir; del pdir; del dnrm

    matA = np.transpose(Amat, axes=(1, 0, 2))
    matB = np.transpose(Bmat, axes=(1, 0, 2))

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)
    Smat = np.einsum(
        "ik..., kj... -> ij...", matB, Bmat)

    Rinv, Rdet = inv_3x3(Rmat)
    Sinv, Sdet = inv_3x3(Smat)
    
    return Rinv, Rdet, matA, Sinv, Sdet, matB


def dual_lsqr_fxyz(mesh):

#-- dual reconstruction via "small" dual-based stencil

    Rinv, Rdet, matR, \
    Sinv, Sdet, matS = dual_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        mask = eidx >= 0;
        vidx = vidx[mask]
        eidx = eidx[mask]

        ivec.append(vidx); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(reals_t(xmul[mask]))
        ynrm.append(reals_t(ymul[mask]))
        znrm.append(reals_t(zmul[mask]))
        
        xmul = Sinv[0, 0, :] * matS[0, edge, :]
        xmul+= Sinv[0, 1, :] * matS[1, edge, :]
        xmul+= Sinv[0, 2, :] * matS[2, edge, :]
        xmul/= Sdet
        
        ymul = Sinv[1, 0, :] * matS[0, edge, :]
        ymul+= Sinv[1, 1, :] * matS[1, edge, :]
        ymul+= Sinv[1, 2, :] * matS[2, edge, :]
        ymul/= Sdet
        
        zmul = Sinv[2, 0, :] * matS[0, edge, :]
        zmul+= Sinv[2, 1, :] * matS[1, edge, :]
        zmul+= Sinv[2, 2, :] * matS[2, edge, :]
        zmul/= Sdet

        xprp.append(reals_t(xmul[mask]))
        yprp.append(reals_t(ymul[mask]))
        zprp.append(reals_t(zmul[mask]))
        
    ivec = np.asarray(
        np.concatenate(ivec), dtype=index_t)
    jvec = np.asarray(
        np.concatenate(jvec), dtype=index_t)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)
        
    xprp = np.concatenate(xprp)
    yprp = np.concatenate(yprp)
    zprp = np.concatenate(zprp)

    return csr_matrix((xnrm, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size)), \
           csr_matrix((ynrm, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size)), \
           csr_matrix((znrm, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size)), \
           csr_matrix((xprp, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size)), \
           csr_matrix((yprp, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size)), \
           csr_matrix((zprp, (ivec, jvec)), 
        shape=(mesh.vert.size, mesh.edge.size))


def cell_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. cell reconstructions

    edir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T

    cnrm = np.vstack((
        mesh.cell.xpos,
        mesh.cell.ypos, mesh.cell.zpos)).T
    
    if (mesh.rsph is not None):
        cnrm = cnrm / mesh.rsph
    else:
        cnrm = np.zeros(
        (mesh.cell.size, 3), dtype=flt64_t)
        cnrm[:, 2] = flt64_t(1)
       
    Amat = np.zeros(
        (np.max(mesh.cell.topo) + 1, 3, 
         mesh.cell.size), dtype=flt64_t)

    Wmat = np.zeros(
        (np.max(mesh.cell.topo) + 1,
         np.max(mesh.cell.topo) + 1,
         mesh.cell.size), dtype=flt64_t)
         
    wval = mesh.edge.area.copy().T
    wval[mesh.edge.mask] *= 2.0  # bnd edges

    for edge in range(np.max(mesh.cell.topo) + 1):

        Wmat[edge, edge, :] = mesh.cell.area.T

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1
        
        have = eidx >= 0;
        mask[cidx[np.logical_not(have)]] = False
        eidx = eidx[have]

        Wmat[edge, edge, mask] = wval[eidx]

        Amat[edge,    :, mask] = edir[eidx]
    
    Amat[-1, :, :] = np.transpose(cnrm)
    
    del edir; del cnrm

    matA = np.transpose(Amat, axes=(1, 0, 2))

    matA = np.einsum(
        "ik..., kj... -> ij...", matA, Wmat)

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)

    Rinv, Rdet = inv_3x3(Rmat)
    
    return Rinv, Rdet, matA


def cell_lsqr_fxyz(mesh):

#-- cell reconstruction via "large" cell-based stencil

    Rinv, Rdet, matR = cell_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1
        
        have = eidx >= 0;
        mask[cidx[np.logical_not(have)]] = False
        cidx = cidx[have]
        eidx = eidx[have]

        ivec.append(cidx); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(reals_t(xmul[mask]))
        ynrm.append(reals_t(ymul[mask]))
        znrm.append(reals_t(zmul[mask]))
        
    ivec = np.asarray(
        np.concatenate(ivec), dtype=index_t)
    jvec = np.asarray(
        np.concatenate(jvec), dtype=index_t)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)

    return csr_matrix((xnrm, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size)), \
           csr_matrix((ynrm, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size)), \
           csr_matrix((znrm, (ivec, jvec)), 
        shape=(mesh.cell.size, mesh.edge.size))


def edge_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. edge reconstructions

    ndir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T
    
    pdir = np.vstack((
        mesh.edge.xprp,
        mesh.edge.yprp, mesh.edge.zprp)).T

    enrm = np.vstack((
        mesh.edge.xpos,
        mesh.edge.ypos, mesh.edge.zpos)).T
    
    if (mesh.rsph is not None):
        enrm = enrm / mesh.rsph
    else:
        enrm = np.zeros(
        (mesh.edge.size, 3), dtype=flt64_t)
        enrm[:, 2] = flt64_t(1)

    Amat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 3,
         mesh.edge.size), dtype=flt64_t)

    Bmat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 3,
         mesh.edge.size), dtype=flt64_t)

    Wmat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 
         np.max(mesh.edge.topo) + 1, 
         mesh.edge.size), dtype=flt64_t)

    wval = mesh.edge.area.copy().T
    wval[mesh.edge.mask] *= 2.0  # bnd edges

    for edge in range(np.max(mesh.edge.topo) + 1):

        Wmat[edge, edge, :] = wval

    for edge in range(np.max(mesh.edge.topo) + 0):

        mask = mesh.edge.topo > edge

        enum = np.argwhere(mask).ravel()

        eidx = mesh.edge.edge[mask, edge] - 1

        have = eidx >= 0;
        mask[enum[np.logical_not(have)]] = False
        eidx = eidx[have]

        Wmat[edge, edge, mask] = wval[eidx]

        Amat[edge,    :, mask] = ndir[eidx]
        Bmat[edge,    :, mask] = pdir[eidx]
    
    Amat[-1, :, :] = np.transpose(enrm)
    Bmat[-1, :, :] = np.transpose(enrm)
    
    del ndir; del pdir; del enrm
    
    matA = np.transpose(Amat, axes=(1, 0, 2))
    matB = np.transpose(Bmat, axes=(1, 0, 2))

    matA = np.einsum(
        "ik..., kj... -> ij...", matA, Wmat)
    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)

    matB = np.einsum(
        "ik..., kj... -> ij...", matB, Wmat)
    Smat = np.einsum(
        "ik..., kj... -> ij...", matB, Bmat)

    Rinv, Rdet = inv_3x3(Rmat)
    Sinv, Sdet = inv_3x3(Smat)
    
    return Rinv, Rdet, matA, Sinv, Sdet, matB


def edge_lsqr_fxyz(mesh):

#-- edge reconstruction via "large" cell-based stencil

    Rinv, Rdet, matR, \
    Sinv, Sdet, matS = edge_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []
    
    for edge in range(np.max(mesh.edge.topo) + 0):

        mask = mesh.edge.topo > edge

        enum = np.argwhere(mask).ravel()

        eidx = mesh.edge.edge[mask, edge] - 1
        
        have = eidx >= 0;
        mask[enum[np.logical_not(have)]] = False
        enum = enum[have]
        eidx = eidx[have]
        
        ivec.append(enum); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(reals_t(xmul[mask]))
        ynrm.append(reals_t(ymul[mask]))
        znrm.append(reals_t(zmul[mask]))

    ivec = np.asarray(
        np.concatenate(ivec), dtype=index_t)
    jvec = np.asarray(
        np.concatenate(jvec), dtype=index_t)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)

    return csr_matrix((xnrm, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.edge.size)), \
           csr_matrix((ynrm, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.edge.size)), \
           csr_matrix((znrm, (ivec, jvec)), 
        shape=(mesh.edge.size, mesh.edge.size))


def edge_lsqr_perp(mesh, mats):

    xprp = mesh.edge.xprp
    xprp = spdiags(
        np.asarray(xprp, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)
    yprp = mesh.edge.yprp
    yprp = spdiags(
        np.asarray(yprp, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)
    zprp = mesh.edge.zprp
    zprp = spdiags(
        np.asarray(zprp, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)
  
    return (
        +1.000 * xprp * mats.edge_lsqr_xnrm +
        +1.000 * yprp * mats.edge_lsqr_ynrm +
        +1.000 * zprp * mats.edge_lsqr_znrm 
    )
    
    
def edge_lsqr_norm(mesh, mats):

    xnrm = mesh.edge.xnrm
    xnrm = spdiags(
        np.asarray(xnrm, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)
    ynrm = mesh.edge.ynrm   
    ynrm = spdiags(
        np.asarray(ynrm, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)
    znrm = mesh.edge.znrm
    znrm = spdiags(
        np.asarray(znrm, dtype=reals_t), 
            0, mesh.edge.size, mesh.edge.size)

    return (
        +1.000 * xnrm * mats.edge_lsqr_xprp +
        +1.000 * ynrm * mats.edge_lsqr_yprp +
        +1.000 * znrm * mats.edge_lsqr_zprp 
    )


def edge_dual_reco(mesh, mats):

#-- EDGE-DUAL-RECO: returns .5 * (xe - xv) * grad(F) part of
#-- edge reconstruction operator,
#-- with gradients estimated using "2-ring" stencil on duals.

    vrt1 = mesh.edge.vert[:, 0] - 1
    xev1 = mesh.edge.xpos - mesh.vert.xpos[vrt1]
    yev1 = mesh.edge.ypos - mesh.vert.ypos[vrt1]
    zev1 = mesh.edge.zpos - mesh.vert.zpos[vrt1]

    xev1 = np.asarray(xev1, dtype=reals_t)
    yev1 = np.asarray(yev1, dtype=reals_t)
    zev1 = np.asarray(zev1, dtype=reals_t)

    vrt2 = mesh.edge.vert[:, 1] - 1
    xev2 = mesh.edge.xpos - mesh.vert.xpos[vrt2]
    yev2 = mesh.edge.ypos - mesh.vert.ypos[vrt2]
    zev2 = mesh.edge.zpos - mesh.vert.zpos[vrt2]

    xev2 = np.asarray(xev2, dtype=reals_t)
    yev2 = np.asarray(yev2, dtype=reals_t)
    zev2 = np.asarray(zev2, dtype=reals_t)

    eidx = np.arange(0, mesh.edge.size)

    ivec = np.hstack((eidx, eidx))
    jvec = np.hstack((vrt1, vrt2))

    xmat = csr_matrix((
        np.hstack((xev1, xev2)), (ivec, jvec)))
    
    ymat = csr_matrix((
        np.hstack((yev1, yev2)), (ivec, jvec)))
    
    zmat = csr_matrix((
        np.hstack((zev1, zev2)), (ivec, jvec)))

    return (
        +0.500 * xmat * (mats.dual_lsqr_xprp * 
                         mats.edge_grad_perp) +
        +0.500 * ymat * (mats.dual_lsqr_yprp * 
                         mats.edge_grad_perp) +
        +0.500 * zmat * (mats.dual_lsqr_zprp * 
                         mats.edge_grad_perp)
    )


def edge_cell_reco(mesh, mats):

#-- EDGE-CELL-RECO: returns .5 * (xe - xc) * grad(F) part of
#-- edge reconstruction operator,
#-- with gradients estimated using "2-ring" stencil on cells.

    cel1 = mesh.edge.cell[:, 0] - 1
    xec1 = mesh.edge.xpos - mesh.cell.xpos[cel1]
    yec1 = mesh.edge.ypos - mesh.cell.ypos[cel1]
    zec1 = mesh.edge.zpos - mesh.cell.zpos[cel1]

    xec1 = np.asarray(xec1, dtype=reals_t)
    yec1 = np.asarray(yec1, dtype=reals_t)
    zec1 = np.asarray(zec1, dtype=reals_t)

    cel2 = mesh.edge.cell[:, 1] - 1
    xec2 = mesh.edge.xpos - mesh.cell.xpos[cel2]
    yec2 = mesh.edge.ypos - mesh.cell.ypos[cel2]
    zec2 = mesh.edge.zpos - mesh.cell.zpos[cel2]
    
    xec2 = np.asarray(xec2, dtype=reals_t)
    yec2 = np.asarray(yec2, dtype=reals_t)
    zec2 = np.asarray(zec2, dtype=reals_t)

    eidx = np.arange(0, mesh.edge.size)

    ivec = np.hstack((eidx, eidx))
    jvec = np.hstack((cel1, cel2))

    xmat = csr_matrix((
        np.hstack((xec1, xec2)), (ivec, jvec)))
    
    ymat = csr_matrix((
        np.hstack((yec1, yec2)), (ivec, jvec)))
    
    zmat = csr_matrix((
        np.hstack((zec1, zec2)), (ivec, jvec)))

    return (
        +0.500 * xmat * (mats.cell_lsqr_xnrm * 
                         mats.edge_grad_norm) +
        +0.500 * ymat * (mats.cell_lsqr_ynrm * 
                         mats.edge_grad_norm) +
        +0.500 * zmat * (mats.cell_lsqr_znrm * 
                         mats.edge_grad_norm)
    )
