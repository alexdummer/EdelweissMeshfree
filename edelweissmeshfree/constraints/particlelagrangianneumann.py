from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleLagrangianNeumann(MPMConstraintBase):
    """
    Exactly enforces a homogeneous Neumann (zero-flux) condition, d(field)/dn = 0, on a scalar
    (or per-component) field at a particle via a Lagrange multiplier -- the exact-enforcement
    counterpart of ParticlePenaltyNeumann, mirroring how ParticleLagrangianWeakDirichlet relates
    to a penalty Dirichlet formulation.

    One additional scalar DOF (the multiplier) is introduced per constrained field component. Its
    converged value is directly the flux reaction needed to hold n . grad(field) = 0 -- since the
    target is always exactly zero (not a moving prescribed value), this Lagrange multiplier is the
    instantaneous flux itself, with no accumulation across time steps needed to interpret it.

    g_l = n . grad(field)_l = sum_A dg_l_dU_A * dU_A   (linear in the nodal DOFs, so this
    linearization is exact, not an approximation -- same as the penalty version).

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticle
        The particle whose field's normal gradient is to be constrained.
    field
        The field this constraint is acting on (e.g. "plastic multiplier").
    normalVector
        The (assumed constant, e.g. domain boundary) outward normal direction at this particle.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        constrainedParticle: BaseParticle,
        field: str,
        normalVector: np.ndarray,
        model,
    ):
        self._name = name
        self._constrainedParticle = constrainedParticle
        self._field = field
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._normalVector = np.asarray(normalVector, dtype=float)
        self._nodes = dict()
        self._nLagrangianMultipliers = self._fieldSize
        self.reactionFlux = np.zeros(self._fieldSize)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize + self._nLagrangianMultipliers

    @property
    def scalarVariables(self) -> list:
        return self._lagrangianMultipliers

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return self._nLagrangianMultipliers

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        self._lagrangianMultipliers = scalarVariables

    def updateConnectivity(self, model):

        nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._constrainedParticle.kernelFunctions))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        dU_U = dU_[: -self._nLagrangianMultipliers]
        dU_L = dU_[-self._nLagrangianMultipliers :]
        PExt_U = PExt[: -self._nLagrangianMultipliers]
        PExt_L = PExt[-self._nLagrangianMultipliers :]

        K = V.reshape((self.nDof, self.nDof))

        K_UL = K[: -self._nLagrangianMultipliers, -self._nLagrangianMultipliers :]
        K_LU = K[-self._nLagrangianMultipliers :, : -self._nLagrangianMultipliers]

        p = self._constrainedParticle

        # grad[A, i] = d(shape function of node A)/dx_i, evaluated at this particle's center.
        grad = p.getInterpolationGradientVectorAtCenter()
        nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

        # dg_dU_A = n . grad[A, :]: shared across field components, same as in the penalty version.
        dg_dU = grad @ self._normalVector

        for lag in range(self._nLagrangianMultipliers):
            i = lag  # one multiplier per field component

            dU_U_i = dU_U[i :: self._fieldSize]
            P_U_i = PExt_U[i :: self._fieldSize]
            K_UL_i = K_UL[i :: self._fieldSize, :]
            K_LU_i = K_LU[:, i :: self._fieldSize]

            dL_l = dU_L[lag]

            g_l = dg_dU @ dU_U_i[nodeIdcs]

            P_U_i[nodeIdcs] += dL_l * dg_dU
            PExt_L[lag] += g_l

            K_UL_i[nodeIdcs, lag] += dg_dU
            K_LU_i[lag, nodeIdcs] += dg_dU

            self.reactionFlux[i] = dL_l


def ParticleLagrangianNeumannOnParticleSetFactory(
    baseName: str,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    normalVector: np.ndarray,
    model: MPMModel,
):
    """
    Factory function to create ParticleLagrangianNeumann constraints on a collection of particles,
    all sharing the same (assumed constant, e.g. a straight domain boundary edge) outward normal.

    Parameters
    ----------
    baseName
        The base name for the constraints.
    particleCollection
        An iterable collection of particles or an EntityBasedSurface.
    field
        The field to be constrained.
    normalVector
        The (constant) outward normal direction shared by all particles in particleCollection.
    model
        The MPMModel instance.

    Returns
    -------
    dict
        A dictionary mapping constraint names to ParticleLagrangianNeumann instances.
    """

    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticleLagrangianNeumann(name, p, field, normalVector, model)
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticleLagrangianNeumann(name, p, field, normalVector, model)
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")
