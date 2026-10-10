from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyNeumann(MPMConstraintBase):
    """
    Weakly enforces a homogeneous Neumann (zero-flux) condition, d(field)/dn = 0, on a scalar
    (or per-component) field at a particle via a pure penalty formulation -- no additional
    (Lagrange multiplier) scalar variables are introduced, unlike ParticleLagrangianWeakDirichlet.

    The constraint penalizes the squared normal gradient of the field's current increment,
    1/2 * penaltyStiffness * (n . grad(field))^2, evaluated at the particle's own center using its
    interpolation gradient operator. Its stationarity condition adds

        P_A += penaltyStiffness * g * dg_dU_A
        K_AB += penaltyStiffness * dg_dU_A * dg_dU_B

    to the residual/stiffness, where g = n . grad(field) and dg_dU_A = n . grad[A, :] is exact and
    constant (g is linear in the nodal DOFs), so this linearization is exact, not an approximation.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticle
        The particle whose field's normal gradient is to be penalized.
    field
        The field this constraint is acting on (e.g. "plastic multiplier").
    normalVector
        The (assumed constant, e.g. domain boundary) outward normal direction at this particle.
    penaltyStiffness
        The penalty parameter. Larger values enforce the zero-flux condition more strictly, at the
        cost of stiffness-matrix conditioning.
    model
        The full MPMModel instance.
    """

    def __init__(
        self,
        name: str,
        constrainedParticle: BaseParticle,
        field: str,
        normalVector: np.ndarray,
        penaltyStiffness: float,
        model,
    ):
        self._name = name
        self._constrainedParticle = constrainedParticle
        self._field = field
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._normalVector = np.asarray(normalVector, dtype=float)
        self._penaltyStiffness = penaltyStiffness
        self._nodes = dict()
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
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list):
        pass

    def updateConnectivity(self, model):

        nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._constrainedParticle.kernelFunctions))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        K = V.reshape((self.nDof, self.nDof))
        p = self._constrainedParticle

        # grad[A, i] = d(shape function of node A)/dx_i, evaluated at this particle's center.
        grad = p.getInterpolationGradientVectorAtCenter()
        nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

        # dg_dU_A = n . grad[A, :]: since g = n . grad(field) = sum_A dg_dU_A * dU_A is linear in
        # the nodal DOFs, this is both the sensitivity used for the tangent and (dotted with the
        # current dU) the normal gradient itself.
        dg_dU = grad @ self._normalVector

        for i in range(self._fieldSize):
            dU_i = dU[i :: self._fieldSize]
            PExt_i = PExt[i :: self._fieldSize]
            K_ii = K[i :: self._fieldSize, i :: self._fieldSize]

            g = dg_dU @ dU_i[nodeIdcs]

            PExt_i[nodeIdcs] += self._penaltyStiffness * g * dg_dU
            K_ii[np.ix_(nodeIdcs, nodeIdcs)] += self._penaltyStiffness * np.outer(dg_dU, dg_dU)

            self.reactionFlux[i] = self._penaltyStiffness * g


def ParticlePenaltyNeumannOnParticleSetFactory(
    baseName: str,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    normalVector: np.ndarray,
    penaltyStiffness: float,
    model: MPMModel,
):
    """
    Factory function to create ParticlePenaltyNeumann constraints on a collection of particles,
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
    penaltyStiffness
        The penalty parameter.
    model
        The MPMModel instance.

    Returns
    -------
    dict
        A dictionary mapping constraint names to ParticlePenaltyNeumann instances.
    """

    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticlePenaltyNeumann(name, p, field, normalVector, penaltyStiffness, model)
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticlePenaltyNeumann(name, p, field, normalVector, penaltyStiffness, model)
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")
