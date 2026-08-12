"""Engine test suite for skellyforge.

Covers the kinematics math: quaternion algebra, the parent-relative composition
convention, coordinate frames and Kabsch alignment, the orientation solver,
critical damping, and the standard human model's validators.

Specification:
``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md``

The governing rule from that spec: **a test must be able to fail for the reason it
exists.** The engine's failure mode is that it produces *plausible* numbers — a
reversed quaternion product is still unit, a sign-flipped basis is still
orthonormal — so these tests pin conventions and invariants rather than checking
that functions run. No test here may pass under both operand orders, both
handedness conventions, or both quaternion component orders.
"""
