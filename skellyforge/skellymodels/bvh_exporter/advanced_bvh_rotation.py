"""
Advanced rotation calculation for BVH export using inverse kinematics.
Add this as a module that can be imported by the BVHExporter.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple


class BVHRotationCalculator:
    """
    Calculates joint rotations from 3D positions for BVH export.
    Uses proper inverse kinematics and rotation decomposition.
    """
    
    def __init__(
        self,
        *,
        rotation_order: str = "ZXY",  # BVH standard rotation order
        up_axis: str = "Z",  # Which axis points up
        forward_axis: str = "Y"  # Which axis points forward
    ):
        """
        Initialize rotation calculator.
        
        Parameters
        ----------
        rotation_order : str
            Euler rotation order for BVH (typically "ZXY")
        up_axis : str
            Which axis points up in the coordinate system
        forward_axis : str
            Which axis points forward in the coordinate system
        """
        self.rotation_order = rotation_order
        self.up_axis = up_axis
        self.forward_axis = forward_axis
        
        # Create axis vectors
        self.up_vector = self._axis_to_vector(up_axis)
        self.forward_vector = self._axis_to_vector(forward_axis)
    
    def _axis_to_vector(self, axis: str) -> np.ndarray:
        """Convert axis name to unit vector."""
        axis = axis.upper()
        if axis == "X":
            return np.array([1, 0, 0], dtype=np.float64)
        elif axis == "Y":
            return np.array([0, 1, 0], dtype=np.float64)
        elif axis == "Z":
            return np.array([0, 0, 1], dtype=np.float64)
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def calculate_joint_rotation(
        self,
        *,
        joint_position: np.ndarray,
        child_position: np.ndarray,
        grandchild_position: np.ndarray | None = None,
        rest_pose_rotation: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Calculate joint rotation from positions.
        
        Parameters
        ----------
        joint_position : np.ndarray
            3D position of the joint
        child_position : np.ndarray
            3D position of the child joint
        grandchild_position : np.ndarray | None
            3D position of the grandchild (for twist calculation)
        rest_pose_rotation : np.ndarray | None
            Rest pose rotation to subtract (for relative rotation)
        
        Returns
        -------
        np.ndarray
            Euler angles in degrees [X, Y, Z rotation]
        """
        # Calculate bone direction
        bone_vector = child_position - joint_position
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 1e-6:
            # Bones too close, return identity rotation
            return np.zeros(3)
        
        bone_direction = bone_vector / bone_length
        
        # Calculate rotation matrix to align with bone
        rotation_matrix = self._calculate_rotation_matrix(
            bone_direction=bone_direction,
            grandchild_position=grandchild_position,
            joint_position=joint_position,
            child_position=child_position
        )
        
        # Apply rest pose correction if provided
        if rest_pose_rotation is not None:
            rest_rotation = Rotation.from_euler(
                self.rotation_order.lower(),
                rest_pose_rotation,
                degrees=True
            )
            rotation_matrix = rotation_matrix @ rest_rotation.inv().as_matrix()
        
        # Convert to Euler angles
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler(self.rotation_order.lower(), degrees=True)
        
        return euler_angles
    
    def _calculate_rotation_matrix(
        self,
        *,
        bone_direction: np.ndarray,
        grandchild_position: np.ndarray | None,
        joint_position: np.ndarray,
        child_position: np.ndarray
    ) -> np.ndarray:
        """
        Calculate rotation matrix to align coordinate system with bone.
        """
        # Primary axis aligns with bone
        primary_axis = bone_direction
        
        # Calculate secondary axis
        if grandchild_position is not None:
            # Use plane defined by joint-child-grandchild
            to_grandchild = grandchild_position - child_position
            
            # Project to perpendicular plane
            perpendicular = to_grandchild - np.dot(to_grandchild, primary_axis) * primary_axis
            perpendicular_length = np.linalg.norm(perpendicular)
            
            if perpendicular_length > 1e-6:
                secondary_axis = perpendicular / perpendicular_length
            else:
                # Grandchild is colinear, use default up
                secondary_axis = self._get_perpendicular_vector(primary_axis)
        else:
            # No grandchild, use default perpendicular
            secondary_axis = self._get_perpendicular_vector(primary_axis)
        
        # Calculate third axis
        tertiary_axis = np.cross(primary_axis, secondary_axis)
        tertiary_axis = tertiary_axis / np.linalg.norm(tertiary_axis)
        
        # Recalculate secondary to ensure orthogonality
        secondary_axis = np.cross(tertiary_axis, primary_axis)
        
        # Build rotation matrix based on axis configuration
        if self.forward_axis == "Z":
            rotation_matrix = np.column_stack([
                tertiary_axis,  # X axis
                secondary_axis,  # Y axis  
                primary_axis     # Z axis (forward)
            ])
        elif self.forward_axis == "X":
            rotation_matrix = np.column_stack([
                primary_axis,    # X axis (forward)
                secondary_axis,  # Y axis
                tertiary_axis    # Z axis
            ])
        else:  # Y forward
            rotation_matrix = np.column_stack([
                tertiary_axis,   # X axis
                primary_axis,    # Y axis (forward)
                secondary_axis   # Z axis
            ])
        
        return rotation_matrix
    
    def _get_perpendicular_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Get a vector perpendicular to the input vector.
        Prefers to use the up axis when possible.
        """
        # Try to use up vector
        perpendicular = self.up_vector - np.dot(self.up_vector, vector) * vector
        perpendicular_length = np.linalg.norm(perpendicular)
        
        if perpendicular_length < 1e-6:
            # Vector is parallel to up, use forward instead
            perpendicular = self.forward_vector - np.dot(self.forward_vector, vector) * vector
            perpendicular_length = np.linalg.norm(perpendicular)
            
            if perpendicular_length < 1e-6:
                # Still parallel, use any perpendicular
                if abs(vector[0]) < 0.9:
                    perpendicular = np.array([1, 0, 0], dtype=np.float64)
                else:
                    perpendicular = np.array([0, 1, 0], dtype=np.float64)
                
                perpendicular = perpendicular - np.dot(perpendicular, vector) * vector
                perpendicular_length = np.linalg.norm(perpendicular)
        
        return perpendicular / perpendicular_length
    
    def calculate_rotation_sequence(
        self,
        *,
        positions: np.ndarray,
        joint_chain: list[int],
        frame_idx: int
    ) -> list[np.ndarray]:
        """
        Calculate rotations for a chain of joints.
        
        Parameters
        ----------
        positions : np.ndarray
            All joint positions for this frame
        joint_chain : list[int]
            Indices of joints in the chain
        frame_idx : int
            Current frame index
        
        Returns
        -------
        list[np.ndarray]
            List of rotation arrays for each joint in chain
        """
        rotations = []
        
        for i, joint_idx in enumerate(joint_chain):
            joint_pos = positions[joint_idx]
            
            # Get child position
            if i + 1 < len(joint_chain):
                child_idx = joint_chain[i + 1]
                child_pos = positions[child_idx]
            else:
                # Last joint in chain, use offset
                child_pos = joint_pos + np.array([0, -10, 0])  # Default end effector
            
            # Get grandchild position for better orientation
            grandchild_pos = None
            if i + 2 < len(joint_chain):
                grandchild_idx = joint_chain[i + 2]
                grandchild_pos = positions[grandchild_idx]
            
            # Calculate rotation
            rotation = self.calculate_joint_rotation(
                joint_position=joint_pos,
                child_position=child_pos,
                grandchild_position=grandchild_pos
            )
            
            rotations.append(rotation)
        
        return rotations
    
    def smooth_rotations(
        self,
        *,
        rotations: np.ndarray,
        window_size: int = 5,
        preserve_extremes: bool = True
    ) -> np.ndarray:
        """
        Apply temporal smoothing to rotation data to reduce jitter.
        
        Parameters
        ----------
        rotations : np.ndarray
            Shape (num_frames, 3) - Euler angles over time
        window_size : int
            Size of the smoothing window
        preserve_extremes : bool
            Whether to preserve rotation extremes (for precise movements)
        
        Returns
        -------
        np.ndarray
            Smoothed rotations
        """
        if window_size <= 1:
            return rotations
        
        num_frames = rotations.shape[0]
        smoothed = np.zeros_like(rotations)
        
        # Convert to quaternions for better interpolation
        quaternions = []
        for i in range(num_frames):
            rot = Rotation.from_euler(
                self.rotation_order.lower(),
                rotations[i],
                degrees=True
            )
            quaternions.append(rot.as_quat())
        
        quaternions = np.array(quaternions)
        
        # Apply smoothing
        half_window = window_size // 2
        
        for i in range(num_frames):
            start_idx = max(0, i - half_window)
            end_idx = min(num_frames, i + half_window + 1)
            
            # Average quaternions in window
            window_quats = quaternions[start_idx:end_idx]
            
            if preserve_extremes and len(window_quats) > 2:
                # Check if current frame is an extreme
                current_magnitude = np.linalg.norm(rotations[i])
                window_magnitudes = [
                    np.linalg.norm(rotations[j])
                    for j in range(start_idx, end_idx)
                ]
                
                if (current_magnitude == max(window_magnitudes) or
                    current_magnitude == min(window_magnitudes)):
                    # Preserve extreme, use smaller window
                    window_quats = quaternions[max(0, i-1):min(num_frames, i+2)]
            
            # Spherical linear interpolation
            avg_quat = self._average_quaternions(window_quats)
            
            # Convert back to Euler
            avg_rotation = Rotation.from_quat(avg_quat)
            smoothed[i] = avg_rotation.as_euler(
                self.rotation_order.lower(),
                degrees=True
            )
        
        return smoothed
    
    def _average_quaternions(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Average quaternions using spherical linear interpolation.
        """
        # Simple averaging method
        # For better results, could use Markley's method
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        return avg_quat


# Helper function to integrate with BVHExporter
def create_advanced_rotation_calculator() -> BVHRotationCalculator:
    """
    Create a rotation calculator with standard BVH settings.
    
    Returns
    -------
    BVHRotationCalculator
        Configured calculator instance
    """
    return BVHRotationCalculator(
        rotation_order="ZXY",  # Standard BVH order
        up_axis="Y",
        forward_axis="Z"
    )