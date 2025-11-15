import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BVHChannel(Enum):
    """BVH channel types"""
    XPOSITION = "Xposition"
    YPOSITION = "Yposition"
    ZPOSITION = "Zposition"
    XROTATION = "Xrotation"
    YROTATION = "Yrotation"
    ZROTATION = "Zrotation"


@dataclass
class BVHJoint:
    """Represents a joint in the BVH hierarchy"""
    name: str
    offset: np.ndarray  # 3D offset from parent
    channels: list[BVHChannel]
    children: list["BVHJoint"]
    is_end_site: bool = False


class BVHExporter:
    """
    Exports skeletal tracking data to BVH (BioVision Hierarchy) format.
    
    BVH files contain:
    1. HIERARCHY section defining skeleton structure
    2. MOTION section containing frame-by-frame animation data
    """
    
    def __init__(
        self,
        *,
        frame_rate: float = 30.0,
        scale_factor: float = 1.0,
        coordinate_system: str = "xyz"  # xyz or zyx depending on target software
    ):
        """
        Initialize BVH exporter.
        
        Parameters
        ----------
        frame_rate : float
            Frames per second for the animation
        scale_factor : float
            Scale factor to apply to position data
        coordinate_system : str
            Coordinate system convention ('xyz' or 'zyx')
        """
        self.frame_rate = frame_rate
        self.scale_factor = scale_factor
        self.coordinate_system = coordinate_system
        self.frame_time = 1.0 / frame_rate
    
    def export_from_actor(
        self,
        *,
        actor: Any,  # Would be Actor type but avoiding circular import
        output_path: Path | str,
        aspect_name: str = "body",
        use_rigid: bool = False
    ) -> None:
        """
        Export an Actor's motion data to BVH format.
        
        Parameters
        ----------
        actor : Actor
            Actor instance containing the motion data
        output_path : Path | str
            Output file path for the BVH file
        aspect_name : str
            Which aspect to export (default: "body")
        use_rigid : bool
            Whether to use rigid body data if available
        """
        output_path = Path(output_path)
        
        # Get the aspect
        if aspect_name not in actor.aspects:
            raise ValueError(f"Aspect '{aspect_name}' not found in actor. Available: {list(actor.aspects.keys())}")
        
        aspect = actor.aspects[aspect_name]
        
        # Get trajectory data
        if use_rigid and aspect.rigid_xyz is not None:
            trajectory = aspect.rigid_xyz
            logger.info(f"Using rigid body data for aspect '{aspect_name}'")
        elif aspect.xyz is not None:
            trajectory = aspect.xyz
            logger.info(f"Using raw tracking data for aspect '{aspect_name}'")
        else:
            raise ValueError(f"No trajectory data found for aspect '{aspect_name}'")
        
        # Get joint hierarchy from anatomical structure
        if aspect.anatomical_structure is None:
            raise ValueError(f"No anatomical structure defined for aspect '{aspect_name}'")
        
        joint_hierarchy = aspect.anatomical_structure.joint_hierarchy
        if joint_hierarchy is None:
            raise ValueError(f"No joint hierarchy defined for aspect '{aspect_name}'")
        
        # Build BVH hierarchy
        root_joint = self._build_bvh_hierarchy(
            trajectory_data=trajectory.as_array,
            joint_hierarchy=joint_hierarchy,
            landmark_names=trajectory.landmark_names
        )
        
        # Calculate rotations from positions
        motion_data = self._calculate_motion_data(
            trajectory_data=trajectory.as_array,
            joint_hierarchy=joint_hierarchy,
            landmark_names=trajectory.landmark_names,
            root_joint=root_joint
        )
        
        # Write BVH file
        self._write_bvh_file(
            output_path=output_path,
            root_joint=root_joint,
            motion_data=motion_data,
            num_frames=trajectory.num_frames
        )
        
        logger.info(f"Successfully exported BVH to {output_path}")
    
    def _build_bvh_hierarchy(
        self,
        *,
        trajectory_data: np.ndarray,
        joint_hierarchy: dict[str, list[str]],
        landmark_names: list[str]
    ) -> BVHJoint:
        """
        Build BVH joint hierarchy from tracking data.
        
        Parameters
        ----------
        trajectory_data : np.ndarray
            Shape (num_frames, num_markers, 3)
        joint_hierarchy : dict
            Joint parent-child relationships
        landmark_names : list
            Ordered list of landmark names
        
        Returns
        -------
        BVHJoint
            Root joint of the hierarchy
        """
        # Find root joint (one without parent)
        all_children = set()
        for children in joint_hierarchy.values():
            all_children.update(children)
        
        potential_roots = set(joint_hierarchy.keys()) - all_children
        
        if len(potential_roots) == 0:
            # If no clear root, use first joint in hierarchy
            root_name = list(joint_hierarchy.keys())[0]
        elif len(potential_roots) == 1:
            root_name = potential_roots.pop()
        else:
            # Multiple potential roots, pick the first one
            root_name = sorted(potential_roots)[0]
            logger.warning(f"Multiple potential root joints found: {potential_roots}, using '{root_name}'")
        
        # Get average positions for offset calculation
        avg_positions = np.nanmean(trajectory_data, axis=0)
        
        # Build hierarchy recursively
        root_joint = self._build_joint_recursive(
            joint_name=root_name,
            joint_hierarchy=joint_hierarchy,
            avg_positions=avg_positions,
            landmark_names=landmark_names,
            parent_position=None
        )
        
        return root_joint
    
    def _build_joint_recursive(
        self,
        *,
        joint_name: str,
        joint_hierarchy: dict[str, list[str]],
        avg_positions: np.ndarray,
        landmark_names: list[str],
        parent_position: np.ndarray | None
    ) -> BVHJoint:
        """
        Recursively build joint hierarchy.
        """
        # Get joint index
        if joint_name not in landmark_names:
            logger.warning(f"Joint '{joint_name}' not found in landmarks, skipping")
            raise ValueError(f"Joint '{joint_name}' not found in landmark names")
        
        joint_idx = landmark_names.index(joint_name)
        joint_position = avg_positions[joint_idx] * self.scale_factor
        
        # Calculate offset from parent
        if parent_position is None:
            # Root joint - use absolute position
            offset = joint_position
            channels = [
                BVHChannel.XPOSITION, BVHChannel.YPOSITION, BVHChannel.ZPOSITION,
                BVHChannel.ZROTATION, BVHChannel.XROTATION, BVHChannel.YROTATION
            ]
        else:
            offset = joint_position - parent_position
            channels = [
                BVHChannel.ZROTATION, BVHChannel.XROTATION, BVHChannel.YROTATION
            ]
        
        # Create joint
        joint = BVHJoint(
            name=joint_name,
            offset=offset,
            channels=channels,
            children=[]
        )
        
        # Add children
        if joint_name in joint_hierarchy:
            for child_name in joint_hierarchy[joint_name]:
                try:
                    child_joint = self._build_joint_recursive(
                        joint_name=child_name,
                        joint_hierarchy=joint_hierarchy,
                        avg_positions=avg_positions,
                        landmark_names=landmark_names,
                        parent_position=joint_position
                    )
                    joint.children.append(child_joint)
                except ValueError:
                    continue
        
        # Add end site if no children
        if not joint.children:
            end_site = BVHJoint(
                name=f"{joint_name}_End",
                offset=np.array([0, -10.0 * self.scale_factor, 0]),  # Default end site offset
                channels=[],
                children=[],
                is_end_site=True
            )
            joint.children.append(end_site)
        
        return joint
    
    def _calculate_motion_data(
        self,
        *,
        trajectory_data: np.ndarray,
        joint_hierarchy: dict[str, list[str]],
        landmark_names: list[str],
        root_joint: BVHJoint
    ) -> np.ndarray:
        """
        Calculate rotation data from position data.
        
        This is a simplified version that calculates rotations based on 
        bone orientations. For production use, consider using inverse kinematics.
        
        Parameters
        ----------
        trajectory_data : np.ndarray
            Shape (num_frames, num_markers, 3)
        joint_hierarchy : dict
            Joint parent-child relationships
        landmark_names : list
            Ordered list of landmark names
        root_joint : BVHJoint
            Root of the hierarchy
        
        Returns
        -------
        np.ndarray
            Motion data array with shape (num_frames, num_channels)
        """
        num_frames = trajectory_data.shape[0]
        
        # Count total channels
        total_channels = self._count_channels(root_joint)
        motion_data = np.zeros((num_frames, total_channels))
        
        for frame_idx in range(num_frames):
            frame_positions = trajectory_data[frame_idx] * self.scale_factor
            channel_idx = 0
            
            # Fill motion data for this frame
            channel_idx = self._fill_joint_motion(
                joint=root_joint,
                frame_positions=frame_positions,
                landmark_names=landmark_names,
                motion_data=motion_data,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                parent_position=None,
                parent_orientation=None
            )
        
        return motion_data
    
    def _fill_joint_motion(
        self,
        *,
        joint: BVHJoint,
        frame_positions: np.ndarray,
        landmark_names: list[str],
        motion_data: np.ndarray,
        frame_idx: int,
        channel_idx: int,
        parent_position: np.ndarray | None,
        parent_orientation: np.ndarray | None
    ) -> int:
        """
        Fill motion data for a joint and its children.
        
        Returns updated channel index.
        """
        if joint.is_end_site:
            return channel_idx
        
        # Get joint position
        if joint.name in landmark_names:
            joint_idx = landmark_names.index(joint.name)
            joint_position = frame_positions[joint_idx]
        else:
            # Use offset from parent if joint not tracked
            joint_position = parent_position + joint.offset if parent_position is not None else joint.offset
        
        # Fill channels for this joint
        for channel in joint.channels:
            if channel == BVHChannel.XPOSITION:
                motion_data[frame_idx, channel_idx] = joint_position[0]
            elif channel == BVHChannel.YPOSITION:
                motion_data[frame_idx, channel_idx] = joint_position[1]
            elif channel == BVHChannel.ZPOSITION:
                motion_data[frame_idx, channel_idx] = joint_position[2]
            elif channel in [BVHChannel.XROTATION, BVHChannel.YROTATION, BVHChannel.ZROTATION]:
                # Calculate rotation based on child orientation
                rotation = self._calculate_joint_rotation(
                    joint=joint,
                    joint_position=joint_position,
                    frame_positions=frame_positions,
                    landmark_names=landmark_names
                )
                
                if channel == BVHChannel.XROTATION:
                    motion_data[frame_idx, channel_idx] = rotation[0]
                elif channel == BVHChannel.YROTATION:
                    motion_data[frame_idx, channel_idx] = rotation[1]
                elif channel == BVHChannel.ZROTATION:
                    motion_data[frame_idx, channel_idx] = rotation[2]
            
            channel_idx += 1
        
        # Process children
        for child in joint.children:
            channel_idx = self._fill_joint_motion(
                joint=child,
                frame_positions=frame_positions,
                landmark_names=landmark_names,
                motion_data=motion_data,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                parent_position=joint_position,
                parent_orientation=None  # Would need to track orientation for full IK
            )
        
        return channel_idx
    
    def _calculate_joint_rotation(
        self,
        *,
        joint: BVHJoint,
        joint_position: np.ndarray,
        frame_positions: np.ndarray,
        landmark_names: list[str]
    ) -> np.ndarray:
        """
        Calculate joint rotation based on bone orientation.
        
        This is a simplified calculation. For production, use proper inverse kinematics.
        """
        # Find first child that's tracked
        for child in joint.children:
            if child.is_end_site:
                continue
            if child.name in landmark_names:
                child_idx = landmark_names.index(child.name)
                child_position = frame_positions[child_idx]
                
                # Calculate bone vector
                bone_vector = child_position - joint_position
                
                # Convert to Euler angles (simplified)
                rotation = self._vector_to_euler(bone_vector)
                return rotation
        
        # No tracked children, return zero rotation
        return np.zeros(3)
    
    def _vector_to_euler(self, vector: np.ndarray) -> np.ndarray:
        """
        Convert a direction vector to Euler angles.
        
        This is a simplified conversion. Consider using scipy.spatial.transform.Rotation
        for more robust conversions.
        """
        # Normalize vector
        length = np.linalg.norm(vector)
        if length < 1e-6:
            return np.zeros(3)
        
        normalized = vector / length
        
        # Calculate angles (simplified spherical coordinates)
        # These formulas assume Y-up coordinate system
        pitch = np.arcsin(-normalized[1]) * 180 / np.pi  # Rotation around X
        yaw = np.arctan2(normalized[0], normalized[2]) * 180 / np.pi  # Rotation around Y
        roll = 0.0  # No roll from direction vector alone
        
        return np.array([pitch, yaw, roll])
    
    def _count_channels(self, joint: BVHJoint) -> int:
        """Count total number of channels in hierarchy."""
        if joint.is_end_site:
            return 0
        
        count = len(joint.channels)
        for child in joint.children:
            count += self._count_channels(child)
        
        return count
    
    def _write_bvh_file(
        self,
        *,
        output_path: Path,
        root_joint: BVHJoint,
        motion_data: np.ndarray,
        num_frames: int
    ) -> None:
        """
        Write BVH file to disk.
        
        Parameters
        ----------
        output_path : Path
            Output file path
        root_joint : BVHJoint
            Root of the hierarchy
        motion_data : np.ndarray
            Motion data array
        num_frames : int
            Number of frames
        """
        with open(output_path, 'w') as f:
            # Write HIERARCHY section
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, root_joint, indent=0)
            
            # Write MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {self.frame_time:.6f}\n")
            
            # Write motion data
            for frame_idx in range(num_frames):
                frame_data = motion_data[frame_idx]
                frame_str = " ".join(f"{val:.6f}" for val in frame_data)
                f.write(f"{frame_str}\n")
    
    def _write_joint_hierarchy(
        self,
        f: Any,  # File handle
        joint: BVHJoint,
        indent: int
    ) -> None:
        """
        Write joint hierarchy to file.
        """
        indent_str = "\t" * indent
        
        if joint.is_end_site:
            f.write(f"{indent_str}End Site\n")
            f.write(f"{indent_str}{{\n")
            f.write(f"{indent_str}\tOFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
            f.write(f"{indent_str}}}\n")
        else:
            # Write joint header
            joint_type = "ROOT" if indent == 0 else "JOINT"
            f.write(f"{indent_str}{joint_type} {joint.name}\n")
            f.write(f"{indent_str}{{\n")
            
            # Write offset
            f.write(f"{indent_str}\tOFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
            
            # Write channels
            if joint.channels:
                channels_str = " ".join(ch.value for ch in joint.channels)
                f.write(f"{indent_str}\tCHANNELS {len(joint.channels)} {channels_str}\n")
            
            # Write children
            for child in joint.children:
                self._write_joint_hierarchy(f, child, indent + 1)
            
            f.write(f"{indent_str}}}\n")
