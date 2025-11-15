# Add this method to your Actor class in actor.py

from skellyforge.skellymodels.bvh_exporter.bvh_exporter import BVHExporter

def export_to_bvh(
    self,
    *,
    output_path: Path | str,
    aspect_name: str = "body",
    use_rigid: bool = False,
    frame_rate: float = 30.0,
    scale_factor: float = 1.0,
    coordinate_system: str = "xyz"
) -> None:
    """
    Export actor motion data to BVH format.
    
    Parameters
    ----------
    output_path : Path | str
        Output file path for the BVH file (e.g., 'output.bvh')
    aspect_name : str
        Which aspect to export (default: "body")
    use_rigid : bool
        Whether to use rigid body data if available (default: False)
    frame_rate : float
        Frames per second for the animation (default: 30.0)
    scale_factor : float
        Scale factor to apply to position data (default: 1.0)
    coordinate_system : str
        Coordinate system convention ('xyz' or 'zyx') (default: 'xyz')
    
    Raises
    ------
    ValueError
        If the specified aspect doesn't exist or lacks required data
    
    Examples
    --------
    >>> actor.export_to_bvh(output_path="motion_capture.bvh")
    >>> actor.export_to_bvh(
    ...     output_path="scaled_motion.bvh",
    ...     scale_factor=0.001,  # Convert mm to meters
    ...     frame_rate=60.0
    ... )
    """
    exporter = BVHExporter(
        frame_rate=frame_rate,
        scale_factor=scale_factor,
        coordinate_system=coordinate_system
    )
    
    exporter.export_from_actor(
        actor=self,
        output_path=output_path,
        aspect_name=aspect_name,
        use_rigid=use_rigid
    )
    
    logger.info(f"BVH file exported to {output_path}")
