/* Shared Z-up viewers use Y-aligned cylinders for segment geometry. */
window.SkeletonGeometry = {
  cylinder: function(length, radius, color, opacity) {
    var material = new THREE.MeshLambertMaterial({color:color, transparent:opacity<1, opacity:opacity});
    return new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,length,12),material);
  },
  place: function(mesh,a,b) {
    var delta = new THREE.Vector3().subVectors(b,a);
    if(delta.length()<0.5) {mesh.visible=false;return;}
    mesh.visible=true;mesh.position.copy(a).addScaledVector(delta,0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());
  }
};
