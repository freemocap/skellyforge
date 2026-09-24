/* Tapered bones: local -Y is proximal, +Y has one quarter of the base width. */
window.SkeletonGeometry = {
  cylinder: function(length, radius, color, opacity) {
    var material = new THREE.MeshLambertMaterial({color:color, transparent:opacity<1, opacity:opacity});
    return new THREE.Mesh(new THREE.CylinderGeometry(radius*0.25,radius,length,12),material);
  },
  place: function(mesh,a,b) {
    var delta = new THREE.Vector3().subVectors(b,a);
    var length=delta.length();
    if(length<0.5) {mesh.visible=false;return;}
    mesh.scale.y=length/mesh.geometry.parameters.height;
    mesh.visible=true;mesh.position.copy(a).addScaledVector(delta,0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());
  }
};
