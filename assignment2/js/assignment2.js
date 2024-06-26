import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const container = document.getElementById( 'container' );
var loader = new PLYLoader()

container.style.position = 'relative';
let renderer, stats, gui;
let scene, camera, controls, cube;

function initScene() {
    // Setup scene
	scene = new THREE.Scene();

    // Setup camera
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
	
    // Setup renderer 
	renderer = new THREE.WebGLRenderer();
	renderer.setSize( window.innerWidth, window.innerHeight );
	container.appendChild( renderer.domElement );

	controls = new OrbitControls( camera, renderer.domElement );
	controls.addEventListener( 'change', function() { renderer.render( scene, camera ); });
	
    loader.load('../assets/assignment2/results/fountain-P11/point-clouds/cloud_11_view.ply', function ( geometry ) {
        // geometry.computeVertexNormals();

        var material = new THREE.PointsMaterial( { size: 0.1 } );
        material.vertexColors = true;
		// cube = new THREE.Mesh( geometry, material );
        cube = new THREE.Points( geometry, material );

        scene.add( cube );
    })
	
	camera.position.z = 15;
}

function initSTATS() {
	stats = new Stats();
	stats.showPanel( 0 );
	stats.dom.style.position = 'absolute';
	stats.dom.style.top = 0;
	stats.dom.style.left = 0;
	container.appendChild( stats.dom );
}

function animate() {
	requestAnimationFrame( animate );

	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

	renderer.render( scene, camera );
	stats.update();
}


function onWindowResize() {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
};

window.addEventListener( 'resize', onWindowResize, false );

initScene();
initSTATS();
animate();