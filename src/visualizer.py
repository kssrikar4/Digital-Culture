
def get_bioreactor_html(rpm):
    rotation_speed = (rpm / 60.0) * 2 * 3.14159 / 60.0 
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #0e1117; }}
            canvas {{ width: 100%; height: 100%; display: block; }}
        </style>
    </head>
    <body>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0e1117);
            const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(5, 4, 5);
            camera.lookAt(0, 1, 0);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            const ambientLight = new THREE.AmbientLight(0x404040, 2); 
            scene.add(ambientLight);
            const pointLight = new THREE.PointLight(0xffffff, 1, 100);
            pointLight.position.set(5, 5, 5);
            scene.add(pointLight);
            const pointLight2 = new THREE.PointLight(0x4040ff, 0.8, 100);
            pointLight2.position.set(-5, 2, -5);
            scene.add(pointLight2);
            const glassMaterial = new THREE.MeshPhysicalMaterial({{
                color: 0x88ccff,
                transparent: true,
                opacity: 0.3,
                metalness: 0.1,
                roughness: 0.1,
                side: THREE.DoubleSide
            }});
            const steelMaterial = new THREE.MeshStandardMaterial({{
                color: 0xaaaaaa,
                metalness: 0.8,
                roughness: 0.2
            }});
            const biomassMaterial = new THREE.MeshPhongMaterial({{
                color: 0x44aa44,
                transparent: true,
                opacity: 0.6,
                shininess: 30
            }});
            const tankGeometry = new THREE.CylinderGeometry(1.5, 1.5, 4, 32);
            const tank = new THREE.Mesh(tankGeometry, glassMaterial);
            tank.position.y = 2;
            scene.add(tank);
            const liquidGeo = new THREE.CylinderGeometry(1.45, 1.45, 3.0, 32);
            const liquid = new THREE.Mesh(liquidGeo, biomassMaterial);
            liquid.position.y = 1.5;
            scene.add(liquid);
            const agitatorGroup = new THREE.Group();
            const shaftGeo = new THREE.CylinderGeometry(0.1, 0.1, 4.5, 16);
            const shaft = new THREE.Mesh(shaftGeo, steelMaterial);
            shaft.position.y = 2.25;
            agitatorGroup.add(shaft);
            const bladeGeo = new THREE.BoxGeometry(0.8, 0.3, 0.05);
            for (let i = 0; i < 6; i++) {{
                const blade = new THREE.Mesh(bladeGeo, steelMaterial);
                blade.position.y = 0.5;
                blade.rotation.y = i * (Math.PI / 3);
                blade.translateX(0.5);
                agitatorGroup.add(blade);
            }}
            for (let i = 0; i < 6; i++) {{
                const blade = new THREE.Mesh(bladeGeo, steelMaterial);
                blade.position.y = 2.5;
                blade.rotation.y = i * (Math.PI / 3);
                blade.translateX(0.5); 
                agitatorGroup.add(blade);
            }}
            scene.add(agitatorGroup);
            const rotationSpeed = {rotation_speed};
            function animate() {{
                requestAnimationFrame(animate);
                agitatorGroup.rotation.y -= rotationSpeed;
                liquid.rotation.y -= rotationSpeed * 0.1;
                renderer.render(scene, camera);
            }}
            window.addEventListener('resize', onWindowResize, false);
            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    return html
