import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.mjs";
import { PLYLoader } from "three/addons/loaders/PLYLoader.mjs";


const viewer = document.querySelector("[data-mesh-viewer]");

if (viewer) {
  const stage = viewer.querySelector("[data-mesh-stage]");
  const canvas = viewer.querySelector("[data-mesh-canvas]");
  const status = viewer.querySelector("[data-mesh-status]");
  const readout = viewer.querySelector("[data-mesh-readout]");
  const labels = viewer.querySelector("[data-mesh-labels]");
  const divider = viewer.querySelector("[data-mesh-divider]");
  const gtLabel = viewer.querySelector('[data-mesh-label="gt"]');
  const synlfLabel = viewer.querySelector('[data-mesh-label="synlf"]');
  const depthLegend = viewer.querySelector("[data-depth-legend]");
  const resetButton = viewer.querySelector("[data-reset-view]");
  const fullscreenButton = viewer.querySelector("[data-fullscreen]");
  const sceneSelect = viewer.querySelector("[data-scene-select]");
  const viewButtons = [...viewer.querySelectorAll("[data-view-mode]")];
  const colorButtons = [...viewer.querySelectorAll("[data-color-mode]")];
  const basePathRoot = viewer.dataset.basePath;
  const mobileLayout = window.matchMedia("(max-width: 680px)");

  let activeScene = viewer.dataset.initialScene;
  let viewMode = "compare";
  let colorMode = "appearance";
  let meshes = null;
  let loadedScene = null;
  let initialCamera = null;
  let loadRevision = 0;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0xffffff, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setScissorTest(true);

  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 50);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.35;
  controls.maxDistance = 6;
  controls.target.set(0, 0, 0);

  const gtScene = createScene();
  const synlfScene = createScene();
  const loader = new PLYLoader();
  const depthStops = [
    [0.0, new THREE.Color("#b31b4b")],
    [0.2, new THREE.Color("#ee6c4d")],
    [0.4, new THREE.Color("#f6d365")],
    [0.55, new THREE.Color("#c7e77f")],
    [0.72, new THREE.Color("#59b7a8")],
    [0.86, new THREE.Color("#3b82b4")],
    [1.0, new THREE.Color("#504aa0")],
  ];

  function createScene() {
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0xaab2bd, 1.9));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
    keyLight.position.set(-1.5, 2.2, 3.5);
    scene.add(keyLight);
    return scene;
  }

  function loadGeometry(url) {
    return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
  }

  function buildDepthColors(geometry) {
    const position = geometry.getAttribute("position");
    const colors = new Float32Array(position.count * 3);
    const color = new THREE.Color();
    for (let index = 0; index < position.count; index += 1) {
      const depth = THREE.MathUtils.clamp((-position.getZ(index) - 0.5) / 2.0, 0, 1);
      let upperIndex = depthStops.findIndex(([stop]) => stop >= depth);
      if (upperIndex <= 0) upperIndex = 1;
      const [lowerStop, lowerColor] = depthStops[upperIndex - 1];
      const [upperStop, upperColor] = depthStops[upperIndex];
      const blend = (depth - lowerStop) / Math.max(upperStop - lowerStop, 1e-6);
      color.copy(lowerColor).lerp(upperColor, blend);
      colors[index * 3] = color.r;
      colors[index * 3 + 1] = color.g;
      colors[index * 3 + 2] = color.b;
    }
    return new THREE.Float32BufferAttribute(colors, 3);
  }

  function prepareGeometry(geometry) {
    if (!geometry.index) throw new Error("PLY asset does not contain mesh faces.");
    geometry.computeBoundingBox();
    geometry.computeVertexNormals();
    geometry.setAttribute("appearanceColor", geometry.getAttribute("color").clone());
    geometry.setAttribute("depthColor", buildDepthColors(geometry));
    geometry.setAttribute("color", geometry.getAttribute("appearanceColor"));
    return geometry;
  }

  function makeMesh(geometry) {
    const materials = {
      appearance: new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.9,
        metalness: 0,
        side: THREE.DoubleSide,
      }),
      depth: new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
      }),
    };
    const mesh = new THREE.Mesh(geometry, materials.appearance);
    mesh.userData.materials = materials;
    mesh.frustumCulled = false;
    return mesh;
  }

  function frameMeshes(gtGeometry, synlfGeometry) {
    const bounds = new THREE.Box3();
    bounds.union(gtGeometry.boundingBox);
    bounds.union(synlfGeometry.boundingBox);
    const center = bounds.getCenter(new THREE.Vector3());
    gtGeometry.translate(-center.x, -center.y, -center.z);
    synlfGeometry.translate(-center.x, -center.y, -center.z);
    bounds.translate(center.clone().multiplyScalar(-1));
    const sphere = bounds.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.25);
    camera.near = Math.max(radius / 100, 0.005);
    camera.far = radius * 30;
    camera.position.set(radius * 0.08, radius * 0.04, radius * 2.4);
    controls.minDistance = radius * 0.55;
    controls.maxDistance = radius * 8;
    controls.target.set(0, 0, 0);
    camera.updateProjectionMatrix();
    controls.update();
    initialCamera = {
      position: camera.position.clone(),
      target: controls.target.clone(),
    };
  }

  function setColorMode(nextMode) {
    colorMode = nextMode;
    colorButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.colorMode === colorMode));
    });
    depthLegend.hidden = colorMode !== "depth";
    if (!meshes) return;
    meshes.forEach((mesh) => {
      const attributeName = colorMode === "depth" ? "depthColor" : "appearanceColor";
      mesh.geometry.setAttribute("color", mesh.geometry.getAttribute(attributeName));
      mesh.geometry.getAttribute("color").needsUpdate = true;
      mesh.material = mesh.userData.materials[colorMode];
    });
  }

  function setViewMode(nextMode) {
    viewMode = nextMode;
    viewButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.viewMode === viewMode));
    });
    const single = viewMode !== "compare";
    labels.classList.toggle("is-single", single);
    stage.classList.toggle("is-compare", !single);
    divider.hidden = single;
    gtLabel.hidden = viewMode === "synlf";
    synlfLabel.hidden = viewMode === "gt";
  }

  function setControlsEnabled(enabled) {
    [sceneSelect, ...viewButtons, ...colorButtons, resetButton].forEach((control) => {
      control.disabled = !enabled;
    });
  }

  function disposeMeshes() {
    if (!meshes) return;
    gtScene.remove(meshes[0]);
    synlfScene.remove(meshes[1]);
    meshes.forEach((mesh) => {
      mesh.geometry.dispose();
      Object.values(mesh.userData.materials).forEach((material) => material.dispose());
    });
    meshes = null;
  }

  async function loadScene(sceneId) {
    const revision = ++loadRevision;
    activeScene = sceneId;
    sceneSelect.value = activeScene;
    const sceneLabel = sceneSelect.selectedOptions[0].textContent.trim();
    setControlsEnabled(false);
    status.hidden = false;
    status.classList.remove("is-error");
    status.textContent = `Loading scene ${sceneLabel}...`;
    const basePath = `${basePathRoot}${sceneId}/`;

    try {
      const [manifest, gtGeometryRaw, synlfGeometryRaw] = await Promise.all([
        fetch(`${basePath}manifest.json`).then((response) => {
          if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
          return response.json();
        }),
        loadGeometry(`${basePath}structured-light-gt.ply`),
        loadGeometry(`${basePath}synlf.ply`),
      ]);
      if (revision !== loadRevision) {
        gtGeometryRaw.dispose();
        synlfGeometryRaw.dispose();
        return;
      }

      const gtGeometry = prepareGeometry(gtGeometryRaw);
      const synlfGeometry = prepareGeometry(synlfGeometryRaw);
      disposeMeshes();
      frameMeshes(gtGeometry, synlfGeometry);
      const gtMesh = makeMesh(gtGeometry);
      const synlfMesh = makeMesh(synlfGeometry);
      gtScene.add(gtMesh);
      synlfScene.add(synlfMesh);
      meshes = [gtMesh, synlfMesh];
      loadedScene = sceneId;
      setColorMode(colorMode);
      setControlsEnabled(true);
      status.hidden = true;
      const metrics = manifest.published_protocol_scene_metrics;
      const coverage = manifest.selection.gt_range_coverage_percent;
      readout.textContent = `Scene ${manifest.scene} · ${metrics.mae_mm.toFixed(2)} mm MAE · ${coverage.toFixed(1)}% valid GT`;
    } catch (error) {
      if (revision !== loadRevision) return;
      if (loadedScene) {
        activeScene = loadedScene;
        sceneSelect.value = activeScene;
      }
      setControlsEnabled(true);
      status.textContent = `Scene ${sceneLabel} could not be loaded.`;
      status.classList.add("is-error");
      console.error(error);
    }
  }

  function resetView() {
    if (!initialCamera) return;
    camera.position.copy(initialCamera.position);
    controls.target.copy(initialCamera.target);
    controls.update();
  }

  function viewportLayout(width, height) {
    if (viewMode === "gt") return [{ scene: gtScene, x: 0, y: 0, width, height }];
    if (viewMode === "synlf") return [{ scene: synlfScene, x: 0, y: 0, width, height }];
    if (mobileLayout.matches) {
      const half = Math.floor(height / 2);
      return [
        { scene: gtScene, x: 0, y: height - half, width, height: half },
        { scene: synlfScene, x: 0, y: 0, width, height: height - half },
      ];
    }
    const half = Math.floor(width / 2);
    return [
      { scene: gtScene, x: 0, y: 0, width: half, height },
      { scene: synlfScene, x: half, y: 0, width: width - half, height },
    ];
  }

  function render() {
    controls.update();
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const pixelRatio = renderer.getPixelRatio();
    const drawingWidth = Math.round(width * pixelRatio);
    const drawingHeight = Math.round(height * pixelRatio);
    if (canvas.width !== drawingWidth || canvas.height !== drawingHeight) {
      renderer.setSize(width, height, false);
    }
    renderer.setScissorTest(true);
    viewportLayout(width, height).forEach((viewport) => {
      camera.aspect = viewport.width / viewport.height;
      camera.zoom = mobileLayout.matches ? 1.45 : 1.12;
      camera.updateProjectionMatrix();
      renderer.setViewport(viewport.x, viewport.y, viewport.width, viewport.height);
      renderer.setScissor(viewport.x, viewport.y, viewport.width, viewport.height);
      renderer.render(viewport.scene, camera);
    });
    requestAnimationFrame(render);
  }

  viewButtons.forEach((button) => {
    button.addEventListener("click", () => setViewMode(button.dataset.viewMode));
  });
  colorButtons.forEach((button) => {
    button.addEventListener("click", () => setColorMode(button.dataset.colorMode));
  });
  sceneSelect.addEventListener("change", () => loadScene(sceneSelect.value));
  resetButton.addEventListener("click", resetView);
  fullscreenButton.addEventListener("click", async () => {
    if (document.fullscreenElement === stage) {
      await document.exitFullscreen();
    } else {
      await stage.requestFullscreen();
    }
  });

  setViewMode(viewMode);
  setColorMode(colorMode);
  loadScene(activeScene);
  render();
}
