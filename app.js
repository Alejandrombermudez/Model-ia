// ================================================================
//  ⚙️  CONFIGURACIÓN — Extraída del código fuente original
// ================================================================

const INPUT_SIZE = 224;

// Clases índice 0-4 — fuente: app_identificacion.py línea 33
const CLASSES = ['Abarco costeño', 'Achapo', 'Balso', 'Yarumo', 'Yopo'];

// Expertos por modelo — fuente: app_identificacion.py líneas 34-35
const EXPERTOS_EFF = ['Abarco costeño', 'Yopo'];
const EXPERTOS_RES = ['Balso', 'Yarumo'];

// Rutas a los modelos ONNX convertidos
const MODEL_PATHS = {
  eff: 'https://huggingface.co/AlejandroBermudez123/especies-arboreas/resolve/main/efficientnet.onnx',
  res: 'https://huggingface.co/AlejandroBermudez123/especies-arboreas/resolve/main/resnet.onnx',
};

// Canales — ORDEN CRÍTICO: B, G, R, RE, NIR
// Fuente: prepare_tensor() → np.stack([b, g, r, re, nir], axis=-1)
const CHANNELS = [
  { id: 'b',   label: 'B',   desc: 'Azul',       color: '#3b82f6', ext: 'MS_B'   },
  { id: 'g',   label: 'G',   desc: 'Verde',       color: '#22c55e', ext: 'MS_G'   },
  { id: 'r',   label: 'R',   desc: 'Rojo',        color: '#ef4444', ext: 'MS_R'   },
  { id: 're',  label: 'RE',  desc: 'Red Edge',    color: '#f97316', ext: 'MS_RE'  },
  { id: 'nir', label: 'NIR', desc: 'Infrarrojo',  color: '#a855f7', ext: 'MS_NIR' },
];

// ================================================================
//  Estado
// ================================================================
const state = {
  files:       {},   // { channelId: File }
  dims:        {},   // { channelId: { w, h } }
  sessions:    { eff: null, res: null },
  modelsReady: { eff: false, res: false },
};

// ================================================================
//  Arranque
// ================================================================
document.addEventListener('DOMContentLoaded', () => {
  renderGrid();
  preloadModels();
  document.getElementById('btn-identify').addEventListener('click', runPrediction);
});

// ================================================================
//  Render grid de canales
// ================================================================
function renderGrid() {
  const grid = document.getElementById('channel-grid');
  grid.innerHTML = '';

  CHANNELS.forEach(ch => {
    const card = document.createElement('div');
    card.className = 'flex flex-col items-center gap-2';
    card.innerHTML = `
      <div class="flex items-center gap-1.5 w-full">
        <span class="channel-badge" style="background:${ch.color}">${ch.label}</span>
        <span class="text-xs text-stone-500 font-medium leading-tight">${ch.desc}</span>
      </div>
      <div class="upload-zone w-full" id="zone-${ch.id}"
           tabindex="0" role="button"
           aria-label="Canal ${ch.label}">
        <input type="file" accept=".tif,.tiff,image/tiff,image/*"
               id="input-${ch.id}" style="display:none">
        <div class="upload-placeholder">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2
                 0 012.828 0L20 14M8 9a2 2 0 100-4 2 2 0 000 4zm-4 12h16a2 2 0
                 002-2V6a2 2 0 00-2-2H4a2 2 0 00-2 2v14a2 2 0 002 2z"/>
          </svg>
          <span>Clic o arrastra</span>
          <span class="text-slate-300" style="font-size:0.65rem">.TIF / .JPG</span>
        </div>
        <img class="preview-img hidden" alt="Preview ${ch.label}" />
        <div class="change-overlay">Cambiar imagen</div>
      </div>
      <div class="text-xs text-slate-400 text-center h-4 leading-4" id="dims-${ch.id}"></div>
    `;
    grid.appendChild(card);

    const zone  = card.querySelector('.upload-zone');
    const input = card.querySelector('input[type=file]');

    zone.addEventListener('click',   ()  => input.click());
    zone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
    input.addEventListener('change', (e) => { if (e.target.files[0]) handleFile(ch.id, e.target.files[0]); });

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) handleFile(ch.id, file);
    });
  });
}

// ================================================================
//  Manejo de archivo
// ================================================================
function handleFile(channelId, file) {
  const isTif  = /\.(tif|tiff)$/i.test(file.name);
  const zone   = document.getElementById(`zone-${channelId}`);
  const ch     = CHANNELS.find(c => c.id === channelId);

  state.files[channelId] = file;

  const placeholder = zone.querySelector('.upload-placeholder');
  const preview     = zone.querySelector('.preview-img');

  if (!isTif) {
    // JPG/PNG — preview nativo del navegador
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      state.dims[channelId] = { w: img.naturalWidth, h: img.naturalHeight };
      preview.src = url;
      preview.classList.remove('hidden');
      placeholder.classList.add('hidden');
      zone.classList.add('loaded');
      zone.style.borderColor = ch.color;
      document.getElementById(`dims-${channelId}`).textContent =
        `${img.naturalWidth} x ${img.naturalHeight}`;
      validate();
    };
    img.src = url;
  } else {
    // TIF — leer dimensiones con GeoTIFF.js
    file.arrayBuffer().then(buf => GeoTIFF.fromArrayBuffer(buf))
      .then(t => t.getImage())
      .then(img => {
        const w = img.getWidth(), h = img.getHeight();
        state.dims[channelId] = { w, h };

        // Preview visual: color del canal sobre fondo neutro
        preview.classList.add('hidden');
        placeholder.classList.add('hidden');
        zone.classList.add('loaded');
        zone.style.borderColor = ch.color;
        zone.style.background  = ch.color + '18';

        let label = zone.querySelector('.tif-label');
        if (!label) {
          label = document.createElement('div');
          label.className = 'tif-label absolute inset-0 flex flex-col items-center justify-center text-center p-2';
          label.style.pointerEvents = 'none';
          zone.appendChild(label);
        }
        const shortName = file.name.length > 16 ? file.name.slice(0, 13) + '...' : file.name;
        label.innerHTML = `
          <div style="font-size:1.6rem">&#128225;</div>
          <div style="font-size:0.65rem;font-weight:700;color:${ch.color};margin-top:4px">${shortName}</div>
          <div style="font-size:0.6rem;color:#94a3b8">${w} x ${h}</div>`;

        document.getElementById(`dims-${channelId}`).textContent = `${w} x ${h}`;
        validate();
      }).catch(err => alert(`No se pudo leer el TIF del canal ${ch.label}:\n${err.message}`));
  }
}

// ================================================================
//  Validación
// ================================================================
function validate() {
  const bar  = document.getElementById('validation-bar');
  const btn  = document.getElementById('btn-identify');
  const n    = Object.keys(state.files).length;
  const nd   = Object.keys(state.dims).length;

  bar.classList.remove('hidden');
  bar.className = 'mt-5 rounded-xl p-3 text-sm';

  if (n < CHANNELS.length) {
    const miss = CHANNELS.filter(ch => !state.files[ch.id]).map(ch =>
      `<span class="font-semibold">${ch.label}</span>`).join(', ');
    bar.className = 'mt-4 rounded-xl p-3 text-xs bg-amber-50 border border-amber-200 text-amber-900';
    bar.innerHTML = `<span class="font-semibold">${n}/${CHANNELS.length}</span> bandas cargadas — Faltan: ${miss}`;
    btn.disabled = true;
    return;
  }

  if (nd < CHANNELS.length) {
    bar.className = 'mt-4 rounded-xl p-3 text-xs bg-amber-50 border border-amber-200 text-amber-900';
    bar.innerHTML = 'Leyendo dimensiones…';
    btn.disabled = true;
    return;
  }

  const dimVals = Object.values(state.dims);
  const refW = dimVals[0].w, refH = dimVals[0].h;
  const ok   = dimVals.every(d => d.w === refW && d.h === refH);

  if (!ok) {
    const table = CHANNELS.map(ch => {
      const d = state.dims[ch.id];
      const match = d && d.w === refW && d.h === refH;
      return `<div class="text-center">
        <div class="font-bold" style="color:${ch.color}">${ch.label}</div>
        <div class="${match ? 'text-stone-600' : 'text-red-600 font-semibold'}">${d ? `${d.w}×${d.h}` : '—'}</div>
      </div>`;
    }).join('');
    bar.className = 'mt-4 rounded-xl p-3 text-xs bg-red-50 border border-red-200 text-red-700';
    bar.innerHTML = `<div class="font-semibold mb-2">Las bandas tienen dimensiones distintas</div>
      <div class="grid grid-cols-5 gap-2">${table}</div>
      <div class="mt-2 text-red-500">Verifica que todas las bandas sean del mismo vuelo.</div>`;
    btn.disabled = true;
    return;
  }

  bar.className = 'mt-4 rounded-xl p-3 text-xs bg-emerald-50 border border-emerald-200 text-emerald-700';
  bar.innerHTML = `<span class="font-semibold">5 bandas cargadas</span> · Dimensiones: ${refW} × ${refH} px · Listo para identificar`;
  btn.disabled = false;
}

// ================================================================
//  Carga de sesiones ONNX (en paralelo al inicio)
// ================================================================
async function preloadModels() {
  const setStatus = (key, text, state) => {
    const el = document.getElementById(`status-${key}`);
    if (!el) return;
    el.textContent = text;
    el.className = 'model-status text-xs font-medium px-2 py-1 rounded-full border ' +
      (state === 'ready' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
       state === 'error' ? 'bg-red-50 text-red-600 border-red-200' :
                          'bg-amber-50 text-amber-600 border-amber-200');
  };

  const loadOne = async (key) => {
    setStatus(key, 'Cargando...', 'text-amber-500');
    try {
      state.sessions[key] = await ort.InferenceSession.create(MODEL_PATHS[key], {
        executionProviders: ['wasm']
      });
      state.modelsReady[key] = true;
      setStatus(key, '✓ Listo', 'ready');
      console.log(`Modelo ${key} cargado. Input: ${state.sessions[key].inputNames}`);
    } catch (e) {
      state.modelsReady[key] = false;
      setStatus(key, 'No encontrado', 'error');
      console.warn(`Modelo ${key} no disponible:`, e.message);
    }
  };

  await Promise.all([loadOne('eff'), loadOne('res')]);
}

// ================================================================
//  Convertir un archivo (TIF o JPG/PNG) → Float32Array [224x224]
//  normalizado a [0,1] — replica load_image_robust() de Python
// ================================================================
async function fileToFloat32(file) {
  const isTif = /\.(tif|tiff)$/i.test(file.name);

  if (isTif) {
    const buf    = await file.arrayBuffer();
    const tiff   = await GeoTIFF.fromArrayBuffer(buf);
    const image  = await tiff.getImage();
    const rasters = await image.readRasters({ samples: [0] });
    const raw    = rasters[0];   // TypedArray (Uint16, Uint8, Float32...)
    const srcW   = image.getWidth();
    const srcH   = image.getHeight();

    // Normalizar a [0, 1]
    let norm;
    if (raw instanceof Uint16Array) {
      norm = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) norm[i] = raw[i] / 65535.0;
    } else if (raw instanceof Uint8Array) {
      norm = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) norm[i] = raw[i] / 255.0;
    } else {
      norm = Float32Array.from(raw);
      const max = norm.reduce((a, b) => Math.max(a, b), 0);
      if (max > 1.0) for (let i = 0; i < norm.length; i++) norm[i] /= max;
    }

    return bilinearResize(norm, srcH, srcW, INPUT_SIZE, INPUT_SIZE);

  } else {
    // JPG/PNG: usar canvas
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        // Escalar directamente al tamaño de entrada
        const canvas = document.createElement('canvas');
        canvas.width  = INPUT_SIZE;
        canvas.height = INPUT_SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
        URL.revokeObjectURL(url);

        const pixels = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data; // RGBA
        const result = new Float32Array(INPUT_SIZE * INPUT_SIZE);
        for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
          // Promedio de canales RGB → grayscale (replica img.mean(axis=2) de Python)
          result[i] = (pixels[i * 4] + pixels[i * 4 + 1] + pixels[i * 4 + 2]) / (3 * 255.0);
        }
        resolve(result);
      };
      img.onerror = () => reject(new Error(`No se pudo leer: ${file.name}`));
      img.src = url;
    });
  }
}

// Interpolación bilineal — replica cv2.resize(..., INTER_LINEAR)
function bilinearResize(input, srcH, srcW, dstH, dstW) {
  const out = new Float32Array(dstH * dstW);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const fy = (y + 0.5) * (srcH / dstH) - 0.5;
      const fx = (x + 0.5) * (srcW / dstW) - 0.5;
      const y0 = Math.max(0, Math.floor(fy));
      const y1 = Math.min(srcH - 1, y0 + 1);
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const dy = fy - y0, dx = fx - x0;
      out[y * dstW + x] =
        (1 - dy) * (1 - dx) * input[y0 * srcW + x0] +
        (1 - dy) *      dx  * input[y0 * srcW + x1] +
             dy  * (1 - dx) * input[y1 * srcW + x0] +
             dy  *      dx  * input[y1 * srcW + x1];
    }
  }
  return out;
}

// ================================================================
//  Predicción — corre ambos modelos ONNX y aplica arbitraje
// ================================================================
async function runPrediction() {
  const btn  = document.getElementById('btn-identify');
  const icon = document.getElementById('btn-icon');
  const text = document.getElementById('btn-text');
  const sec  = document.getElementById('results-section');

  btn.disabled     = true;
  icon.innerHTML   = '<span class="spinner"></span>';
  text.textContent = 'Analizando...';
  sec.classList.add('hidden');

  try {
    if (!state.modelsReady.eff && !state.modelsReady.res) {
      throw new Error(
        'Ningun modelo disponible. Coloca los archivos .onnx en /model/'
      );
    }

    // ── Procesar las 5 bandas en paralelo ────────────────────
    text.textContent = 'Procesando bandas...';
    const channelArrays = await Promise.all(
      CHANNELS.map(ch => fileToFloat32(state.files[ch.id]))
    );

    // ── Construir tensor NHWC [1, 224, 224, 5] ───────────────
    // Orden canales: B(0), G(1), R(2), RE(3), NIR(4)
    const inputData = new Float32Array(1 * INPUT_SIZE * INPUT_SIZE * 5);
    for (let h = 0; h < INPUT_SIZE; h++) {
      for (let w = 0; w < INPUT_SIZE; w++) {
        for (let c = 0; c < 5; c++) {
          inputData[h * INPUT_SIZE * 5 + w * 5 + c] = channelArrays[c][h * INPUT_SIZE + w];
        }
      }
    }

    // ── Inferencia ONNX ───────────────────────────────────────
    text.textContent = 'Ejecutando modelos...';
    let probEff = null, probRes = null;

    const runSession = async (key) => {
      if (!state.modelsReady[key]) return null;
      const session = state.sessions[key];
      const tensor  = new ort.Tensor('float32', inputData, [1, INPUT_SIZE, INPUT_SIZE, 5]);
      const feeds   = { [session.inputNames[0]]: tensor };
      const result  = await session.run(feeds);
      return Array.from(result[session.outputNames[0]].data);
    };

    [probEff, probRes] = await Promise.all([runSession('eff'), runSession('res')]);

    if (probEff) {
      const top = CLASSES[probEff.indexOf(Math.max(...probEff))];
      console.log(`[EfficientNet] ${top} (${(Math.max(...probEff)*100).toFixed(1)}%)`);
    }
    if (probRes) {
      const top = CLASSES[probRes.indexOf(Math.max(...probRes))];
      console.log(`[ResNet] ${top} (${(Math.max(...probRes)*100).toFixed(1)}%)`);
    }

    showResults(probEff, probRes);

  } catch (err) {
    console.error('Error en prediccion:', err);
    showError(err.message);
  } finally {
    btn.disabled     = false;
    icon.textContent = '&#128269;';
    text.textContent = 'Identificar especie';
  }
}

// ================================================================
//  Arbitraje de expertos — replica arbitraje() de Python
// ================================================================
function arbitraje(probEff, probRes) {
  let lEff = null, pEff = 0, lRes = null, pRes = 0;

  if (probEff) {
    const i = probEff.indexOf(Math.max(...probEff));
    lEff = CLASSES[i]; pEff = probEff[i];
  }
  if (probRes) {
    const i = probRes.indexOf(Math.max(...probRes));
    lRes = CLASSES[i]; pRes = probRes[i];
  }

  let winner, color, reason;

  if (probEff && probRes) {
    if (lEff === lRes) {
      winner = lEff; color = '#2E7D32'; reason = 'Ambos modelos coinciden';
    } else if (EXPERTOS_EFF.includes(lEff) && EXPERTOS_RES.includes(lRes)) {
      winner = 'CONFLICTO'; color = '#D32F2F';
      reason = `EfficientNet: ${lEff} | ResNet: ${lRes}`;
    } else if (EXPERTOS_EFF.includes(lEff)) {
      winner = lEff; color = '#43A047'; reason = `EfficientNet experto en ${lEff}`;
    } else if (EXPERTOS_RES.includes(lRes)) {
      winner = lRes; color = '#43A047'; reason = `ResNet experto en ${lRes}`;
    } else {
      winner = pEff >= pRes ? lEff : lRes; color = '#1976D2';
      reason = `Mayor confianza: ${pEff >= pRes ? 'EfficientNet' : 'ResNet'}`;
    }
  } else if (probEff) {
    winner = lEff; color = '#1976D2'; reason = 'Solo EfficientNet disponible';
  } else {
    winner = lRes; color = '#1976D2'; reason = 'Solo ResNet disponible';
  }

  return { winner, color, reason, lEff, pEff, lRes, pRes };
}

// ================================================================
//  Mostrar resultados
// ================================================================
function showResults(probEff, probRes) {
  const sec     = document.getElementById('results-section');
  const content = document.getElementById('result-content');
  const arb     = arbitraje(probEff, probRes);
  const mainP   = probEff || probRes;
  const isConf  = arb.winner === 'CONFLICTO';

  const bars = CLASSES
    .map((name, i) => ({ name, prob: mainP[i] ?? 0 }))
    .sort((a, b) => b.prob - a.prob)
    .map((p, i) => {
      const isWinner = arb.winner === p.name;
      const fill = isWinner ? arb.color : '#d6d3d1';  /* stone-300 */
      return `<div>
        <div class="flex justify-between text-xs mb-1.5">
          <span class="font-medium text-stone-700">${p.name}</span>
          <span class="tabular-nums ${isWinner ? 'font-bold' : 'text-stone-400'}"
                style="${isWinner ? `color:${arb.color}` : ''}">${(p.prob * 100).toFixed(1)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:0%;background:${fill}"
               data-target="${(p.prob * 100).toFixed(1)}%"></div>
        </div>
      </div>`;
    }).join('');

  const modelBox = (label, lbl, prob) => !lbl
    ? `<div class="flex-1 rounded-xl border border-stone-200 bg-stone-50 p-3 text-center">
         <p class="text-xs font-semibold text-stone-400 mb-1">${label}</p>
         <p class="text-stone-300 text-xs">No disponible</p>
       </div>`
    : `<div class="flex-1 rounded-xl border border-stone-200 bg-stone-50 p-3 text-center">
         <p class="text-xs font-semibold text-stone-400 mb-1">${label}</p>
         <p class="font-bold text-stone-800 text-sm">${lbl}</p>
         <p class="text-xs text-stone-400 mt-0.5">${(prob * 100).toFixed(1)}%</p>
       </div>`;

  /* Color semántico según tipo de resultado */
  const resultBg    = isConf ? '#fffbeb' : arb.color === '#2E7D32' || arb.color === '#43A047'
                      ? '#f0fdf4' : '#eff6ff';
  const resultBorder = isConf ? '#fde68a' : arb.color === '#2E7D32' || arb.color === '#43A047'
                      ? '#bbf7d0' : '#bfdbfe';

  content.innerHTML = `
    <div class="fade-in space-y-4">

      <!-- Resultado principal -->
      <div class="flex items-center gap-4 p-4 rounded-xl border"
           style="background:${resultBg}; border-color:${resultBorder}">
        <div class="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
             style="background:${arb.color}20">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" fill="none"
               viewBox="0 0 24 24" stroke="currentColor" style="color:${arb.color}">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714
                 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/>
          </svg>
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-xl font-bold truncate" style="color:${arb.color}">${arb.winner}</p>
          <p class="text-xs text-stone-500 mt-0.5">${arb.reason}</p>
        </div>
      </div>

      <!-- Resultado por modelo -->
      <div>
        <p class="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-2">Por modelo</p>
        <div class="flex gap-2">
          ${modelBox('EfficientNet', arb.lEff, arb.pEff)}
          ${modelBox('ResNet50', arb.lRes, arb.pRes)}
        </div>
      </div>

      <!-- Barras de probabilidad -->
      <div class="space-y-2.5">
        <p class="text-xs font-semibold text-stone-400 uppercase tracking-wider">
          Probabilidades · <span class="font-normal normal-case text-stone-400">${probEff ? 'EfficientNet' : 'ResNet'}</span>
        </p>
        ${bars}
      </div>

      ${isConf ? `
      <div class="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
        <span class="font-semibold">Conflicto entre expertos:</span>
        EfficientNet predice <em>${arb.lEff}</em> y ResNet predice <em>${arb.lRes}</em>.
        Se recomienda validación en campo.
      </div>` : ''}

      <div class="pt-1 text-center">
        <button onclick="resetAll()"
          class="text-xs font-medium underline underline-offset-2 transition-colors text-stone-400 hover:text-stone-700">
          Nueva identificación
        </button>
      </div>
    </div>`;

  sec.classList.remove('hidden');
  sec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.prob-bar-fill').forEach(b => { b.style.width = b.dataset.target; });
    }, 80);
  });
}

// ================================================================
//  Error y Reset
// ================================================================
function showError(message) {
  const sec = document.getElementById('results-section');
  document.getElementById('result-content').innerHTML = `
    <div class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 fade-in">
      <p class="font-semibold text-sm mb-1">Error durante la identificación</p>
      <p class="text-xs font-mono break-all text-red-600">${message}</p>
    </div>`;
  sec.classList.remove('hidden');
}

function resetAll() {
  state.files = {};
  state.dims  = {};
  CHANNELS.forEach(ch => {
    const zone   = document.getElementById(`zone-${ch.id}`);
    const tifLbl = zone.querySelector('.tif-label');
    zone.querySelector('.preview-img').classList.add('hidden');
    zone.querySelector('.preview-img').src = '';
    zone.querySelector('.upload-placeholder').classList.remove('hidden');
    if (tifLbl) tifLbl.remove();
    zone.classList.remove('loaded');
    zone.style.borderColor = zone.style.background = '';
    document.getElementById(`input-${ch.id}`).value = '';
    document.getElementById(`dims-${ch.id}`).textContent = '';
  });
  document.getElementById('validation-bar').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('btn-identify').disabled = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
