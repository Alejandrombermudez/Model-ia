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

// Canales — ORDEN CRÍTICO: B(del RGB), G, R, RE, NIR
// El canal B proviene de la imagen RGB; el modelo recibe [B, G, R, RE, NIR]
const CHANNELS = [
  { id: 'rgb', label: 'RGB', desc: 'Color RGB',   color: '#3b82f6', ext: 'RGB'    },
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
  initMonitoreo();
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

  // Solo G, R, RE, NIR deben coincidir en resolución; RGB puede ser diferente
  const msChs = CHANNELS.filter(ch => ch.id !== 'rgb');
  const allDimsReady = CHANNELS.every(ch => state.dims[ch.id]);

  if (!allDimsReady) {
    bar.className = 'mt-4 rounded-xl p-3 text-xs bg-amber-50 border border-amber-200 text-amber-900';
    bar.innerHTML = 'Leyendo dimensiones…';
    btn.disabled = true;
    return;
  }

  const msDimVals = msChs.map(ch => state.dims[ch.id]);
  const refW = msDimVals[0].w, refH = msDimVals[0].h;
  const msOk = msDimVals.every(d => d.w === refW && d.h === refH);

  if (!msOk) {
    const table = CHANNELS.map(ch => {
      const d = state.dims[ch.id];
      const isRgb = ch.id === 'rgb';
      const match = isRgb || (d && d.w === refW && d.h === refH);
      return `<div class="text-center">
        <div class="font-bold" style="color:${ch.color}">${ch.label}</div>
        <div class="${match ? 'text-stone-600' : 'text-red-600 font-semibold'}">${d ? `${d.w}×${d.h}` : '—'}</div>
      </div>`;
    }).join('');
    bar.className = 'mt-4 rounded-xl p-3 text-xs bg-red-50 border border-red-200 text-red-700';
    bar.innerHTML = `<div class="font-semibold mb-2">Las bandas MS tienen dimensiones distintas</div>
      <div class="grid grid-cols-5 gap-2">${table}</div>
      <div class="mt-2 text-red-500">Verifica que G, R, RE y NIR sean del mismo vuelo.</div>`;
    btn.disabled = true;
    return;
  }

  const rgbDims = state.dims['rgb'];
  const rgbDiff = rgbDims.w !== refW || rgbDims.h !== refH;

  bar.className = 'mt-4 rounded-xl p-3 text-xs bg-emerald-50 border border-emerald-200 text-emerald-700';
  bar.innerHTML = `<span class="font-semibold">5 bandas cargadas</span> · MS: ${refW}×${refH} px` +
    (rgbDiff
      ? ` · RGB: ${rgbDims.w}×${rgbDims.h} px <span class="text-emerald-500 font-medium">(se reescala automáticamente)</span>`
      : '') +
    ' · Listo para identificar';
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
//  channelId: 'rgb' extrae solo el canal azul (índice 2); otros canales → banda 0
// ================================================================
async function fileToFloat32(file, channelId) {
  const isTif = /\.(tif|tiff)$/i.test(file.name);
  const isRGB = channelId === 'rgb';

  if (isTif) {
    const buf    = await file.arrayBuffer();
    const tiff   = await GeoTIFF.fromArrayBuffer(buf);
    const image  = await tiff.getImage();
    const srcW   = image.getWidth();
    const srcH   = image.getHeight();

    // Para RGB: leer canal azul (muestra 2 si hay ≥3 bandas, si no muestra 0)
    const numBands = image.getSamplesPerPixel();
    const sampleIdx = (isRGB && numBands >= 3) ? 2 : 0;
    const rasters = await image.readRasters({ samples: [sampleIdx] });
    const raw    = rasters[0];   // TypedArray (Uint16, Uint8, Float32...)

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
        const canvas = document.createElement('canvas');
        canvas.width  = INPUT_SIZE;
        canvas.height = INPUT_SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
        URL.revokeObjectURL(url);

        const pixels = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data; // RGBA
        const result = new Float32Array(INPUT_SIZE * INPUT_SIZE);
        for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
          if (isRGB) {
            // Extraer solo canal azul para alimentar la banda B del tensor
            result[i] = pixels[i * 4 + 2] / 255.0;
          } else {
            // Bandas MS en JPG: promedio RGB → gris (replica img.mean(axis=2) de Python)
            result[i] = (pixels[i * 4] + pixels[i * 4 + 1] + pixels[i * 4 + 2]) / (3 * 255.0);
          }
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
      CHANNELS.map(ch => fileToFloat32(state.files[ch.id], ch.id))
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

// ================================================================
//  MÓDULO MONITOREO
// ================================================================

const FENOLOGIA_PARAMS = [
  { id: 'fruto',         label: 'Fruto',         emoji: '🍎' },
  { id: 'hoja',          label: 'Hoja',          emoji: '🍃' },
  { id: 'boton_floral',  label: 'Botón floral',  emoji: '🌸' },
  { id: 'aborto_floral', label: 'Aborto floral', emoji: '🥀' },
];

const PHENO_VALS = [0, 25, 50, 75, 100];

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

const monitorState = {
  photos: [],   // [{ id, batchId, file, url, fecha, species, fenologia:{...} }]
  nextId: 0,
  batchCount: 0,
  mode: 'individual',   // 'individual' | 'lote'
  sessionDate: todayISO(),
  sessionSpecies: '',
  sessionFenologia: Object.fromEntries(['fruto','hoja','boton_floral','aborto_floral'].map(k => [k, null])),
};

// ── Tab switching ────────────────────────────────────────────
function switchTab(tab) {
  ['identificacion', 'monitoreo'].forEach(t => {
    document.getElementById(`panel-${t}`).classList.toggle('hidden', t !== tab);
    document.getElementById(`tab-btn-${t}`).classList.toggle('active', t === tab);
  });
}

// ── Init dropzone ────────────────────────────────────────────
function initMonitoreo() {
  const dropzone = document.getElementById('monitor-dropzone');
  const input    = document.getElementById('monitor-input');
  if (!dropzone || !input) return;

  // Inicializar fecha de sesión con hoy
  const dateInput = document.getElementById('session-date');
  if (dateInput) dateInput.value = monitorState.sessionDate;

  // Poblar opciones de especie en el selector de lote
  const sessionSpeciesEl = document.getElementById('session-species');
  if (sessionSpeciesEl) {
    CLASSES.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      sessionSpeciesEl.appendChild(opt);
    });
  }

  // Generar botones de fenología de sesión
  const phenoRows = document.getElementById('session-pheno-rows');
  if (phenoRows) {
    FENOLOGIA_PARAMS.forEach(p => {
      const row = document.createElement('div');
      row.className = 'session-pheno-row';
      row.innerHTML = `
        <span class="text-xs text-stone-500 shrink-0" style="width:6.5rem">${p.emoji} ${p.label}</span>
        <div class="flex gap-1">
          ${PHENO_VALS.map(v => `
            <button class="session-pheno-btn"
                    data-param="${p.id}"
                    data-val="${v}">${v}</button>
          `).join('')}
        </div>`;
      phenoRows.appendChild(row);
    });
    phenoRows.addEventListener('click', e => {
      const btn = e.target.closest('.session-pheno-btn');
      if (btn) setSessionFenology(btn.dataset.param, parseInt(btn.dataset.val));
    });
  }

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    addPhotos(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
  });
  input.addEventListener('change', e => {
    addPhotos(Array.from(e.target.files));
    input.value = '';
  });
}

// ── Add photos batch ─────────────────────────────────────────
function addPhotos(files) {
  if (!files.length) return;
  const isLote = monitorState.mode === 'lote';
  files.forEach(file => {
    const id  = monitorState.nextId++;
    const url = URL.createObjectURL(file);
    monitorState.photos.push({
      id,
      batchId: monitorState.batchCount,
      file, url,
      fecha:   monitorState.sessionDate,
      species: isLote ? monitorState.sessionSpecies : '',
      fenologia: Object.fromEntries(
        FENOLOGIA_PARAMS.map(p => [p.id, isLote ? monitorState.sessionFenologia[p.id] : null])
      ),
    });
    renderPhotoCard(monitorState.photos[monitorState.photos.length - 1]);
  });
  updateMonitorProgress();
}

// ── Render one photo card ────────────────────────────────────
function renderPhotoCard(photo) {
  const grid = document.getElementById('monitor-grid');

  const shortName = photo.file.name.length > 24
    ? photo.file.name.slice(0, 21) + '…'
    : photo.file.name;

  const speciesOpts = ['', ...CLASSES]
    .map(s => `<option value="${s}"${s === photo.species ? ' selected' : ''}>${s || '— Seleccionar especie —'}</option>`)
    .join('');

  const phenoRows = FENOLOGIA_PARAMS.map(p => `
    <div class="flex items-center gap-2">
      <span class="text-xs text-stone-500 shrink-0" style="width:6rem">
        ${p.emoji} ${p.label}
      </span>
      <div class="flex gap-1">
        ${PHENO_VALS.map(v => `
          <button class="pheno-btn${photo.fenologia[p.id] === v ? ' selected' : ''}"
                  data-photo="${photo.id}"
                  data-param="${p.id}"
                  data-val="${v}">${v}</button>
        `).join('')}
      </div>
    </div>`).join('');

  const card = document.createElement('div');
  card.id        = `monitor-card-${photo.id}`;
  card.className = 'monitor-card bg-white rounded-2xl border border-stone-200 shadow-sm overflow-hidden fade-in';
  card.innerHTML = `
    <!-- Foto -->
    <div class="relative">
      <img src="${photo.url}" alt="${photo.file.name}"
           class="w-full object-cover bg-stone-100" style="height:160px">
      <button onclick="removePhoto(${photo.id})"
              title="Eliminar"
              class="absolute top-2 right-2 w-6 h-6 rounded-full bg-black/50 hover:bg-black/70
                     text-white flex items-center justify-center text-base leading-none
                     transition-colors" style="font-size:1rem; line-height:1">×</button>
      <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/65 to-transparent px-3 py-2">
        <span class="text-white text-xs font-medium block truncate">${shortName}</span>
      </div>
    </div>

    <!-- Formulario de anotación -->
    <div class="p-4 space-y-3">

      <!-- Fecha -->
      <div>
        <label class="text-xs font-semibold text-stone-400 uppercase tracking-wider">Fecha</label>
        <input type="date" class="card-date mt-1"
               value="${photo.fecha}"
               onchange="setCardDate(${photo.id}, this.value)">
      </div>

      <!-- Especie -->
      <div>
        <label class="text-xs font-semibold text-stone-400 uppercase tracking-wider">Especie</label>
        <select class="species-select mt-1"
                onchange="setSpecies(${photo.id}, this.value)">
          ${speciesOpts}
        </select>
      </div>

      <!-- Fenología -->
      <div>
        <label class="text-xs font-semibold text-stone-400 uppercase tracking-wider">Fenología · %</label>
        <div class="mt-2 space-y-1.5">
          ${phenoRows}
        </div>
      </div>

      <!-- Estado -->
      <div id="card-status-${photo.id}" class="text-xs text-stone-300 pt-0.5 border-t border-stone-100">
        Pendiente de calificación
      </div>
    </div>`;

  grid.appendChild(card);

  // Vincular botones de fenología
  card.querySelectorAll('.pheno-btn').forEach(btn => {
    btn.addEventListener('click', () =>
      setPhenology(parseInt(btn.dataset.photo), btn.dataset.param, parseInt(btn.dataset.val))
    );
  });
}

// ── Session controls ─────────────────────────────────────────
function setMode(mode) {
  monitorState.mode = mode;
  document.getElementById('btn-mode-individual').classList.toggle('active', mode === 'individual');
  document.getElementById('btn-mode-lote').classList.toggle('active', mode === 'lote');
  document.getElementById('session-species-wrap').classList.toggle('hidden', mode === 'individual');
  document.getElementById('session-pheno-wrap').classList.toggle('hidden', mode === 'individual');
  updateMonitorProgress();
}

function setSessionDate(val) {
  monitorState.sessionDate = val;
}

function setSessionSpecies(val) {
  monitorState.sessionSpecies = val;
}

function setSessionFenology(param, val) {
  monitorState.sessionFenologia[param] = val;
  document.getElementById('session-pheno-rows')
    ?.querySelectorAll(`.session-pheno-btn[data-param="${param}"]`)
    .forEach(btn => btn.classList.toggle('selected', parseInt(btn.dataset.val) === val));
}

function setCardDate(photoId, val) {
  const photo = monitorState.photos.find(p => p.id === photoId);
  if (!photo) return;
  photo.fecha = val;
  updateCardStatus(photoId);
  updateMonitorProgress();
}

// ── Setters ──────────────────────────────────────────────────
function setSpecies(photoId, species) {
  const photo = monitorState.photos.find(p => p.id === photoId);
  if (!photo) return;
  photo.species = species;
  updateCardStatus(photoId);
  updateMonitorProgress();
}

function setPhenology(photoId, param, val) {
  const photo = monitorState.photos.find(p => p.id === photoId);
  if (!photo) return;
  photo.fenologia[param] = val;

  // Actualizar visual de botones del parámetro
  const card = document.getElementById(`monitor-card-${photoId}`);
  if (!card) return;
  card.querySelectorAll(`.pheno-btn[data-param="${param}"]`).forEach(btn =>
    btn.classList.toggle('selected', parseInt(btn.dataset.val) === val)
  );

  updateCardStatus(photoId);
  updateMonitorProgress();
}

// ── Card status ──────────────────────────────────────────────
function isPhotoComplete(photo) {
  return photo.fecha !== '' &&
    photo.species !== '' &&
    FENOLOGIA_PARAMS.every(p => photo.fenologia[p.id] !== null);
}

function updateCardStatus(photoId) {
  const photo = monitorState.photos.find(p => p.id === photoId);
  const el    = document.getElementById(`card-status-${photoId}`);
  const card  = document.getElementById(`monitor-card-${photoId}`);
  if (!photo || !el) return;

  const complete = isPhotoComplete(photo);
  card.classList.toggle('completed', complete);

  if (complete) {
    el.className   = 'text-xs font-semibold text-emerald-600 pt-0.5 border-t border-stone-100';
    el.textContent = '✓ Calificación completa';
  } else {
    el.className = 'text-xs text-stone-400 pt-0.5 border-t border-stone-100';
    const pending = [];
    if (!photo.fecha) pending.push('fecha');
    if (!photo.species) pending.push('especie');
    FENOLOGIA_PARAMS.forEach(p => {
      if (photo.fenologia[p.id] === null) pending.push(p.label.toLowerCase());
    });
    el.textContent = `Pendiente: ${pending.join(', ')}`;
  }
}

// ── Progress bar ─────────────────────────────────────────────
function updateMonitorProgress() {
  const total  = monitorState.photos.length;
  const done   = monitorState.photos.filter(isPhotoComplete).length;
  const secEl  = document.getElementById('monitor-progress-section');
  const textEl = document.getElementById('monitor-progress-text');
  const fillEl = document.getElementById('monitor-progress-fill');
  if (!secEl) return;

  secEl.classList.toggle('hidden', total === 0);
  if (total > 0) {
    textEl.textContent = `${done} de ${total} ${total === 1 ? 'foto calificada' : 'fotos calificadas'}`;
    fillEl.style.width = `${(done / total) * 100}%`;
  }

  const hasCurrentBatch = monitorState.photos.some(p => p.batchId === monitorState.batchCount);
  document.getElementById('save-batch-wrap')
    ?.classList.toggle('hidden', monitorState.mode !== 'lote' || !hasCurrentBatch);
}

// ── Remove / Clear ───────────────────────────────────────────
function removePhoto(photoId) {
  const idx = monitorState.photos.findIndex(p => p.id === photoId);
  if (idx !== -1) {
    URL.revokeObjectURL(monitorState.photos[idx].url);
    monitorState.photos.splice(idx, 1);
  }
  document.getElementById(`monitor-card-${photoId}`)?.remove();
  updateMonitorProgress();
}

function saveBatch() {
  const currentPhotos = monitorState.photos.filter(p => p.batchId === monitorState.batchCount);
  if (!currentPhotos.length) return;

  const num   = monitorState.batchCount + 1;
  const fecha = monitorState.sessionDate || '—';
  const spec  = monitorState.sessionSpecies || '—';
  const count = currentPhotos.length;

  const grid = document.getElementById('monitor-grid');
  const div  = document.createElement('div');
  div.className = 'batch-divider';
  div.innerHTML = `
    <span style="opacity:0.55;font-weight:700">Lote ${num}</span>
    <span style="opacity:0.35">·</span>
    <span>${fecha}</span>
    <span style="opacity:0.35">·</span>
    <span>${spec}</span>
    <span style="opacity:0.35">·</span>
    <span>${count} ${count === 1 ? 'foto' : 'fotos'}</span>`;
  grid.appendChild(div);

  monitorState.batchCount++;
  monitorState.sessionDate    = todayISO();
  monitorState.sessionSpecies = '';
  FENOLOGIA_PARAMS.forEach(p => { monitorState.sessionFenologia[p.id] = null; });

  const dateEl = document.getElementById('session-date');
  if (dateEl) dateEl.value = monitorState.sessionDate;
  const specEl = document.getElementById('session-species');
  if (specEl) specEl.value = '';
  document.getElementById('session-pheno-rows')
    ?.querySelectorAll('.session-pheno-btn')
    .forEach(btn => btn.classList.remove('selected'));

  updateMonitorProgress();
}

function clearMonitor() {
  if (!confirm('¿Limpiar todas las fotos y calificaciones?')) return;
  monitorState.photos.forEach(p => URL.revokeObjectURL(p.url));
  monitorState.photos    = [];
  monitorState.nextId    = 0;
  monitorState.batchCount = 0;
  FENOLOGIA_PARAMS.forEach(p => { monitorState.sessionFenologia[p.id] = null; });
  document.getElementById('monitor-grid').innerHTML = '';
  document.getElementById('session-pheno-rows')
    ?.querySelectorAll('.session-pheno-btn')
    .forEach(btn => btn.classList.remove('selected'));
  updateMonitorProgress();
}

// ── Export CSV ───────────────────────────────────────────────
function exportCSV() {
  if (!monitorState.photos.length) return;

  const headers = ['fecha', 'archivo', 'especie', ...FENOLOGIA_PARAMS.map(p => p.id)];
  const rows    = monitorState.photos.map(p => [
    p.fecha || '',
    p.file.name,
    p.species || '',
    ...FENOLOGIA_PARAMS.map(fp => p.fenologia[fp.id] ?? ''),
  ]);

  const esc = v => `"${String(v).replace(/"/g, '""')}"`;
  const csv = [headers, ...rows].map(r => r.map(esc).join(',')).join('\r\n');

  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `monitoreo_fenologia_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ================================================================
//  IDENTIFICACIÓN — Reset
// ================================================================
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
