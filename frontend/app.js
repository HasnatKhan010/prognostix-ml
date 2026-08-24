/* Prognostix ML operations dashboard.
 *
 * Vanilla ES modules-free JS, no build step. Reads the FastAPI service and draws
 * inline SVG against the design tokens in styles.css.
 *
 * Chart rules applied here: one axis per plot, bars capped at 24px with a 4px
 * rounded data-end and air between bands, 2px lines with round caps, an >=8px
 * end marker carrying a 2px surface ring, hairline solid gridlines, hover +
 * keyboard parity, and a table-view twin for every chart so no value is gated
 * behind a tooltip.
 */

'use strict';

/* --- risk bands: colour is the fixed status palette, never hue alone -------- */

const RISK = {
  critical: { label: 'Critical', icon: '▲', color: 'var(--status-critical)', rank: 0 },
  warning:  { label: 'Warning',  icon: '◆', color: 'var(--status-warning)',  rank: 1 },
  watch:    { label: 'Watch',    icon: '●', color: 'var(--status-watch)',    rank: 2 },
  healthy:  { label: 'Healthy',  icon: '✓', color: 'var(--status-healthy)',  rank: 3 },
};

const BAND_ORDER = ['critical', 'warning', 'watch', 'healthy'];
const SVG_NS = 'http://www.w3.org/2000/svg';

const state = {
  apiBase: '',
  models: [],
  model: null,
  fleet: null,
  selected: null,
  detail: null,
  sensor: null,
  barsView: 'chart',
};

/* --- tiny DOM helpers ------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (key === 'class') node.className = value;
    else if (key === 'style') node.setAttribute('style', value);
    else if (key.startsWith('data-') || key.startsWith('aria-')) node.setAttribute(key, value);
    else node[key] = value;
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
const fmt = (value, digits = 0) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : Number(value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/* --- API ------------------------------------------------------------------- */

function defaultApiBase() {
  const stored = localStorage.getItem('prognostix.apiBase');
  if (stored !== null) return stored;
  return location.protocol === 'file:' ? 'http://localhost:8000' : '';
}

async function api(path, params) {
  const url = new URL(`${state.apiBase}/api/v1${path}`, location.href);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined) url.searchParams.set(key, value);
  }

  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

/* --- bootstrap ------------------------------------------------------------- */

async function connect() {
  showSplash('Connecting to the API…', `Reading ${state.apiBase || ''}/api/v1/health`);

  try {
    const health = await api('/health');
    const models = await api('/models');
    state.models = models.models.filter((entry) => entry.loaded || entry.path);
    state.model = health.default_model || (state.models[0] && state.models[0].name) || null;

    if (!state.models.length) {
      showSplash(
        'No trained model available',
        'The API is running but has no checkpoint to serve.',
        'python scripts/prepare_data.py\npython scripts/train.py --model gru'
      );
      return;
    }

    fillModelSelect();
    await loadFleet();
  } catch (error) {
    showSplash(
      'Cannot reach the API',
      error.message,
      'uvicorn api.main:app --reload --host 0.0.0.0 --port 8000\n\n' +
      'If the API runs on another host or port, set it in the API field above.'
    );
  }
}

function fillModelSelect() {
  const select = $('model-select');
  clear(select);
  for (const entry of state.models) {
    select.appendChild(el('option', { value: entry.name }, entry.name));
  }
  if (state.model) select.value = state.model;
}

async function loadFleet(keepSelection = false) {
  const main = $('main');
  main.classList.add('is-loading');
  setStatus('Scoring fleet…');

  try {
    const fleet = await api('/fleet', { model: state.model, limit: 1000 });
    state.fleet = fleet;

    $('splash').hidden = true;
    main.hidden = false;
    renderFleet();

    const visible = visibleEngines();
    const stillThere = visible.some((engine) => engine.engine_id === state.selected);
    if (!keepSelection || !stillThere) {
      state.selected = visible.length ? visible[0].engine_id : null;
    }
    if (state.selected !== null) await loadDetail(state.selected);
    else clearDetail();

    setStatus(
      `${fleet.count} machines · ${fleet.model} · ${fleet.source}` +
      (fleet.mae !== null && fleet.mae !== undefined ? ` · MAE ${fmt(fleet.mae, 1)} cycles` : '')
    );
  } catch (error) {
    if (error.status === 503) {
      showSplash(
        'Fleet data not prepared',
        error.message,
        'python scripts/prepare_data.py'
      );
    } else {
      showSplash('Could not load the fleet', error.message);
    }
  } finally {
    main.classList.remove('is-loading');
  }
}

async function loadDetail(engineId) {
  try {
    const detail = await api(`/fleet/${engineId}`, { model: state.model, history: 80 });
    state.detail = detail;
    if (!detail.feature_columns.includes(state.sensor)) {
      state.sensor = detail.feature_columns[0];
    }
    renderDetail();
  } catch (error) {
    state.detail = null;
    clearDetail(error.message);
  }
}

/* --- fleet view ------------------------------------------------------------ */

function visibleEngines() {
  const engines = (state.fleet && state.fleet.engines) || [];
  const filter = $('risk-select').value;

  const matched = engines.filter((engine) => {
    if (filter === 'all') return true;
    if (filter === 'action') return engine.risk_level === 'warning' || engine.risk_level === 'critical';
    return engine.risk_level === filter;
  });

  return matched.slice(0, Number($('limit-select').value));
}

function renderFleet() {
  const fleet = state.fleet;

  $('hero-value').textContent = fmt(fleet.fleet_health, 1);
  $('hero-note').textContent =
    `${fleet.action_required} of ${fleet.count} machines need action · median RUL ${fmt(fleet.median_rul)} cycles`;

  renderTiles(fleet.risk_summary, fleet.count);
  renderLegend();

  const engines = visibleEngines();
  $('bars-sub').textContent =
    `Most urgent first · cycles remaining · ${engines.length} shown of ${fleet.count}`;

  drawBars(engines);
  renderFleetTable(engines);
}

function renderTiles(summary, total) {
  const container = $('tiles');
  clear(container);

  for (const band of BAND_ORDER) {
    const meta = RISK[band];
    const count = summary[band] || 0;
    const tile = el('div', { class: 'tile' });

    const head = el('div', { class: 'tile-head' });
    head.appendChild(el('span', { class: 'dot', style: `--dot:${meta.color}`, 'aria-hidden': 'true' }));
    head.appendChild(el('span', { class: 'tile-icon', 'aria-hidden': 'true' }, meta.icon));
    head.appendChild(el('span', {}, meta.label));

    tile.appendChild(head);
    tile.appendChild(el('div', { class: 'tile-value' }, fmt(count)));
    tile.appendChild(el('div', { class: 'tile-sub' },
      total ? `${Math.round((count / total) * 100)}% of fleet` : '—'));
    container.appendChild(tile);
  }
}

function renderLegend() {
  const legend = $('risk-legend');
  clear(legend);

  for (const band of BAND_ORDER) {
    const meta = RISK[band];
    const item = el('li');
    item.appendChild(el('span', { class: 'dot', style: `--dot:${meta.color}`, 'aria-hidden': 'true' }));
    item.appendChild(el('span', { class: 'icon', 'aria-hidden': 'true' }, meta.icon));
    item.appendChild(el('span', {}, meta.label));
    legend.appendChild(item);
  }
}

/* Horizontal bars: one axis, capped thickness, rounded data-end only. */
function drawBars(engines) {
  const chart = $('bars');
  clear(chart);

  if (!engines.length) {
    chart.setAttribute('height', 60);
    const note = svg('text', { x: 0, y: 30, class: 'axis-label' });
    note.textContent = 'No machine matches this filter.';
    chart.appendChild(note);
    return;
  }

  const labelWidth = 74;
  const valueWidth = 52;
  const padTop = 8;
  const axisBand = 26;
  const band = 30;
  const barThickness = Math.min(24, band - 10);
  const plotHeight = engines.length * band;
  const height = padTop + plotHeight + axisBand;
  const width = chart.clientWidth || 640;
  const plotWidth = Math.max(120, width - labelWidth - valueWidth);

  chart.setAttribute('height', height);
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const maxRul = Math.max(...engines.map((engine) => engine.rul_cycles), 1);
  const ticks = niceTicks(0, maxRul, 4);
  const scale = (value) => (value / ticks[ticks.length - 1]) * plotWidth;

  // Gridlines and x-axis ticks: solid hairlines, recessive.
  for (const tick of ticks) {
    const x = labelWidth + scale(tick);
    chart.appendChild(svg('line', {
      x1: x, x2: x, y1: padTop, y2: padTop + plotHeight, class: 'gridline',
    }));
    const label = svg('text', {
      x, y: padTop + plotHeight + 16, class: 'axis-label', 'text-anchor': 'middle',
    });
    label.textContent = fmt(tick);
    chart.appendChild(label);
  }

  chart.appendChild(svg('line', {
    x1: labelWidth, x2: labelWidth, y1: padTop, y2: padTop + plotHeight, class: 'axis-rule',
  }));

  const axisTitle = svg('text', {
    x: labelWidth + plotWidth / 2, y: height - 2, class: 'axis-label', 'text-anchor': 'middle',
  });
  axisTitle.textContent = 'Predicted RUL (cycles)';
  chart.appendChild(axisTitle);

  engines.forEach((engine, index) => {
    const meta = RISK[engine.risk_level] || RISK.watch;
    const y = padTop + index * band;
    const barY = y + (band - barThickness) / 2;
    const barWidth = Math.max(2, scale(engine.rul_cycles));

    const row = svg('g', {
      class: `bar-row${engine.engine_id === state.selected ? ' is-selected' : ''}`,
      tabindex: '0',
      role: 'button',
      'aria-label':
        `Machine ${engine.engine_id}: ${fmt(engine.rul_cycles)} cycles remaining, ` +
        `${meta.label}, health ${fmt(engine.health_score)} of 100`,
    });

    // Hit target spans the whole band, well past the 24px minimum.
    row.appendChild(svg('rect', {
      class: 'bar-hit', x: 0, y, width, height: band, fill: 'transparent', rx: 6,
    }));

    const name = svg('text', {
      x: labelWidth - 10, y: y + band / 2 + 4, class: 'cat-label', 'text-anchor': 'end',
    });
    name.textContent = `#${engine.engine_id}`;
    row.appendChild(name);

    row.appendChild(svg('path', {
      d: barPath(labelWidth, barY, barWidth, barThickness, 4),
      fill: meta.color,
    }));

    // Bars are labelled at the tip - outside the mark, so nothing is clipped.
    const value = svg('text', {
      x: labelWidth + barWidth + 8, y: barY + barThickness / 2 + 4, class: 'value-label',
    });
    value.textContent = fmt(engine.rul_cycles);
    row.appendChild(value);

    const show = (event) => {
      showTip($('bar-tip'), event, `Machine ${engine.engine_id}`, [
        ['Predicted RUL', `${fmt(engine.rul_cycles, 1)} cyc`],
        ['Health score', fmt(engine.health_score, 1)],
        ['Risk band', meta.label],
        ['Cycles run', fmt(engine.cycles_observed)],
        ...(engine.actual_rul !== null && engine.actual_rul !== undefined
          ? [['Actual RUL', `${fmt(engine.actual_rul, 1)} cyc`]]
          : []),
      ]);
    };

    row.addEventListener('mousemove', show);
    row.addEventListener('mouseenter', show);
    row.addEventListener('focus', show);
    row.addEventListener('mouseleave', () => hideTip($('bar-tip')));
    row.addEventListener('blur', () => hideTip($('bar-tip')));
    row.addEventListener('click', () => select(engine.engine_id));
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select(engine.engine_id);
      }
    });

    chart.appendChild(row);
  });
}

/* Square at the baseline, 4px rounded at the data end. */
function barPath(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  const right = x + width;
  return [
    `M${x},${y}`,
    `H${right - r}`,
    `A${r},${r} 0 0 1 ${right},${y + r}`,
    `V${y + height - r}`,
    `A${r},${r} 0 0 1 ${right - r},${y + height}`,
    `H${x}`,
    'Z',
  ].join(' ');
}

function niceTicks(min, max, count) {
  const raw = (max - min) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || magnitude * 10;
  const ticks = [];
  for (let value = 0; value <= max + step * 0.5; value += step) ticks.push(Math.round(value * 100) / 100);
  return ticks.length > 1 ? ticks : [0, max];
}

function renderFleetTable(engines) {
  const container = $('bars-table');
  clear(container);

  const table = el('table');
  const head = el('tr');
  for (const heading of ['Machine', 'RUL (cyc)', 'Health', 'Risk', 'Cycles', 'Actual']) {
    head.appendChild(el('th', {}, heading));
  }
  table.appendChild(el('thead')).appendChild(head);

  const body = el('tbody');
  for (const engine of engines) {
    const meta = RISK[engine.risk_level] || RISK.watch;
    const row = el('tr');
    row.appendChild(el('td', {}, `#${engine.engine_id}`));
    row.appendChild(el('td', { class: 'num' }, fmt(engine.rul_cycles, 1)));
    row.appendChild(el('td', { class: 'num' }, fmt(engine.health_score, 1)));

    const risk = el('td');
    risk.appendChild(el('span', { class: 'dot', style: `--dot:${meta.color}`, 'aria-hidden': 'true' }));
    risk.appendChild(el('span', {}, ` ${meta.icon} ${meta.label}`));
    row.appendChild(risk);

    row.appendChild(el('td', { class: 'num' }, fmt(engine.cycles_observed)));
    row.appendChild(el('td', { class: 'num' },
      engine.actual_rul === null || engine.actual_rul === undefined ? '—' : fmt(engine.actual_rul, 1)));
    body.appendChild(row);
  }
  table.appendChild(body);
  container.appendChild(table);
}

/* --- detail view ----------------------------------------------------------- */

function select(engineId) {
  state.selected = engineId;
  document.querySelectorAll('.bar-row').forEach((row) => row.classList.remove('is-selected'));
  drawBars(visibleEngines());
  loadDetail(engineId);
}

function clearDetail(message) {
  $('detail-body').hidden = true;
  $('detail-badge').hidden = true;
  const empty = $('detail-empty');
  empty.hidden = false;
  empty.textContent = message || 'No machine selected.';
  $('detail-title').textContent = 'Machine detail';
  $('detail-sub').textContent = 'Choose a machine from the chart';
}

function renderDetail() {
  const detail = state.detail;
  if (!detail) return clearDetail();

  const meta = RISK[detail.risk_level] || RISK.watch;
  $('detail-empty').hidden = true;
  $('detail-body').hidden = false;

  $('detail-title').textContent = `Machine #${detail.engine_id}`;
  $('detail-sub').textContent = `${detail.model} · ${fmt(detail.cycles_observed)} cycles recorded`;

  const badge = $('detail-badge');
  clear(badge);
  badge.hidden = false;
  badge.appendChild(el('span', { class: 'dot', style: `--dot:${meta.color}`, 'aria-hidden': 'true' }));
  badge.appendChild(el('span', {}, `${meta.icon} ${meta.label}`));

  $('meter-value').textContent = fmt(detail.health_score, 1);
  drawMeter(detail.health_score, meta.color);

  const facts = $('detail-facts');
  clear(facts);
  const rows = [
    ['Predicted RUL', `${fmt(detail.rul_cycles, 1)} cyc`],
    ['Cycles run', fmt(detail.cycles_observed)],
  ];
  if (detail.actual_rul !== null && detail.actual_rul !== undefined) {
    rows.push(['Actual RUL', `${fmt(detail.actual_rul, 1)} cyc`]);
    rows.push(['Error', `${fmt(detail.rul_cycles - detail.actual_rul, 1)} cyc`]);
  }
  for (const [term, value] of rows) {
    facts.appendChild(el('dt', {}, term));
    facts.appendChild(el('dd', {}, value));
  }

  $('detail-action').textContent = detail.recommended_action;

  fillSensorSelect(detail.feature_columns);
  drawSpark();
}

/* Meter: fill carries severity, track is a lighter step of the fill's own ramp. */
function drawMeter(score, color) {
  const meter = $('meter');
  clear(meter);

  const width = meter.clientWidth || 320;
  const height = 18;
  const radius = height / 2;
  meter.setAttribute('height', height);
  meter.setAttribute('viewBox', `0 0 ${width} ${height}`);

  meter.appendChild(svg('rect', {
    x: 0, y: 0, width, height, rx: radius,
    fill: `color-mix(in oklab, ${color} 18%, var(--surface-1))`,
  }));

  const filled = Math.max(radius * 2, (Math.min(Math.max(score, 0), 100) / 100) * width);
  meter.appendChild(svg('rect', { x: 0, y: 0, width: filled, height, rx: radius, fill: color }));

  // Threshold marks in the surface colour, so they separate without extra ink.
  for (const threshold of [16, 40, 64]) {
    const x = (threshold / 100) * width;
    meter.appendChild(svg('line', {
      x1: x, x2: x, y1: 0, y2: height, stroke: 'var(--surface-1)', 'stroke-width': 2,
    }));
  }
}

function fillSensorSelect(columns) {
  const select = $('sensor-select');
  if (select.dataset.filled === columns.join(',')) {
    select.value = state.sensor;
    return;
  }

  clear(select);
  for (const column of columns) {
    select.appendChild(el('option', { value: column }, column.replace('sensor_', 'Sensor ')));
  }
  select.dataset.filled = columns.join(',');
  select.value = state.sensor || columns[0];
  state.sensor = select.value;
}

/* Single series: 2px line, no legend box (the title names it), crosshair on hover. */
function drawSpark() {
  const chart = $('spark');
  clear(chart);

  const detail = state.detail;
  if (!detail) return;

  const values = detail.sensor_history[state.sensor] || [];
  const cycles = detail.cycles || [];
  $('spark-title').textContent = `${state.sensor.replace('sensor_', 'Sensor ')} trend`;
  renderSparkTable(cycles, values);

  if (values.length < 2) {
    chart.setAttribute('height', 40);
    const note = svg('text', { x: 0, y: 24, class: 'axis-label' });
    note.textContent = 'Not enough history to plot.';
    chart.appendChild(note);
    return;
  }

  const width = chart.clientWidth || 420;
  const height = 132;
  const padLeft = 44;
  const padRight = 14;
  const padTop = 12;
  const axisBand = 24;
  const plotWidth = Math.max(60, width - padLeft - padRight);
  const plotHeight = height - padTop - axisBand;

  chart.setAttribute('height', height);
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = span * 0.12;
  const low = min - pad;
  const high = max + pad;

  const x = (index) => padLeft + (index / (values.length - 1)) * plotWidth;
  const y = (value) => padTop + plotHeight - ((value - low) / (high - low)) * plotHeight;

  for (const value of [low + (high - low) / 2, high]) {
    chart.appendChild(svg('line', {
      x1: padLeft, x2: padLeft + plotWidth, y1: y(value), y2: y(value), class: 'gridline',
    }));
  }
  chart.appendChild(svg('line', {
    x1: padLeft, x2: padLeft + plotWidth, y1: y(low), y2: y(low), class: 'axis-rule',
  }));

  for (const value of [low, high]) {
    const label = svg('text', {
      x: padLeft - 8, y: y(value) + 4, class: 'axis-label', 'text-anchor': 'end',
    });
    label.textContent = value.toFixed(1);
    chart.appendChild(label);
  }

  for (const index of [0, values.length - 1]) {
    const label = svg('text', {
      x: x(index), y: height - 8, class: 'axis-label',
      'text-anchor': index === 0 ? 'start' : 'end',
    });
    label.textContent = `cycle ${cycles[index] ?? index + 1}`;
    chart.appendChild(label);
  }

  chart.appendChild(svg('polyline', {
    points: values.map((value, index) => `${x(index)},${y(value)}`).join(' '),
    fill: 'none',
    stroke: 'var(--series-1)',
    'stroke-width': 2,
    'stroke-linejoin': 'round',
    'stroke-linecap': 'round',
  }));

  const lastIndex = values.length - 1;
  // End marker >=8px with a 2px surface ring so it stays legible over the line.
  chart.appendChild(svg('circle', {
    cx: x(lastIndex), cy: y(values[lastIndex]), r: 4.5,
    fill: 'var(--series-1)', stroke: 'var(--surface-1)', 'stroke-width': 2,
  }));

  const endLabel = svg('text', {
    x: x(lastIndex) - 6, y: y(values[lastIndex]) - 10, class: 'value-label', 'text-anchor': 'end',
  });
  endLabel.textContent = values[lastIndex].toFixed(1);
  chart.appendChild(endLabel);

  attachCrosshair(chart, { x, y, values, cycles, padLeft, plotWidth, padTop, plotHeight });
}

function attachCrosshair(chart, geometry) {
  const { x, y, values, cycles, padLeft, plotWidth, padTop, plotHeight } = geometry;

  const crosshair = svg('line', {
    y1: padTop, y2: padTop + plotHeight, class: 'gridline', 'stroke-width': 1, visibility: 'hidden',
  });
  const marker = svg('circle', {
    r: 4.5, fill: 'var(--series-1)', stroke: 'var(--surface-1)', 'stroke-width': 2, visibility: 'hidden',
  });
  chart.appendChild(crosshair);
  chart.appendChild(marker);

  const surface = svg('rect', {
    x: padLeft, y: padTop, width: plotWidth, height: plotHeight, fill: 'transparent',
  });

  const move = (event) => {
    const box = chart.getBoundingClientRect();
    const scale = (chart.viewBox.baseVal.width || box.width) / box.width;
    const local = (event.clientX - box.left) * scale;
    const ratio = Math.min(Math.max((local - padLeft) / plotWidth, 0), 1);
    const index = Math.round(ratio * (values.length - 1));

    crosshair.setAttribute('x1', x(index));
    crosshair.setAttribute('x2', x(index));
    crosshair.setAttribute('visibility', 'visible');
    marker.setAttribute('cx', x(index));
    marker.setAttribute('cy', y(values[index]));
    marker.setAttribute('visibility', 'visible');

    showTip($('spark-tip'), event, `Cycle ${cycles[index] ?? index + 1}`, [
      [state.sensor.replace('sensor_', 'Sensor '), values[index].toFixed(3)],
    ], true);
  };

  const leave = () => {
    crosshair.setAttribute('visibility', 'hidden');
    marker.setAttribute('visibility', 'hidden');
    hideTip($('spark-tip'));
  };

  surface.addEventListener('mousemove', move);
  surface.addEventListener('mouseleave', leave);
  chart.appendChild(surface);
}

function renderSparkTable(cycles, values) {
  const container = $('spark-table');
  clear(container);

  const table = el('table');
  const head = el('tr');
  head.appendChild(el('th', {}, 'Cycle'));
  head.appendChild(el('th', {}, state.sensor));
  table.appendChild(el('thead')).appendChild(head);

  const body = el('tbody');
  values.forEach((value, index) => {
    const row = el('tr');
    row.appendChild(el('td', { class: 'num' }, String(cycles[index] ?? index + 1)));
    row.appendChild(el('td', { class: 'num' }, Number(value).toFixed(3)));
    body.appendChild(row);
  });
  table.appendChild(body);
  container.appendChild(table);
}

/* --- tooltips -------------------------------------------------------------- */

function showTip(tip, event, title, rows, above = false) {
  clear(tip);
  tip.appendChild(el('div', { class: 'tip-title' }, title));
  for (const [label, value] of rows) {
    const row = el('div', { class: 'tip-row' });
    row.appendChild(el('span', {}, label));
    row.appendChild(el('b', {}, String(value)));
    tip.appendChild(row);
  }

  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const source = event.clientX !== undefined
    ? { x: event.clientX, y: event.clientY }
    : centreOf(event.target);

  const left = Math.min(source.x + 14, window.innerWidth - box.width - 12);
  const top = above
    ? Math.max(source.y - box.height - 14, 8)
    : Math.min(source.y + 14, window.innerHeight - box.height - 12);

  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function centreOf(target) {
  const box = target.getBoundingClientRect();
  return { x: box.left + box.width / 2, y: box.top + box.height / 2 };
}

function hideTip(tip) { tip.hidden = true; }

/* --- chrome ---------------------------------------------------------------- */

function setStatus(text) { $('status-line').textContent = text; }

function showSplash(title, body, hint) {
  $('main').hidden = true;
  const splash = $('splash');
  splash.hidden = false;
  $('splash-title').textContent = title;
  $('splash-body').textContent = body || '';

  const hintNode = $('splash-hint');
  hintNode.hidden = !hint;
  hintNode.textContent = hint || '';
  $('splash-retry').hidden = false;
  setStatus('');
}

function applyTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode === 'auto' ? '' : mode);
  localStorage.setItem('prognostix.theme', mode);
  $('theme-icon').textContent = mode === 'dark' ? '◑' : mode === 'light' ? '○' : '◐';
  $('theme-label').textContent = mode === 'dark' ? 'Dark' : mode === 'light' ? 'Light' : 'Auto';
  redraw();
}

function redraw() {
  if (!state.fleet) return;
  drawBars(visibleEngines());
  if (state.detail) {
    drawMeter(state.detail.health_score, (RISK[state.detail.risk_level] || RISK.watch).color);
    drawSpark();
  }
}

/* --- wiring ---------------------------------------------------------------- */

function init() {
  state.apiBase = defaultApiBase();
  $('api-base').value = state.apiBase;

  const themes = ['auto', 'light', 'dark'];
  applyTheme(localStorage.getItem('prognostix.theme') || 'auto');

  $('theme-toggle').addEventListener('click', () => {
    const current = localStorage.getItem('prognostix.theme') || 'auto';
    applyTheme(themes[(themes.indexOf(current) + 1) % themes.length]);
  });

  $('api-base').addEventListener('change', (event) => {
    state.apiBase = event.target.value.replace(/\/+$/, '');
    localStorage.setItem('prognostix.apiBase', state.apiBase);
    connect();
  });

  $('model-select').addEventListener('change', (event) => {
    state.model = event.target.value;
    loadFleet(true);
  });

  $('risk-select').addEventListener('change', () => renderFleet());
  $('limit-select').addEventListener('change', () => renderFleet());
  $('refresh').addEventListener('click', () => loadFleet(true));
  $('splash-retry').addEventListener('click', () => connect());

  $('sensor-select').addEventListener('change', (event) => {
    state.sensor = event.target.value;
    drawSpark();
  });

  document.querySelectorAll('.toggle-btn').forEach((button) => {
    button.addEventListener('click', () => {
      const view = button.dataset.view;
      state.barsView = view;
      document.querySelectorAll('.toggle-btn').forEach((other) => {
        const on = other === button;
        other.classList.toggle('is-on', on);
        other.setAttribute('aria-selected', String(on));
      });
      $('bars-view').hidden = view !== 'chart';
      $('bars-table').hidden = view !== 'table';
      if (view === 'chart') drawBars(visibleEngines());
    });
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redraw, 140);
  });

  connect();
}

document.addEventListener('DOMContentLoaded', init);
