// initialize socket.io client using polling-only transport to avoid websocket upgrade issues
// connect explicitly to the same origin to avoid any path/origin mismatch
const socket = io(window.location.origin, { transports: ['polling'] });

socket.on('connect', () => {
  console.log('socket connected, id=', socket.id);
});
socket.on('connect_error', (err) => {
  console.warn('socket connect_error', err);
  // do not override server status text here; server will emit status events
});
socket.on('error', (err) => {
  console.error('socket error', err);
});
socket.on('reconnect_attempt', () => {
  console.log('socket reconnect attempt');
});

let running = false;
let charts = {};

function makeChart(ctx, label, color) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          tension: 0.1,
          pointRadius: 0,
        },
        {
          label: 'Flagged',
          data: [],
          borderColor: 'rgba(255,0,0,0.2)',
          backgroundColor: 'rgba(255,0,0,0.6)',
          pointRadius: 5,
          pointHoverRadius: 6,
          showLine: false,
          spanGaps: false,
        }
      ]
    },
    options: {
      animation: false,
      scales: { x: { display: false } },
    }
  });
}

window.addEventListener('load', () => {
  charts.temp = makeChart(document.getElementById('chartTemp').getContext('2d'), 'Temperature', 'rgb(255,99,132)');
  charts.ph = makeChart(document.getElementById('chartPH').getContext('2d'), 'pH', 'rgb(54,162,235)');
  charts.rpm = makeChart(document.getElementById('chartRPM').getContext('2d'), 'RPM', 'rgb(75,192,192)');

  document.getElementById('startSynthetic').onclick = () => {
    // gather parameters from inputs
    const mtbf = parseFloat(document.getElementById('mtbf')?.value || '300');
    const mttr = parseFloat(document.getElementById('mttr')?.value || '60');
    const window = parseInt(document.getElementById('window')?.value || '120');
    const threshold = parseFloat(document.getElementById('threshold').value || '4.0');
    const thresholdLow = parseFloat(document.getElementById('thresholdLow').value || String(threshold / 2.0));
    const mahalanobis = document.getElementById('mahalanobis').checked;
    socket.emit('start_synthetic', { mtbf, mttr, seed: null, window, threshold, threshold_low: thresholdLow, mahalanobis });
    running = true;
  };
  document.getElementById('startMQTT').onclick = () => {
    // read MQTT connection inputs (defaults to UCL simulator)
    const broker = document.getElementById('mqttBroker')?.value || 'engf0001.cs.ucl.ac.uk';
    const port = parseInt(document.getElementById('mqttPort')?.value || '1883');
    const stream = document.getElementById('mqttStream')?.value || 'nofaults';
    const topic = `bioreactor_sim/${stream}/telemetry/summary`;
    const train_samples = parseInt(document.getElementById('trainSamples')?.value || '120');
    const threshold = parseFloat(document.getElementById('threshold').value || '4.0');
    const thresholdLow = parseFloat(document.getElementById('thresholdLow')?.value || String(threshold / 2.0));
    const mahalanobis = document.getElementById('mahalanobis').checked;
    const skipTraining = document.getElementById('skipTraining')?.checked || false;
    // show immediate feedback that training has started
    document.querySelector('#conn span').textContent = 'training...';
    socket.emit('start_mqtt', { broker, port, topic, train_samples, threshold, threshold_low: thresholdLow, mahalanobis, skip_training: skipTraining });
    running = true;
  };
  document.getElementById('stop').onclick = () => {
    socket.emit('stop', {});
    running = false;
  };

  document.getElementById('threshold').onchange = (e) => {
    // nothing server-side to set globally; send on start actions. Update UI only.
  };
  document.getElementById('thresholdLow').onchange = (e) => {
    // no-op: thresholds are sent when starting streams
  };
  document.getElementById('mahalanobis').onchange = (e) => {
    // no-op: mahalanobis toggle is applied when starting streams
  };

  socket.on('connect', () => {
    document.querySelector('#conn span').textContent = 'connected';
  });

  // Extra debugging: log any event received so we can see what the server sends
  if (socket.onAny) {
    socket.onAny((event, ...args) => {
      console.log('socket event', event, args);
    });
  }
  socket.on('disconnect', () => {
    document.querySelector('#conn span').textContent = 'disconnected';
  });

  socket.on('status', (m) => {
    // log entire payload for debugging
    console.log('status', m);
    if (!m) return;
    // show training progress if provided
    if (m.msg === 'training' && (m.collected !== undefined)) {
      document.querySelector('#conn span').textContent = `training: ${m.collected}`;
      document.getElementById('trainingCount').textContent = m.collected;
      return;
    }
    if (m.msg === 'training_done') {
      document.querySelector('#conn span').textContent = 'training done';
      document.getElementById('trainingCount').textContent = 'done';
      return;
    }
    if (m.msg) {
      document.querySelector('#conn span').textContent = m.msg;
    }
  });

  socket.on('message', (m) => {
    console.log('server message', m);
    if (m && m.info) {
      // show transient info in status
      document.querySelector('#conn span').textContent = m.info;
      setTimeout(() => { document.querySelector('#conn span').textContent = 'connected'; }, 2000);
    }
  });

  socket.on('telemetry', (msg) => {
    console.log('telemetry event', msg);
    // msg: { t, sample, score, flag, cm }
    const t = new Date().toLocaleTimeString();
    const s = msg.sample;
    const score = msg.score;
    const flag = msg.flag;
    document.getElementById('score').textContent = score.toFixed ? score.toFixed(3) : score;
    document.getElementById('flag').textContent = flag;
    if (msg.cm) {
      // server sends uppercase keys (TP/TN/FP/FN) — support both formats
      const cm = msg.cm;
      document.getElementById('tp').textContent = (cm.tp ?? cm.TP ?? 0);
      document.getElementById('tn').textContent = (cm.tn ?? cm.TN ?? 0);
      document.getElementById('fp').textContent = (cm.fp ?? cm.FP ?? 0);
      document.getElementById('fn').textContent = (cm.fn ?? cm.FN ?? 0);
    }

    // show last sample JSON for debugging
    try {
      document.getElementById('lastSample').textContent = JSON.stringify(msg.sample, null, 2);
    } catch (e) {
      // ignore
    }

    // helper: add a single numeric value to a chart's first dataset
    function addPoint(chart, value, flagged) {
      if (!chart || !chart.data || !chart.data.datasets || !chart.data.datasets[0]) return;
      const ds = chart.data.datasets[0];
      // append label (we use simple incremental index)
      chart.data.labels.push('');
      ds.data.push(Number(value));
      const flagDs = chart.data.datasets[1];
      if (flagDs) {
        flagDs.data.push(flagged ? Number(value) : NaN);
      }
      // trim history to last N points
      const MAX = 200;
      if (ds.data.length > MAX) {
        const excess = ds.data.length - MAX;
        ds.data.splice(0, excess);
        chart.data.labels.splice(0, chart.data.labels.length - MAX);
        if (flagDs) {
          flagDs.data.splice(0, excess);
        }
      }
    }

    addPoint(charts.temp, s.temperature, !!flag);
    addPoint(charts.ph, s.ph, !!flag);
    addPoint(charts.rpm, s.rpm, !!flag);
    charts.temp.update(); charts.ph.update(); charts.rpm.update();
  });
});
