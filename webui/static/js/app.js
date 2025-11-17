const socket = io();

let running = false;
let charts = {};

function makeChart(ctx, label, color) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label,
        data: [],
        borderColor: color,
        tension: 0.1,
      }]
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
    // show immediate feedback that training has started
    document.querySelector('#conn span').textContent = 'training...';
    socket.emit('start_mqtt', { broker, port, topic, train_samples, threshold, threshold_low: thresholdLow, mahalanobis });
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
  socket.on('disconnect', () => {
    document.querySelector('#conn span').textContent = 'disconnected';
  });

  socket.on('status', (m) => {
    if (m && m.msg) {
      console.log('status', m.msg);
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

    // helper: add a single numeric value to a chart's first dataset
    function addPoint(chart, value) {
      if (!chart || !chart.data || !chart.data.datasets || !chart.data.datasets[0]) return;
      const ds = chart.data.datasets[0];
      // append label (we use simple incremental index)
      chart.data.labels.push('');
      ds.data.push(Number(value));
      // trim history to last N points
      const MAX = 200;
      if (ds.data.length > MAX) {
        ds.data.splice(0, ds.data.length - MAX);
        chart.data.labels.splice(0, chart.data.labels.length - MAX);
      }
    }

    addPoint(charts.temp, s.temperature);
    addPoint(charts.ph, s.ph);
    addPoint(charts.rpm, s.rpm);
    charts.temp.update(); charts.ph.update(); charts.rpm.update();
  });
});
