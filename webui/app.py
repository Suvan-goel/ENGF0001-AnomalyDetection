import os
import sys
import time
import threading
import json
import random

from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detector import compute_baseline, compute_covariance_baseline, anomaly_score, mahalanobis_score, update_confusion_matrix, extract_sample
from synthetic_test import make_sample, generate_segmented_fault_sequence
from detector import MQTTDetector

app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, async_mode='eventlet')


class StreamController:
    def __init__(self):
        self.mode = None
        self.thread = None
        self.running = False
        self.window = []
        self.window_size = 120
        self.threshold_high = 4.0
        self.threshold_low = 2.0
        self.use_mahalanobis = False
        self.cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self._last_flag = 0
        self.detector = None
        # persistent synthetic state so stop/start will resume
        self._synthetic_state = None

    def start_synthetic(self, mtbf=300.0, mttr=60.0, seed=None, window_size=120, threshold_high=4.0, threshold_low=None, mahalanobis=False, resume=True):
        """Start or resume the synthetic producer.

        If `resume` is True and a previous synthetic run existed, resume from
        the last saved time/next_fault state. Otherwise start a fresh schedule.
        """
        if self.running:
            return False
        self.mode = 'synthetic'
        self.running = True
        self.window_size = window_size
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low if threshold_low is not None else threshold_high * 0.5
        self.use_mahalanobis = mahalanobis
        # keep confusion matrix across resume; reset only when starting fresh
        if not resume or self._synthetic_state is None:
            self.cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            self._last_flag = 0
            # initialize persistent state
            self._synthetic_state = {
                't': 0,
                'next_fault': int(random.expovariate(1.0 / mtbf)),
                'fault_until': -1,
                'seed': seed,
                'mtbf': mtbf,
                'mttr': mttr,
            }
        else:
            # update parameters in stored state
            st = self._synthetic_state
            st['mtbf'] = mtbf
            st['mttr'] = mttr
            if seed is not None:
                st['seed'] = seed

        # start as a socketio background task (works with eventlet/gevent)
        # pass the state dict to the worker
        self.thread = socketio.start_background_task(self._synthetic_thread, self._synthetic_state)
        return True

    def start_mqtt(self, broker='engf0001.cs.ucl.ac.uk', port=1883, topic='bioreactor_sim/nofaults/telemetry/summary', train_samples=120, threshold_high=4.0, threshold_low=None, mahalanobis=False):
        if self.running:
            return False
        self.mode = 'mqtt'
        self.running = True
        self.window_size = train_samples
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low if threshold_low is not None else threshold_high * 0.5
        self.use_mahalanobis = mahalanobis
        self.cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self._last_flag = 0
        # start as a socketio background task
        self.thread = socketio.start_background_task(self._mqtt_thread, broker, port, topic, train_samples)
        return True

    def stop(self):
        # stop the running producer but keep synthetic state for resume
        self.running = False
        # if detector exists, signal it to stop
        if self.detector is not None:
            self.detector._stop.set()

    def _emit(self, payload):
        # emit to all connected clients
        socketio.emit('telemetry', payload)

    def _synthetic_thread(self, state):
        # state is a dict containing persistent synthetic state
        st = state or {}
        mtbf = st.get('mtbf', 300.0)
        mttr = st.get('mttr', 60.0)
        seed = st.get('seed', None)

        # seed random generator if requested for reproducibility
        if seed is not None:
            random.seed(seed)

        base_means = {'temperature': 37.0, 'ph': 7.0, 'rpm': 500.0}
        base_stds = {'temperature': 0.2, 'ph': 0.05, 'rpm': 5.0}

        # initialize sliding window with fault-free samples if not present
        if not self.window:
            self.window = [make_sample(base_means, base_stds, False) for _ in range(self.window_size)]
            # convert to simple dicts
            self.window = [{'temperature': s['temperature'], 'ph': s['ph'], 'rpm': s['rpm']} for s in self.window]

        # restore persistent counters
        t = int(st.get('t', 0))
        next_fault = int(st.get('next_fault', int(random.expovariate(1.0 / mtbf))))
        fault_until = int(st.get('fault_until', -1))

        while self.running:
            try:
                if t >= next_fault:
                    # start fault
                    dur = int(random.expovariate(1.0 / mttr)) or 1
                    fault_until = t + dur
                    next_fault = t + int(random.expovariate(1.0 / mtbf)) + dur
                    current_fault = True
                else:
                    current_fault = (t < fault_until)

                fault_type = None
                if current_fault:
                    fault_type = random.choice(['therm_voltage_bias', 'ph_offset_bias', 'heater_power_loss'])

                sample = make_sample(base_means, base_stds, current_fault, fault_type)  # type: ignore
                sample_dict = {'temperature': sample['temperature'], 'ph': sample['ph'], 'rpm': sample['rpm']}

                # update window
                self.window.append(sample_dict)
                if len(self.window) > self.window_size:
                    self.window.pop(0)

                # score
                try:
                    if self.use_mahalanobis and len(self.window) >= 2:
                        vars_, mu, invcov = compute_covariance_baseline(self.window)
                        score = mahalanobis_score(sample_dict, vars_, mu, invcov)
                    else:
                        baseline = compute_baseline(self.window)
                        score, z = anomaly_score(sample_dict, baseline)
                except Exception:
                    score, z = anomaly_score(sample_dict, compute_baseline(self.window))

                # hysteresis
                if score >= self.threshold_high:
                    flag = 1
                elif score <= self.threshold_low:
                    flag = 0
                else:
                    flag = self._last_flag
                self._last_flag = flag

                update_confusion_matrix(self.cm, flag, 1 if current_fault else 0)

                payload = {
                    'sample': sample_dict,
                    'score': float(score),
                    'flag': int(flag),
                    'faults': [fault_type] if fault_type else [],
                    'cm': self.cm,
                    'timestamp': int(time.time())
                }
                # debug: log emitted sample so we can trace in server logs
                print(f"[synthetic] t={t} score={score:.3f} temp={sample_dict['temperature']:.3f} ph={sample_dict['ph']:.3f} rpm={sample_dict['rpm']:.1f} flag={flag}")
                self._emit(payload)

                t += 1
                # use socketio.sleep so the eventlet/gevent loop can schedule other tasks
                try:
                    socketio.sleep(1)
                except Exception:
                    # fallback to time.sleep if socketio.sleep not available
                    time.sleep(1.0)
                # persist updated counters to state so a later resume can continue
                try:
                    st['t'] = t
                    st['next_fault'] = next_fault
                    st['fault_until'] = fault_until
                except Exception:
                    pass
            except Exception as e:
                # log and notify clients; then stop the synthetic producer
                import traceback
                tb = traceback.format_exc()
                print('Synthetic thread error:', e)
                print(tb)
                try:
                    socketio.emit('message', {'info': f'synthetic error: {e}'})
                except Exception:
                    pass
                self.running = False
                break

    def _mqtt_thread(self, broker, port, topic, train_samples):
        # Start MQTT detector and train
        self.detector = MQTTDetector(broker, port, topic, self.threshold_high, train_samples)
        try:
            self.detector.connect()
        except Exception:
            pass
        # training
        try:
            self.detector.train()
        except Exception:
            # training may fail if no MQTT available
            pass

        # initialize window
        self.window = []
        while self.running:
            try:
                payload = self.detector._queue.get(timeout=1.0)
            except Exception:
                continue
            try:
                msg = json.loads(payload)
            except Exception:
                continue
            try:
                sample = extract_sample(msg)
            except Exception:
                continue

            # update window
            self.window.append(sample)
            if len(self.window) > self.window_size:
                self.window.pop(0)

            # score
            try:
                if self.use_mahalanobis and len(self.window) >= 2:
                    vars_, mu, invcov = compute_covariance_baseline(self.window)
                    score = mahalanobis_score(sample, vars_, mu, invcov)
                else:
                    baseline = compute_baseline(self.window)
                    score, z = anomaly_score(sample, baseline)
            except Exception:
                score, z = anomaly_score(sample, compute_baseline(self.window))

            # hysteresis
            if score >= self.threshold_high:
                flag = 1
            elif score <= self.threshold_low:
                flag = 0
            else:
                flag = self._last_flag
            self._last_flag = flag

            update_confusion_matrix(self.cm, flag, 1 if msg.get('faults') else 0)

            payload_out = {
                'sample': sample,
                'score': float(score),
                'flag': int(flag),
                'faults': msg.get('faults') or [],
                'cm': self.cm,
                'timestamp': msg.get('timestamp') or int(time.time())
            }
            self._emit(payload_out)


controller = StreamController()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    emit('status', {'msg': 'connected'})


@socketio.on('start_synthetic')
def handle_start_synthetic(data):
    mtbf = float(data.get('mtbf', 300.0))
    mttr = float(data.get('mttr', 60.0))
    seed = data.get('seed')
    window = int(data.get('window', 120))
    tau_h = float(data.get('threshold', 4.0))
    tau_l = data.get('threshold_low')
    if tau_l is not None:
        tau_l = float(tau_l)
    mahalanobis = bool(data.get('mahalanobis', False))
    controller.start_synthetic(mtbf=mtbf, mttr=mttr, seed=seed, window_size=window, threshold_high=tau_h, threshold_low=tau_l, mahalanobis=mahalanobis)
    emit('status', {'msg': 'synthetic_started'})


@socketio.on('start_mqtt')
def handle_start_mqtt(data):
    broker = data.get('broker') or 'engf0001.cs.ucl.ac.uk'
    port = int(data.get('port') or 1883)
    topic = data.get('topic', 'bioreactor_sim/nofaults/telemetry/summary')
    train_samples = int(data.get('train_samples', 120))
    tau_h = float(data.get('threshold', 4.0))
    tau_l = data.get('threshold_low')
    if tau_l is not None:
        tau_l = float(tau_l)
    mahalanobis = bool(data.get('mahalanobis', False))
    controller.start_mqtt(broker=broker, port=port, topic=topic, train_samples=train_samples, threshold_high=tau_h, threshold_low=tau_l, mahalanobis=mahalanobis)
    emit('status', {'msg': 'mqtt_started'})


@socketio.on('stop')
def handle_stop(data=None):
    controller.stop()
    emit('status', {'msg': 'stopped'})


@app.route('/artifacts/<path:filename>')
def artifacts(filename):
    return send_from_directory(os.path.join(ROOT, 'artifacts'), filename)


if __name__ == '__main__':
    # run with eventlet; allow overriding port via PORT env var
    port = int(os.environ.get('PORT', '5000'))
    try:
        socketio.run(app, host='0.0.0.0', port=port)
    except OSError as e:
        print(f"Failed to bind to port {port}: {e}")
        alt = port + 1
        print(f"Trying alternate port {alt} instead. To use a different port permanently, set environment variable PORT.")
        socketio.run(app, host='0.0.0.0', port=alt)
