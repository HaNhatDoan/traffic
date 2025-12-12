# traffic_predictor_app.py
# Flask app + Traffic predictor (fixed LSTM scaling logic + automatic feature alignment)

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb
import torch
import torch.nn as nn
import os
import joblib
from datetime import timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j import GraphDatabase
import json
import traceback
import networkx as nx

print("--- KHỞI TẠO HỆ THỐNG PHÂN TÍCH VÀ DỰ BÁO GIAO THÔNG ---")

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def classify_congestion(volume):
    if volume < 50: return 1
    elif volume < 150: return 2
    elif volume < 300: return 3
    elif volume < 600: return 4
    else: return 5

# -----------------------------
# Predictor Class
# -----------------------------
class TrafficPredictor:
    def __init__(self, mongo_uri, db_name, collection_name, model_folder):
        print("\n[Predictor] Đang khởi tạo...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_folder = model_folder

        # Load data
        self.df_original, self.df_processed = self._load_and_process_data(mongo_uri, db_name, collection_name)

        # placeholders; will try to load from scaler artifact
        self.features_lstm = None
        self.target_col = None

        # Load models and artifacts
        print("[Predictor] Đang tải Models và artifacts...")
        try:
            self.model_xgb = self._load_xgb_model_cpu(os.path.join(model_folder, "model_xgb_stacked_road.pkl"))
            self.label_encoders = joblib.load(os.path.join(model_folder, "label_encoders_road.pkl"))

            # Load scaler_lstm that was used during training (CRUCIAL)
            scaler_path = os.path.join(model_folder, "scaler_road.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Missing scaler_lstm.pkl at {scaler_path}")
            self.scaler_lstm = joblib.load(scaler_path)

            # Determine feature names from scaler if available
            try:
                feature_names = list(self.scaler_lstm.feature_names_in_)
                self.features_lstm = feature_names
                # assume the target column is the last one used during training
                self.target_col = feature_names[-1]
            except Exception:
                # fallback to original hard-coded list (must be 15)
                self.features_lstm = [
                    'year', 'hour', 'region_name_encoded', 'local_authority_name_encoded',
                    'road_name_encoded', 'road_type_encoded', 'link_length_km',
                    'pedal_cycles', 'two_wheeled_motor_vehicles', 'cars_and_taxis',
                    'buses_and_coaches', 'lgvs', 'hgvs_2_rigid_axle', 'all_hgvs', 'all_motor_vehicles'
                ]
                self.target_col = 'all_motor_vehicles'

            # Input size for LSTM is (num_features_used_for_input)
            # Common training pattern: LSTM input = features_without_target
            input_size = len(self.features_lstm) - 1

            self.model_lstm = LSTMModel(input_size=input_size).to(self.device)
            lstm_path = os.path.join(model_folder, "model_lstm_road.pth")
            if not os.path.exists(lstm_path):
                raise FileNotFoundError(f"Missing model_lstm_road.pth at {lstm_path}")
            self.model_lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
            self.model_lstm.eval()

            print(f"[Predictor] ✅ Ready. LSTM Input Size: {input_size}. Features: {len(self.features_lstm)}")
        except Exception as e:
            print(f"[Predictor] ❌ Lỗi tải Model/artifacts: {e}")
            raise e

    def _load_xgb_model_cpu(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
        model = joblib.load(path)
        try:
            model.set_params(device='cpu', tree_method='hist')
        except Exception:
            pass
        return model

    def _load_and_process_data(self, mongo_uri, db_name, collection_name):
        self.mongo_client = MongoClient(mongo_uri)
        coll = self.mongo_client[db_name][collection_name]

        projection = {
            '_id': 0, 'road_name': 1, 'count_date': 1, 'year': 1, 'month': 1, 'hour': 1,
            'day_of_week': 1, 'is_weekend': 1, 'road_type': 1, 'region_name': 1,
            'local_authority_name': 1, 'link_length_km': 1, 'pedal_cycles': 1,
            'two_wheeled_motor_vehicles': 1, 'cars_and_taxis': 1, 'buses_and_coaches': 1,
            'lgvs': 1, 'hgvs_2_rigid_axle': 1, 'all_hgvs': 1, 'all_motor_vehicles': 1
        }

        print("[Predictor] Đang tải dữ liệu từ MongoDB...")
        df_orig = pd.DataFrame(list(coll.find({}, projection)))
        if 'count_date' in df_orig.columns:
            df_orig['count_date'] = pd.to_datetime(df_orig['count_date'])
        print(f"[Predictor] Loaded {len(df_orig)} rows.")

        df_proc = df_orig.copy()
        for col in df_proc.columns:
            if col not in ['road_name', 'count_date', 'day_of_week', 'road_type', 'region_name', 'local_authority_name']:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        df_proc.fillna(0, inplace=True)
        return df_orig, df_proc

    def predict_next_day(self, road_name, n_steps=5):
        df_road = self.df_processed[self.df_processed['road_name'] == road_name].sort_values(['count_date', 'hour']).copy()
        if df_road.empty:
            return {"error": "Không có dữ liệu cho đường này"}, []

        # Ensure encodings exist
        for col in ["day_of_week", "road_type", "region_name", "local_authority_name", "road_name"]:
            enc_name = col + '_encoded'
            df_road[enc_name] = 0
            if col in self.label_encoders:
                le = self.label_encoders[col]
                def safe_transform(val):
                    try:
                        return int(le.transform([str(val)])[0])
                    except Exception:
                        return 0
                df_road[enc_name] = df_road[col].apply(safe_transform)

        # Make sure all required feature columns exist in df_road
        for col in self.features_lstm:
            if col not in df_road.columns:
                df_road[col] = 0

        last_row = df_road.iloc[-1]
        next_date = last_row['count_date'] + timedelta(days=1)

        # ------------------ LSTM: use scaler trained with the model ------------------
        # scaler_lstm.feature_names_in_ should list the columns the scaler expects (including target)
        try:
            scaled_full = self.scaler_lstm.transform(df_road[self.features_lstm])
        except Exception as e:
            print('[Predictor] Warning: scaler_lstm.transform failed, falling back to local scaler. ', e)
            scaler_local = MinMaxScaler()
            scaled_full = scaler_local.fit_transform(df_road[self.features_lstm])

        # confirm shapes
        num_features = scaled_full.shape[1]
        expected_features = len(self.features_lstm)
        if num_features != expected_features:
            raise ValueError(f"Scaled feature count mismatch: got {num_features}, expected {expected_features}")

        # last n_steps window in scaled space
        last_scaled_window = scaled_full[-n_steps:, :].copy()  # shape (n_steps, num_features)

        lstm_scaled_preds = []
        for i in range(24):
            # Prepare input for LSTM: remove target column (assumed last column)
            model_input = last_scaled_window[:, :-1]  # shape (n_steps, num_features-1)
            # Debug check
            if model_input.shape[1] != (len(self.features_lstm) - 1):
                raise ValueError(f"LSTM input width mismatch. Model expects {len(self.features_lstm)-1} features, got {model_input.shape[1]}")

            X_input_tensor = torch.tensor(model_input.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, n_steps, input_size)
            with torch.no_grad():
                pred_scaled = float(self.model_lstm(X_input_tensor).cpu().numpy().flatten()[0])

            lstm_scaled_preds.append(pred_scaled)

            # shift window and append pred_scaled into last column (target) in scaled space
            if n_steps > 1:
                last_scaled_window[:-1, :] = last_scaled_window[1:, :]
            last_scaled_window[-1, -1] = pred_scaled

        # Convert scaled preds to real values via scaler inverse
        real_lstm_preds = []
        for val in lstm_scaled_preds:
            dummy = np.zeros((1, len(self.features_lstm)))
            dummy[0, -1] = val
            try:
                real_val = self.scaler_lstm.inverse_transform(dummy)[0, -1]
            except Exception:
                real_val = val
            real_lstm_preds.append(max(0, real_val))

        # Prepare dataframe for XGBoost
        df_next = pd.DataFrame({'hour': np.arange(24)})

        xgb_features_list = [
            'hour', 'month', 'is_weekend', 'pedal_cycles', 'two_wheeled_motor_vehicles',
            'cars_and_taxis', 'buses_and_coaches', 'lgvs', 'hgvs_2_rigid_axle', 'all_hgvs',
            'day_of_week_encoded', 'road_type_encoded', 'region_name_encoded',
            'local_authority_name_encoded', 'road_name_encoded', 'lstm_predicted_volume'
        ]

        last_real_row = df_road.iloc[-1]
        for col in xgb_features_list:
            if col != 'lstm_predicted_volume' and col not in df_next.columns:
                if col in last_real_row:
                    df_next[col] = last_real_row[col]
                else:
                    df_next[col] = 0

        df_next['year'] = next_date.year
        df_next['month'] = next_date.month
        df_next['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0

        df_next['lstm_predicted_volume'] = [max(0, v) for v in real_lstm_preds]

        # Align to XGBoost feature order
        final_xgb_cols = list(self.model_xgb.feature_names_in_)
        X_xgb = pd.DataFrame(0, index=np.arange(len(df_next)), columns=final_xgb_cols)
        for col in final_xgb_cols:
            if col in df_next.columns:
                X_xgb[col] = df_next[col]

        try:
            final_preds = self.model_xgb.predict(X_xgb)
        except Exception as e:
            print('[Predictor] XGBoost predict error:', e)
            final_preds = np.zeros(24)

        results = []
        for i, val in enumerate(final_preds):
            vol = max(0, int(round(float(val))))
            results.append({'hour': i, 'predicted_volume': vol, 'congestion_level': classify_congestion(vol)})

        return {"date": next_date.strftime('%Y-%m-%d'), "road": road_name}, results


# -----------------------------
# Flask app and graph loading
# -----------------------------
app = Flask(__name__)
CORS(app)

uri_neo4j = "bolt://localhost:7687"
driver_neo4j = GraphDatabase.driver(uri_neo4j, auth=("neo4j", "12345678"))

print("="*20, "LOADING GRAPH", "="*20)
global_graph = nx.DiGraph()

# Biến toàn cục lưu tọa độ để vẽ map (Node Name -> {lat, lon})
node_coords = {}

# --- KHAI BÁO BIẾN TOÀN CỤC ---
global_graph = nx.DiGraph()
node_coords = {} # Dùng để lưu toạ độ vẽ map

def load_graph_to_memory():
    global global_graph, node_coords
    global_graph = nx.DiGraph()
    node_coords = {} 

    print("[Graph] Đang load dữ liệu theo TÊN (Road Name)...")
    
    # Query: Lấy TÊN Road và TÊN Intersection để tạo node
    query = """
    MATCH (r:Road)-[rel:CONNECTS_TO]->(i:Intersection)
    RETURN r.name AS road_name, 
           i.name AS int_name, 
           rel.weight AS weight,
           r.latitude AS r_lat, r.longitude AS r_lon,
           i.latitude AS i_lat, i.longitude AS i_lon
    """
    try:
        with driver_neo4j.session(database="traffic-england-3") as session:
            result = session.run(query)
            count = 0
            for record in result:
                # Lấy tên làm Node ID
                u = record['road_name']
                v = record['int_name']
                w = record['weight'] if record['weight'] else 100.0 # Mặc định nếu null

                if u and v:
                    # Thêm cạnh vào đồ thị: M4 -> A4
                    global_graph.add_edge(u, v, weight=w)
                    
                    # Lưu toạ độ Road (nếu có)
                    if record['r_lat'] and record['r_lon']:
                        node_coords[u] = [record['r_lat'], record['r_lon']]
                    
                    # Lưu toạ độ Intersection (nếu có)
                    if record['i_lat'] and record['i_lon']:
                        node_coords[v] = [record['i_lat'], record['i_lon']]
                    
                    count += 1
            
            print(f"[Graph] ✅ Đã load {count} cạnh.")
            print(f"[Graph] ✅ Tổng số Node: {global_graph.number_of_nodes()}")
            
            # --- DEBUG QUAN TRỌNG: In thử vài node để xem đúng tên chưa ---
            first_5_nodes = list(global_graph.nodes)[:5]
            print(f"[Graph] 👉 Ví dụ các node trong RAM: {first_5_nodes}")
            
    except Exception as e:
        print(f"[Graph] ❌ Lỗi load graph: {e}")

# Gọi hàm ngay khi khởi động
load_graph_to_memory()

MODEL_DIR = os.path.join(os.getcwd(), "model")
print("="*20, "INIT PREDICTOR", "="*20)
try:
    predictor = TrafficPredictor(
        mongo_uri="mongodb://localhost:27017/",
        db_name="traffic",
        collection_name="traffic_features",
        model_folder=MODEL_DIR
    )
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    predictor = None

# -----------------------------
# API endpoints
# -----------------------------
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/regions', methods=['GET'])
def get_regions():
    if not predictor: return jsonify([]), 500
    try:
        regions = sorted(predictor.df_original['region_name'].unique().tolist())
        return jsonify(regions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/roads-by-region', methods=['GET'])
def get_roads_by_region():
    if not predictor: return jsonify([]), 500
    region = request.args.get('region')
    try:
        roads = sorted(predictor.df_original[predictor.df_original['region_name'] == region]['road_name'].unique().tolist())
        return jsonify(roads)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-daily-congestion', methods=['GET'])
def predict_daily_congestion_api():
    if not predictor: return jsonify({"error": "Model not loaded"}), 500
    road = request.args.get('road')
    try:
        info, data = predictor.predict_next_day(road)
        if "error" in info: return jsonify({"error": info["error"]}), 400
        return jsonify({"info": info, "data": data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/traffic-data', methods=['GET'])
def get_traffic_data():
    query = """
    MATCH (start:Intersection)-[rel:CONNECTS_TO]->(end:Intersection)
    WHERE rel.congestion_level IS NOT NULL
    RETURN start.latitude AS sl, start.longitude AS slon,
           end.latitude AS el, end.longitude AS elon,
           rel.congestion_level AS level
    LIMIT 10000
    """
    try:
        with driver_neo4j.session(database="traffic-england-3") as session:
            res = session.run(query)
            data = [{"polyline": json.dumps([[r["sl"], r["slon"]], [r["el"], r["elon"]]]), "level": r["level"]} for r in res]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/intersections', methods=['GET'])
def get_intersections():
    query = "MATCH (i:Intersection) WITH i, rand() AS r ORDER BY r LIMIT 10000 RETURN i.id AS id, i.latitude AS lat, i.longitude AS lon"
    try:
        with driver_neo4j.session(database="traffic-england-3") as session:
            res = session.run(query)
            return jsonify([{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in res])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/find-routes', methods=['GET'])
def find_routes():
    # Lấy tham số source và target
    source = request.args.get('source') # Ví dụ: "M4"
    target = request.args.get('target') # Ví dụ: "A5"
    
    # Debug: In ra xem Frontend gửi gì lên
    print(f"[API] Yêu cầu tìm đường từ '{source}' đến '{target}'")

    if not source or not target:
        return jsonify({"error": "Thiếu thông tin điểm đi/đến"}), 400

    # Kiểm tra node có trong graph không
    if source not in global_graph:
        return jsonify({"error": f"Không tìm thấy điểm đi '{source}' trong dữ liệu"}), 404
    if target not in global_graph:
        return jsonify({"error": f"Không tìm thấy điểm đến '{target}' trong dữ liệu"}), 404

    try:
        # Tìm đường ngắn nhất (Dijkstra)
        path = nx.dijkstra_path(global_graph, source, target, weight='weight')
        
        # Tính tổng trọng số
        total_weight = sum(global_graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

        # Lấy toạ độ để vẽ
        path_coords = []
        for node in path:
            if node in node_coords:
                path_coords.append(node_coords[node])
        
        # Trả về kết quả
        result = {
            "rank": 1,
            "path_ids": path,
            "coords": path_coords,
            "total_weight": round(total_weight, 2)
        }
        
        return jsonify({"source": source, "target": target, "results": [result]})

    except nx.NetworkXNoPath:
        return jsonify({"error": "Không có đường đi nối giữa 2 điểm này"}), 404
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
