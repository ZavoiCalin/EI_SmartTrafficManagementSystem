# FILE: inference.py




def _load_artifacts(path=ARTIFACT_DIR):
global _route_clf, _weight_reg, _route_labels, _inter_enc
if _route_clf is None:
with open(os.path.join(path, "route_clf.pkl"), "rb") as f:
_route_clf = pickle.load(f)
with open(os.path.join(path, "weight_reg.pkl"), "rb") as f:
_weight_reg = pickle.load(f)
with open(os.path.join(path, "route_labels.json"), "r") as f:
_route_labels = json.load(f)
with open(os.path.join(path, "intersection_encoder.pkl"), "rb") as f:
_inter_enc = pickle.load(f)




def predict(features, artifacts_path=ARTIFACT_DIR):
"""
features: list or np.array with order [intersection (string or int), hour, dow, vehicle_count, pedestrian_count, traffic_light_status, avg_speed]
If the first element is a string intersection name, it will be encoded using the saved encoder.


Returns: JSON serializable dict {"optimal_route": [...], "total_weight": float}
"""
_load_artifacts(artifacts_path)
feat = np.array(features, dtype=object)
# if first element is string, encode
if isinstance(feat[0], str):
try:
inter_val = _inter_enc.transform([feat[0]])[0]
except Exception:
inter_val = 0
feat[0] = inter_val
# ensure numeric array
X = np.array([feat.astype(float)])


# predict
route_out = _route_clf.predict(X)[0]
weight_out = _weight_reg.predict(X)[0]


route_str = _route_labels[int(route_out)] if _route_labels is not None else str(int(route_out))
route_list = route_str.split("->") if "->" in route_str else [route_str]


return {"optimal_route": route_list, "total_weight": float(weight_out)}


# if run as script for local testing
if __name__ == "__main__":
import sys
# accept JSON array on command line or test sample
if len(sys.argv) > 1:
features = json.loads(sys.argv[1])
else:
features = ["A", 12, 2, 20, 0, 0, 0.0]
out = predict(features)
print(json.dumps(out, indent=2))