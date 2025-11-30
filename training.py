# FILE: training.py
route_str = "->".join(route) if isinstance(route, list) else r.get("optimal_route_str")
rows.append({
"intersection": r.get("intersection", ""),
"hour": hour,
"dow": dow,
"vehicle_count": r.get("vehicle_count", 0),
"pedestrian_count": r.get("pedestrian_count", 0),
"traffic_light_status": r.get("traffic_light_status", 0),
"avg_speed": r.get("avg_speed", 0.0),
"route_str": route_str,
"total_weight": r.get("total_weight")
})
return pd.DataFrame(rows)




def train():
# Called by EI training. The platform should place the training JSON at the working dir.
if not os.path.exists(TRAINING_FILENAME):
raise FileNotFoundError(f"Training data not found: {TRAINING_FILENAME}")


records = load_records(TRAINING_FILENAME)
df = prepare_features(records)
df = df.dropna(subset=["route_str", "total_weight"]) # require targets


# Encode route labels
route_enc = LabelEncoder()
df["route_label"] = route_enc.fit_transform(df["route_str"].astype(str))


# Encode intersections
inter_enc = LabelEncoder()
df["intersection_enc"] = inter_enc.fit_transform(df["intersection"].astype(str))


features = ["intersection_enc", "hour", "dow", "vehicle_count", "pedestrian_count", "traffic_light_status", "avg_speed"]
X = df[features].fillna(-1).astype(float).values
y_route = df["route_label"].values
y_weight = df["total_weight"].astype(float).values


# Train models
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X, y_route)
reg.fit(X, y_weight)


# Save artifacts: classifier, regressor, route labels, intersection encoder
with open(os.path.join(ARTIFACT_DIR, "route_clf.pkl"), "wb") as f:
pickle.dump(clf, f)
with open(os.path.join(ARTIFACT_DIR, "weight_reg.pkl"), "wb") as f:
pickle.dump(reg, f)
with open(os.path.join(ARTIFACT_DIR, "route_labels.json"), "w") as f:
json.dump(list(map(str, route_enc.classes_)), f)
with open(os.path.join(ARTIFACT_DIR, "intersection_encoder.pkl"), "wb") as f:
pickle.dump(inter_enc, f)


print("Training complete. Artifacts written to", ARTIFACT_DIR)




# Allow running locally for quick tests
if __name__ == "__main__":
train()