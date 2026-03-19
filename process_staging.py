"""
process_staging.py
──────────────────
Runs in GitHub Actions. Two modes:

  python process_staging.py            # process staging only
  python process_staging.py --retrain  # process + run full retrain pipeline

PROCESS STAGING
  - new_user / updated_profile → upsert into questionnaires collection
    so the live app serves profiles immediately without retraining
  - poi_new   → validates pois_col, appends to data/pois.jsonl
  - poi_review / interaction → appends to data/reviews.jsonl
  - Marks all items processed in MongoDB
  - Exports timestamped JSON snapshot to staging_exports/

RETRAIN (--retrain flag)
  - Patches pipeline/retrain_pipeline.py:
      strips Jupyter %magic lines
      replaces /kaggle/working/ paths with backend/data/
  - Runs the patched script
  - New .pt files land in backend/data/ ready to be committed
"""

import os, sys, json, re, shutil, argparse, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
from pymongo import MongoClient

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true",
                    help="Run full retrain pipeline after processing staging")
args = parser.parse_args()

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI = os.environ.get("MONGODB_URI", "")
if not MONGODB_URI:
    print("ERROR: MONGODB_URI not set", file=sys.stderr)
    sys.exit(1)

client         = MongoClient(MONGODB_URI)
db             = client["POIRS"]
staging_col    = db["staging"]
pois_col       = db["pois"]
questionnaires = db["questionnaires"]

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
DATA_DIR     = REPO_ROOT / "data"
BACKEND_DATA = REPO_ROOT / "backend" / "data"
EXPORT_DIR   = REPO_ROOT / "staging_exports"
PIPELINE_SRC = REPO_ROOT / "pipeline" / "retrain_pipeline.py"

DATA_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

POIS_JSONL    = DATA_DIR / "pois.jsonl"
REVIEWS_JSONL = DATA_DIR / "reviews.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PROCESS STAGING
# ─────────────────────────────────────────────────────────────────────────────
pending = list(staging_col.find({"status": "pending"}).sort("submitted_at", 1))
print(f"[Staging] {len(pending)} pending item(s)")

if not pending:
    print("[Staging] Nothing to do.")
    sys.exit(0)

processed_ids  = []
export_records = []
new_reviews    = []
new_pois       = []
counts = dict(new_user=0, updated_profile=0, poi_new=0,
              poi_review=0, interaction=0, unknown=0, errors=0)

for doc in pending:
    doc_id = doc["_id"]
    ftype  = doc.get("feedback_type", "unknown")

    try:
        if ftype in ("new_user", "updated_profile"):
            uid = doc.get("user_id")
            if uid:
                questionnaires.update_one(
                    {"userId": uid},
                    {"$set": {
                        "userId":      uid,
                        "username":    doc.get("username"),
                        "age":         doc.get("age"),
                        "gender":      doc.get("gender"),
                        "categories":  doc.get("categories", {}),
                        "sensory":     doc.get("sensory", {}),
                        "poi_ratings": doc.get("poi_ratings", {}),
                        "source":      "staging",
                        "updated_at":  datetime.now(timezone.utc).isoformat(),
                    }},
                    upsert=True
                )
                print(f"  [profile] upserted questionnaire for {uid}")
            counts[ftype] += 1

        elif ftype == "poi_new":
            poi_id = doc.get("poi_id")
            name   = doc.get("poi_name", "")
            if poi_id and not pois_col.find_one({"poi_id": poi_id}):
                pois_col.insert_one({
                    "poi_id":   poi_id, "name": name,
                    "category": doc.get("category", ""),
                    "place_id": doc.get("place_id"),
                    "added_at": doc.get("submitted_at"),
                })
                print(f"  [poi_new] safety-inserted {poi_id} ({name})")
            new_pois.append({"poi_id": poi_id, "name": name,
                             "category": doc.get("category", ""),
                             "place_id": doc.get("place_id")})
            if doc.get("review_text") or doc.get("sensory_scores"):
                new_reviews.append({
                    "user_id": doc.get("user_id"), "poi_id": poi_id,
                    "rating": doc.get("rating"),
                    "review_text": doc.get("review_text", ""),
                    "sensory_scores": doc.get("sensory_scores", {}),
                    "submitted_at": doc.get("submitted_at"),
                })
            counts["poi_new"] += 1

        elif ftype == "poi_review":
            assert doc.get("poi_id"), "missing poi_id"
            new_reviews.append({
                "user_id": doc.get("user_id"), "poi_id": doc.get("poi_id"),
                "poi_name": doc.get("poi_name", ""),
                "rating": doc.get("rating"),
                "review_text": doc.get("review_text", ""),
                "sensory_scores": doc.get("sensory_scores", {}),
                "submitted_at": doc.get("submitted_at"),
            })
            counts["poi_review"] += 1

        elif ftype == "interaction":
            assert doc.get("poi_id"), "missing poi_id"
            new_reviews.append({
                "user_id": doc.get("user_id"), "poi_id": doc.get("poi_id"),
                "rating": doc.get("rating"), "aligned": doc.get("aligned"),
                "submitted_at": doc.get("submitted_at"), "type": "interaction",
            })
            counts["interaction"] += 1

        else:
            print(f"  [unknown] type={ftype}")
            counts["unknown"] += 1

        processed_ids.append(doc_id)
        export_records.append({k: v for k, v in doc.items() if k != "_id"})

    except Exception as exc:
        print(f"  [ERROR] {doc.get('staging_id')}: {exc}", file=sys.stderr)
        counts["errors"] += 1
        staging_col.update_one({"_id": doc_id}, {"$set": {
            "status": "error", "error_msg": str(exc),
            "error_at": datetime.now(timezone.utc).isoformat(),
        }})

# Write jsonl
if new_reviews:
    with open(REVIEWS_JSONL, "a", encoding="utf-8") as f:
        for r in new_reviews:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"[Staging] +{len(new_reviews)} reviews → {REVIEWS_JSONL.name}")

if new_pois:
    with open(POIS_JSONL, "a", encoding="utf-8") as f:
        for p in new_pois:
            f.write(json.dumps(p, default=str) + "\n")
    print(f"[Staging] +{len(new_pois)} POIs → {POIS_JSONL.name}")

# Mark processed
if processed_ids:
    res = staging_col.update_many(
        {"_id": {"$in": processed_ids}},
        {"$set": {"status": "processed",
                  "processed_at": datetime.now(timezone.utc).isoformat()}}
    )
    print(f"[Staging] marked {res.modified_count} processed")

# Export snapshot
if export_records:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    ep = EXPORT_DIR / f"{ts}.json"
    ep.write_text(json.dumps(export_records, indent=2, default=str), encoding="utf-8")
    print(f"[Staging] snapshot → staging_exports/{ep.name}")

# Summary
print("\n── Summary " + "─"*38)
for k, v in counts.items():
    if v:
        print(f"  {k:<22} {v}")
print("─"*49)

if not args.retrain:
    print("[Staging] Done (--retrain not passed, skipping model training).")
    sys.exit(1 if counts["errors"] else 0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RETRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Retrain] Preparing patched pipeline script...")

if not PIPELINE_SRC.exists():
    print(f"[Retrain] ERROR: {PIPELINE_SRC} not found", file=sys.stderr)
    sys.exit(1)

if not (BACKEND_DATA / "poi_graph_v6.pt").exists():
    print(f"[Retrain] ERROR: backend/data/poi_graph_v6.pt not found", file=sys.stderr)
    sys.exit(1)

src = PIPELINE_SRC.read_text(encoding="utf-8")

# 1. Strip Jupyter magic commands (%pip install, %matplotlib, etc.)
src = re.sub(r"^\s*%\S.*$", "# (jupyter magic removed)", src, flags=re.MULTILINE)

# 2. Replace /kaggle/working/ paths with our backend/data/ path
backend_str = str(BACKEND_DATA).replace("\\", "/")
src = src.replace("/kaggle/working/", backend_str + "/")
src = src.replace("/kaggle/working",  backend_str)

# 3. Inject Config overrides after Config class instantiation
override = f"""
# ── CI path overrides (injected by process_staging.py) ──
import os as _ci_os
_BD = r"{str(BACKEND_DATA)}"
Config.TRAIN_GRAPH = _ci_os.path.join(_BD, "train_graph.pt")
Config.VAL_GRAPH   = _ci_os.path.join(_BD, "val_graph.pt")
Config.TEST_GRAPH  = _ci_os.path.join(_BD, "test_graph.pt")
Config.OUTPUT_DIR  = _BD
# ────────────────────────────────────────────────────────
"""
src = src.replace("config = Config()\n", override + "config = Config()\n", 1)

# Write to temp file in repo root so relative imports work
with tempfile.NamedTemporaryFile(
    mode="w", suffix="_retrain_ci.py",
    dir=str(REPO_ROOT), delete=False, encoding="utf-8"
) as tmp:
    tmp.write(src)
    patched = Path(tmp.name)

print(f"[Retrain] Patched script: {patched.name}")
print("[Retrain] Running pipeline (this may take a while)...")

try:
    subprocess.run(
        [sys.executable, str(patched)],
        cwd=str(REPO_ROOT),
        check=True,
    )
    print("[Retrain] Pipeline completed successfully ✓")
except subprocess.CalledProcessError as e:
    print(f"[Retrain] Pipeline FAILED (exit {e.returncode})", file=sys.stderr)
    sys.exit(1)
finally:
    patched.unlink(missing_ok=True)

# Copy any .pt files that landed outside backend/data/ into it
for pt in BACKEND_DATA.parent.rglob("*.pt"):
    if pt.parent != BACKEND_DATA:
        dest = BACKEND_DATA / pt.name
        shutil.copy2(pt, dest)
        print(f"[Retrain] copied {pt.name} → backend/data/")

print("\n[Retrain] New .pt files ready in backend/data/")
sys.exit(1 if counts["errors"] else 0)
