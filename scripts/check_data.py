"""Quick sanity check on the preprocessed data.


Verifies:
 * metadata.json exists and has num_users
 * All user IDs in train/val/test are within [0, num_users)
 * Reports max id observed in each split


Run:
   PYTHONPATH=. python scripts/check_data.py
"""


import json
import os
import pickle
import sys




def check(proc_dir: str = "./data/processed"):
   meta_path = os.path.join(proc_dir, "metadata.json")
   if not os.path.exists(meta_path):
       print(f"❌ MISSING: {meta_path}")
       print("   Run `python data/preprocess.py` first.")
       sys.exit(1)


   with open(meta_path) as f:
       meta = json.load(f)
   num_users = int(meta["num_users"])
   print(f"✓ metadata.json: num_users = {num_users:,}")
   print()


   all_ok = True
   for split in ["train", "val", "test"]:
       path = os.path.join(proc_dir, f"{split}.pkl")
       if not os.path.exists(path):
           print(f"❌ MISSING: {path}")
           all_ok = False
           continue
       with open(path, "rb") as f:
           tuples = pickle.load(f)


       max_target = 0
       max_neighbor = 0
       pos = 0
       neg = 0
       for t in tuples:
           max_target = max(max_target, int(t["target_node"]))
           for nb in t["neighbors"]:
               max_neighbor = max(max_neighbor, int(nb))
           if float(t["activated"]) >= 0.5:
               pos += 1
           else:
               neg += 1


       max_id = max(max_target, max_neighbor)
       in_range = max_id < num_users
       has_both_classes = pos > 0 and neg > 0
       ok = in_range and has_both_classes
       status = "✓" if ok else "❌"
       print(
           f"{status} {split:5s}: {len(tuples):,} tuples | "
           f"pos={pos:,} | neg={neg:,} | "
           f"max_id={max_id:,} (limit={num_users:,})"
       )
       if not in_range:
           print(f"   ↳ ids exceed num_users")
       if not has_both_classes:
           print(f"   ↳ degenerate label set (need both pos and neg)")
       if not ok:
           all_ok = False


   print()
   if all_ok:
       print("✓ All checks passed. Data is ready for training.")
   else:
       print("❌ Some IDs exceed num_users. Re-run `python data/preprocess.py`.")
       sys.exit(1)




if __name__ == "__main__":
   check()
