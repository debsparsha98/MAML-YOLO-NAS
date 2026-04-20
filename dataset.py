import torch
from torch.utils.data import Dataset
import random
from typing import List, Dict

from prepare_data import get_annotations_for_image

# ===== EPISODE LOGGING =====
import json
import os
import datetime
#from threading import Lock


class FSODDataset(Dataset):
    """
    Few-Shot Object Detection episodic dataset.

    Each __getitem__ generates ONE episode:
    - Support set (K-shot)
    - Query set (Q-query)
    Episodes are logged internally to an append-only JSONL file.
    """

    def __init__(
        self,
        dataset_info: Dict[int, List[str]],
        n_way: int,
        k_shot: int,
        q_query: int,
        task_count: int,
        fallback: str = "raise",              # "raise" | "repeat" | "fill_with_other_classes"
        unique_across_episode: bool = False
    ):
        super().__init__()

        # ---- Dataset config ----
        self.dataset_info = {k: list(dict.fromkeys(v)) for k, v in dataset_info.items()}
        self.classes = list(self.dataset_info.keys())

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.task_count = task_count

        assert fallback in ("raise", "repeat", "fill_with_other_classes")
        self.fallback = fallback
        self.unique_across_episode = unique_across_episode

        # ---- Episode logging (ALWAYS ON) ----
        #self.episode_log_path = "episode_log.jsonl"
        pid = os.getpid()
        self.episode_log_path = f"episode_log_worker_{pid}.jsonl"
        #self._episode_log_lock = Lock()
        open(self.episode_log_path, "a").close()

    def __len__(self):
        return self.task_count

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_for_class(
        self, cls: int, num_needed: int, used_paths: set
    ) -> List[str]:
        pool = self.dataset_info.get(cls, [])

        # --- FIX: keep only images that actually contain this class ---
        valid_pool = []

        for p in pool:
            ann = get_annotations_for_image(p)

            if ann["labels"].numel() == 0:
                continue

            if (ann["labels"] == cls).any():
                valid_pool.append(p)

        pool = valid_pool
        # ---------------------------------------------------------------

        if self.unique_across_episode:
            pool = [p for p in pool if p not in used_paths]

        if len(pool) >= num_needed:
            return random.sample(pool, num_needed)

        if self.fallback == "raise":
            raise ValueError(f"Class {cls} has only {len(pool)} images")

        if self.fallback == "repeat":
            if not pool:
                raise ValueError(f"No images for class {cls}")
            out = pool.copy()
            while len(out) < num_needed:
                out.append(random.choice(pool))
            return out

        # fill_with_other_classes
        out = pool.copy()
        other = []
        for ocls, paths in self.dataset_info.items():
            if ocls == cls:
                continue
            for p in paths:
                if not (self.unique_across_episode and p in used_paths):
                    other.append(p)
        other = list(dict.fromkeys(other))

        if len(out) + len(other) < num_needed:
            raise ValueError("Not enough images to fill episode")

        out += random.sample(other, num_needed - len(out))
        return out

    # ------------------------------------------------------------------
    # Core FSOD logic
    # ------------------------------------------------------------------
    def _filter_and_remap(self, ann, episode_classes):
        boxes = ann["boxes"]
        labels = ann["labels"]

        if boxes.numel() == 0 or labels.numel() == 0:
            return {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long),
                "global_labels": torch.zeros((0,), dtype=torch.long),
            }

        mask = torch.isin(labels, torch.tensor(episode_classes, device=labels.device))
        boxes = boxes[mask]
        global_labels = labels[mask]

        if boxes.numel() == 0:
            return {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long),
                "global_labels": torch.zeros((0,), dtype=torch.long),
            }

        label_map = {c: i for i, c in enumerate(episode_classes)}

        safe_boxes = []
        safe_local_labels = []
        safe_global_labels = []

        for g, b in zip(global_labels, boxes):
            g = int(g)

            if g not in label_map:
                # Skip objects not belonging to episode classes
                continue

            safe_boxes.append(b)
            safe_local_labels.append(label_map[g])
            safe_global_labels.append(g)

        if len(safe_boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long),
                "global_labels": torch.zeros((0,), dtype=torch.long),
            }

        boxes = torch.stack(safe_boxes)
        local_labels = torch.tensor(safe_local_labels, dtype=torch.long)
        global_labels = torch.tensor(safe_global_labels, dtype=torch.long)

        boxes = boxes.clamp(min=0)

        return {
            "boxes": boxes.float(),
            "labels": local_labels,
            "global_labels": global_labels,
        }

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------
    def _append_episode_log(
        self,
        episode_index,
        episode_classes,
        support_paths,
        support_targets,
        query_paths,
        query_targets,
    ):
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "pid": os.getpid(),
            "requested_index": episode_index,
            "episode_classes": episode_classes,
            "support_set": [],
            "query_set": [],
        }

        for p, t in zip(support_paths, support_targets):
            record["support_set"].append({
                "image_path": p,
                "objects": [
                    {
                        "global_class": int(a["global_class"]),
                        "episodic_class": int(a["class"]),
                        "box": a["box"],
                    }
                    for a in t["json_annotations"]
                ],
            })

        for p, t in zip(query_paths, query_targets):
            record["query_set"].append({
                "image_path": p,
                "objects": [
                    {
                        "global_class": int(a["global_class"]),
                        "episodic_class": int(a["class"]),
                        "box": a["box"],
                    }
                    for a in t["json_annotations"]
                ],
            })

        with open(self.episode_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        #with self._episode_log_lock:
            #with open(self.episode_log_path, "a") as f:
                #f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        print(f"[DATASET] Generating episode {index}")

        episode_classes = random.sample(self.classes, self.n_way)

        support_paths, query_paths = [], []
        support_targets, query_targets = [], []

        used_paths = set()

        for cls in episode_classes:
            paths = self._sample_for_class(
                cls, self.k_shot + self.q_query, used_paths
            )

            sup, qry = paths[: self.k_shot], paths[self.k_shot :]

            if self.unique_across_episode:
                used_paths.update(paths)

            support_paths.extend(sup)
            query_paths.extend(qry)

            for p in sup:
                ann = get_annotations_for_image(p)
                filt = self._filter_and_remap(ann, episode_classes)
                filt["json_annotations"] = [
                    {
                        "global_class": int(g),
                        "class": int(l),
                        "box": b.tolist(),
                    }
                    for g, l, b in zip(
                        filt["global_labels"], filt["labels"], filt["boxes"]
                    )
                ]
                support_targets.append(filt)
                '''
                #---temp---

                print("\n[DEBUG]")
                print("Image:", p)
                print("Raw labels:", ann["labels"])
                print("Raw boxes count:", ann["boxes"].shape[0])
                '''


            for p in qry:
                ann = get_annotations_for_image(p)
                filt = self._filter_and_remap(ann, episode_classes)
                filt["json_annotations"] = [
                    {
                        "global_class": int(g),
                        "class": int(l),
                        "box": b.tolist(),
                    }
                    for g, l, b in zip(
                        filt["global_labels"], filt["labels"], filt["boxes"]
                    )
                ]
                query_targets.append(filt)

        # ---- Log episode (inside dataset only) ----
        self._append_episode_log(
            index,
            episode_classes,
            support_paths,
            support_targets,
            query_paths,
            query_targets,
        )

        return (
            support_paths,
            support_targets,
            query_paths,
            query_targets,
            episode_classes,
        )


def fsod_collate_fn(batch):
    """Return episodes as-is (1 batch = list of episodes)."""
    return batch
