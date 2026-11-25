"""
Face gallery: the "address book" of known people for the attendance system.

Each known person is one file gallery/<slug>.npz holding:
  name        - display name (e.g. "Mohamad Alaa")
  embeddings  - all their L2-normalized ArcFace embeddings, shape (n, 512)

Matching: each person is represented by the centroid (mean, re-normalized) of their
embeddings. A new face goes to the closest centroid by cosine similarity, and is
accepted only if that similarity >= threshold; otherwise it is unknown.

Command line (run from the face_recognition/ folder):
  python gallery.py list
  python gallery.py import-npz "Mohamad Alaa" data/processed/alaa.npz
  python gallery.py remove "Mohamad Alaa"
  python gallery.py check
"""
import argparse
import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np

import config as C

log = logging.getLogger("gallery")
EMB_DIM = 512


def _normalize(v):
    """Scale vectors to length 1, so a dot product equals cosine similarity."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _slug(name):
    """Safe file name for a person; non-Latin names (e.g. Arabic) get a stable hash instead."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "person_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]


def load_reviewed(npz_path):
    """Embeddings from an extract_faces.py file, keeping only crops that survived manual review."""
    d = np.load(npz_path)
    keep = np.array([os.path.exists(p) for p in d["crop_paths"]])
    return d["embeddings"][keep]


class Gallery:
    def __init__(self, directory=C.GALLERY_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reload()

    def reload(self):
        """Read every person file and rebuild the list of names and centroids."""
        names, counts, centroids = [], [], []
        for path in sorted(self.dir.glob("*.npz")):
            d = np.load(path)
            emb = d["embeddings"].astype(np.float32)
            names.append(str(d["name"]))
            counts.append(len(emb))
            centroids.append(_normalize(emb.mean(axis=0)))
        self.names, self.counts = names, counts
        self.centroids = np.stack(centroids) if centroids else np.zeros((0, EMB_DIM), np.float32)
        log.info("gallery: %d people %s", len(names), dict(zip(names, counts)))

    def __len__(self):
        return len(self.names)

    def add(self, name, embeddings):
        """Add embeddings for a person (new or existing). Returns their total number of embeddings."""
        embeddings = _normalize(np.asarray(embeddings, np.float32))
        path = self.dir / f"{_slug(name)}.npz"
        if path.exists():
            embeddings = np.vstack([np.load(path)["embeddings"], embeddings])
        np.savez(path, name=np.array(name), embeddings=embeddings)
        self.reload()
        return len(embeddings)

    def remove(self, name):
        path = self.dir / f"{_slug(name)}.npz"
        if not path.exists():
            return False
        path.unlink()
        self.reload()
        return True

    def best_match(self, emb):
        """(name, similarity) of the closest known person, or (None, -1.0) if the gallery is empty.
        emb must be L2-normalized (InsightFace normed_embedding)."""
        if not len(self):
            return None, -1.0
        sims = self.centroids @ emb
        i = int(np.argmax(sims))
        return self.names[i], float(sims[i])

    def identify(self, emb, threshold=C.MATCH_THRESHOLD):
        """Name if the closest person is similar enough, else None (unknown). Also returns the similarity."""
        name, sim = self.best_match(emb)
        return (name if sim >= threshold else None), sim


def _check(gallery):
    """Sanity check on data we already have: Alaa's separate test session vs all LFW strangers."""
    tests = {"alaa_test (should be Alaa)": C.OUT_DIR / "alaa_test.npz",
             "lfw (should be unknown)": C.OUT_DIR / "lfw.npz"}
    for label, path in tests.items():
        emb = load_reviewed(path)
        results = [gallery.identify(e)[0] for e in emb]
        accepted = {n: results.count(n) for n in set(results) if n is not None}
        print(f"{label:<28} faces={len(emb):<6} accepted={accepted or 'none'}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Manage the face gallery")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    imp = sub.add_parser("import-npz")
    imp.add_argument("name")
    imp.add_argument("npz")
    rem = sub.add_parser("remove")
    rem.add_argument("name")
    sub.add_parser("check")
    args = parser.parse_args()

    g = Gallery()
    if args.cmd == "list":
        for n, c in zip(g.names, g.counts):
            print(f"{n:<24} {c} embeddings")
    elif args.cmd == "import-npz":
        total = g.add(args.name, load_reviewed(args.npz))
        print(f"{args.name}: {total} embeddings in the gallery")
    elif args.cmd == "remove":
        print("removed" if g.remove(args.name) else "not found")
    elif args.cmd == "check":
        print(f"threshold = {C.MATCH_THRESHOLD}")
        _check(g)


if __name__ == "__main__":
    main()