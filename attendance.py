"""
Attendance log: one CSV row per person, never twice, with a manual reset.

File: attendance.csv  (columns: name, date, time)
- mark(name): adds a row only if the name is not already in the file.
- Names already in the file are loaded at start, so restarting the camera
  does not log the same person again.
- reset(): moves the current file to attendance_archive/ (nothing is deleted)
  and starts a new empty one.

Command line (run from the face_recognition/ folder):
  python attendance.py show
  python attendance.py mark "Mohamad Alaa"     # manual test without the camera
  python attendance.py reset
"""
import argparse
import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

import config as C

log = logging.getLogger("attendance")
FIELDS = ["name", "date", "time"]
ENCODING = "utf-8-sig"      # adds a BOM so Excel shows Arabic names correctly


class AttendanceLog:
    def __init__(self, path=C.ATTENDANCE_CSV, archive_dir=C.ATTENDANCE_ARCHIVE_DIR):
        self.path = Path(path)
        self.archive_dir = Path(archive_dir)
        self.rows = []
        if self.path.exists():
            with self.path.open(newline="", encoding=ENCODING) as f:
                self.rows = list(csv.DictReader(f))
        else:
            self._write_header()
        self.present = {r["name"] for r in self.rows}      # who is already logged
        log.info("attendance: %d already present in %s", len(self.present), self.path)

    def _write_header(self):
        with self.path.open("w", newline="", encoding=ENCODING) as f:
            csv.writer(f).writerow(FIELDS)

    def __contains__(self, name):
        return name in self.present

    def __len__(self):
        return len(self.present)

    def mark(self, name):
        """Log a person once. Returns True if a new row was written, False if already present."""
        if name in self.present:
            return False
        now = datetime.now()
        row = {"name": name, "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}
        with self.path.open("a", newline="", encoding=ENCODING) as f:   # written at once: a crash loses nothing
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        self.rows.append(row)
        self.present.add(name)
        log.info("attendance: %s at %s", name, row["time"])
        return True

    def reset(self):
        """Archive the current file (if it has rows) and start an empty one. Returns the archive path or None."""
        archived = None
        if self.rows:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            archived = self.archive_dir / f"attendance_{datetime.now():%Y%m%d_%H%M%S}.csv"
            shutil.move(str(self.path), archived)
        self._write_header()
        self.rows, self.present = [], set()
        log.info("attendance reset%s", f" (old file -> {archived})" if archived else "")
        return archived


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Attendance log")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    m = sub.add_parser("mark")
    m.add_argument("name")
    sub.add_parser("reset")
    args = parser.parse_args()

    att = AttendanceLog()
    if args.cmd == "show":
        for r in att.rows:
            print(f"{r['name']:<24} {r['date']} {r['time']}")
        print(f"total present: {len(att)}")
    elif args.cmd == "mark":
        print("logged" if att.mark(args.name) else "already present, not logged again")
    elif args.cmd == "reset":
        archived = att.reset()
        print(f"reset done; old file archived to {archived}" if archived else "reset done (file was empty)")


if __name__ == "__main__":
    main()