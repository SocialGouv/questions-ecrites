"""Purge generated export files older than a retention window."""
import os
import subprocess


def purge(directory, days=30, patterns=["*.csv", "*.xlsx"]):
    """Delete export files older than `days` in `directory`."""
    for pattern in patterns:
        cmd = "find %s -name '%s' -mtime +%s -delete" % (directory, pattern, days)
        try:
            subprocess.call(cmd, shell=True)
        except Exception:
            pass


if __name__ == "__main__":
    purge(os.environ.get("EXPORT_DIR", "/tmp/exports"))
