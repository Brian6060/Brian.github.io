import os, time, subprocess, shlex, sys
from datetime import datetime

PORT = os.environ.get("PORT")
HOST = os.environ.get("HOST")
LOCAL_DIR = os.environ.get("LOCAL_DIR")
REMOTE_DIR = os.environ.get("REMOTE_DIR")
INTERVAL = float(os.environ.get("INTERVAL", "2.0"))

if not all([PORT, HOST, LOCAL_DIR, REMOTE_DIR]):
    print("Need env: PORT, HOST, LOCAL_DIR, REMOTE_DIR")
    sys.exit(1)

def local_files():
    files = []
    for name in os.listdir(LOCAL_DIR):
        if name.lower().endswith(".zip"):
            p = os.path.join(LOCAL_DIR, name)
            if os.path.isfile(p):
                files.append((name, os.path.getsize(p)))
    files.sort()
    return files

def ssh_stat_sizes(names):
    # Use remote 'stat -c %s' (Linux). Print 0 if file not exists yet.
    # Build a single remote command for efficiency.
    # Remote path must be quoted.
    remote_cmd_parts = []
    for n in names:
        rp = f"{REMOTE_DIR.rstrip('/')}/{n}"
        remote_cmd_parts.append(f"stat -c %s {shlex.quote(rp)} 2>/dev/null || echo 0")
    remote_cmd = " ; ".join(remote_cmd_parts)

    cmd = ["/usr/bin/ssh", "-p", PORT, "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
           HOST, remote_cmd]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
    sizes = []
    for line in out:
        try:
            sizes.append(int(line.strip()))
        except:
            sizes.append(0)
    return sizes

def fmt_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit=="B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def main():
    lf = local_files()
    if not lf:
        print(f"No .zip files found in {LOCAL_DIR}")
        return
    names = [n for n,_ in lf]
    totals = [s for _,s in lf]
    total_all = sum(totals)

    prev_remote = [0]*len(names)
    prev_t = time.time()

    while True:
        try:
            remote_sizes = ssh_stat_sizes(names)
        except subprocess.CalledProcessError as e:
            print("\n[ERROR] ssh failed. Output:\n", e.output.decode(errors="ignore"))
            return

        now = time.time()
        dt = max(1e-6, now - prev_t)
        delta_all = sum(max(0, r - pr) for r,pr in zip(remote_sizes, prev_remote))
        speed = delta_all / dt  # bytes/s

        done_all = sum(min(r,t) for r,t in zip(remote_sizes, totals))
        pct_all = 100.0 * done_all / total_all if total_all else 0.0
        eta = (total_all - done_all) / speed if speed > 0 else float("inf")

        # Clear screen (ANSI)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Upload monitor")
        print(f"Remote: {HOST}:{REMOTE_DIR}")
        print(f"Local : {LOCAL_DIR}")
        print("-"*80)
        print(f"TOTAL: {fmt_bytes(done_all)} / {fmt_bytes(total_all)}  ({pct_all:6.2f}%)"
              f"   SPEED: {fmt_bytes(speed)}/s   ETA: {int(eta) if eta!=float('inf') else 'INF'}s")
        print("-"*80)

        for (name, total), remote in zip(lf, remote_sizes):
            done = min(remote, total)
            pct = 100.0 * done / total if total else 0.0
            print(f"{name:<28} {fmt_bytes(done):>8} / {fmt_bytes(total):<8}  ({pct:6.2f}%)")

        if done_all >= total_all:
            print("\n==> Reached 100% by size. Next step: verify zip integrity on cloud: unzip -tq *.zip")
            return

        prev_remote = remote_sizes
        prev_t = now
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
