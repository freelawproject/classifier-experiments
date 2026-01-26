import os
import subprocess

from clx.settings import CLX_HOME

RUNPOD_POD_IP = os.getenv("RUNPOD_POD_IP")
RUNPOD_POD_PORT = os.getenv("RUNPOD_POD_PORT")
RUNPOD_SSH_KEY = os.getenv("RUNPOD_SSH_KEY")


if __name__ == "__main__":
    remote = f"root@{RUNPOD_POD_IP}:/workspace/clx/projects/docketbert/runs"
    local = CLX_HOME / "projects" / "docketbert"
    include_patterns = [
        "config.json",
        "results.json",
        "events.out.tfevents.*",
    ]

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -i {RUNPOD_SSH_KEY} -p {RUNPOD_POD_PORT}",
    ]

    for pattern in include_patterns:
        cmd.append(f"--include={pattern}")
    cmd.append("--exclude=checkpoint-*")
    cmd.append("--include=*/")
    cmd.append("--exclude=*")

    cmd += [
        remote,
        str(local),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
