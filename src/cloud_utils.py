import subprocess
from pathlib import Path
import os
import zipfile

def backup_model(path, name_batch: str):
    last_model = os.listdir(path)[-1]
    src = Path(os.path.join(path, last_model)).resolve()
    out = Path(os.path.join("models/zip", name_batch)).resolve()

    # Prevent accidentally zipping the zip file into itself (if you place it inside src)
    def should_skip(p: Path) -> bool:
        return p.resolve() == out

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file() and not should_skip(p):
                # Put "contents-only" paths into the zip
                zf.write(p, arcname=p.relative_to(src))


def migrate_instance_cloud(zip_path, instance, connection):
    cmd = [
        "python3", "vast.py", "cloud", "copy",
        "--src", "/workspace/" + zip_path,
        "--dst", "/HighRes-net/models/" + zip_path,
        "--instance", instance,
        "--connection", connection,
        "--transfer", "Instance To Cloud",
    ]
    subprocess.run(cmd, check=True)



def load_zip(zip_path: str, dest_dir: str) -> None:
    zip_path = Path(zip_path).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

