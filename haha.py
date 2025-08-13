import os

path = "reflectorch_edit"

if os.path.islink(path):
    os.unlink(path)
    print(f"Removed symbolic link: {path}")
else:
    print(f"{path} is not a symlink.")
