import os

from ..utils import find_unique_path


class ZipOutput:

    def __init__(self, path: str, replace: bool = False):
        import zipfile
        self.path = path if replace else find_unique_path(path)
        if os.path.isfile(self.path):
            os.remove(self.path)
        self.archive = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED)

    def write_meta(self, data: dict):
        if not data:
            return
        import json
        with self.archive.open("meta.json", "w") as file:
            file.write(json.dumps(data).encode())

    def write_object(self, filename: str, obj: object):
        import pickle
        with self.archive.open(filename, "w") as file:
            pickle.dump(obj, file)

    def close(self):
        self.archive.close()
