from .common import *

class PipelineWorker(QtCore.QObject):
    finished = QtCore.Signal(str, object)
    failed   = QtCore.Signal(str, str)

    def __init__(self, key: str, cb: Callable[[], object]) -> None:
        super().__init__(); self.task_key=key; self.callback=cb

    @QtCore.Slot()
    def run(self) -> None:
        try: self.finished.emit(self.task_key, self.callback())
        except Exception as e: self.failed.emit(self.task_key, str(e))

