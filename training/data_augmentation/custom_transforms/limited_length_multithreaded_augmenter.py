from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


class LimitedLenWrapper:
    """Small compatibility wrapper used by older custom trainers in this workspace."""

    def __init__(
        self,
        num_batches,
        data_loader,
        transform,
        num_processes=0,
        num_cached=2,
        seeds=None,
        pin_memory=False,
        wait_time=0.02,
    ):
        self.num_batches = int(num_batches)
        self.iteration = 0
        if num_processes == 0:
            self.generator = SingleThreadedAugmenter(data_loader, transform)
        else:
            self.generator = MultiThreadedAugmenter(
                data_loader,
                transform,
                num_processes,
                num_cached,
                seeds,
                pin_memory=pin_memory,
                wait_time=wait_time,
            )

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self):
        if self.iteration >= self.num_batches:
            self.iteration = 0
            raise StopIteration
        self.iteration += 1
        return next(self.generator)

    def _finish(self):
        if hasattr(self.generator, "_finish"):
            self.generator._finish()
