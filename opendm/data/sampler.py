import torch
from torch.utils.data import Sampler


class TaskBalancedSampler(Sampler[int]):
    """Sample every task equally, independently of its number of frames.

    Each logical epoch contains ``len(dataset)`` samples.  Per-task counts
    differ by at most one; frames inside a task are sampled uniformly with
    replacement.  A common seed is intentionally used on all distributed
    ranks so Accelerate can shard the same global stream without changing the
    task distribution.
    """

    def __init__(
        self,
        task_to_indices: dict[str, list[int]],
        num_samples: int,
        seed: int = 42,
    ):
        if not task_to_indices:
            raise ValueError("TaskBalancedSampler requires at least one task")
        if num_samples <= 0:
            raise ValueError("TaskBalancedSampler requires num_samples > 0")
        if any(not indices for indices in task_to_indices.values()):
            raise ValueError("TaskBalancedSampler cannot sample an empty task")

        self.task_names = sorted(task_to_indices)
        self.task_indices = [
            torch.tensor(task_to_indices[name], dtype=torch.int64)
            for name in self.task_names
        ]
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        num_tasks = len(self.task_indices)
        base_count, remainder = divmod(self.num_samples, num_tasks)
        task_order = torch.randperm(num_tasks, generator=generator).tolist()
        sampled = []
        for order, task_id in enumerate(task_order):
            count = base_count + int(order < remainder)
            population = self.task_indices[task_id]
            offsets = torch.randint(
                len(population), (count,), generator=generator
            )
            sampled.append(population[offsets])

        indices = torch.cat(sampled)
        permutation = torch.randperm(len(indices), generator=generator)
        return iter(indices[permutation].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
