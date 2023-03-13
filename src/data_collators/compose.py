from typing import Dict, List, Any, Callable


class Compose:
    def __init__(self, data_colllators: List[Callable[[Any], Any]] = []):
        self.data_collators = data_colllators

    def __call__(self, batch: Dict[str, List[Any]]) -> Any:
        for data_collator in self.data_collators:
            batch = data_collator(batch)

        return batch