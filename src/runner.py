from typing import Mapping, Any
from catalyst.dl.core import Runner
from models import GAIN, GCAM, GAINMask


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        if isinstance(self.model, GAIN):
            output, output_am, heatmap = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
                "logits_am": output_am,
                "heatmap": heatmap
            }
        elif isinstance(self.model, GCAM):
            output = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
            }
        elif isinstance(self.model, GAINMask):
            output, output_am, heatmap, mask = self.model(batch["images"], batch['targets'])
            return {
                "logits": output,
                "logits_am": output_am,
                "heatmap": heatmap,
                "soft_mask": mask
            }
        else:
            output = self.model(batch["images"])
            return {
                "logits": output
            }