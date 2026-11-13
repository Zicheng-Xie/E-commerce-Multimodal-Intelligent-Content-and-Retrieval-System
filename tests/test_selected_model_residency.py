"""Model stage release must not depend on whether a final image request was cached."""

from unittest.mock import Mock

import pytest

from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.selected_models import LocalModels
from multimodal_ecommerce.selected_runtime_io import load


@pytest.mark.parametrize("failure", [False, True])
def test_vision_release_unloads_even_when_final_request_was_cached(
    failure: bool,
) -> None:
    """Always request server unload before loading a following PyTorch stage.

    Args:
        failure: Simulated unload failure; propagate it rather than overlap models.
    """
    models = LocalModels(load(PROJECT_ROOT / "configs/module2_selected_runtime.json"))
    client = Mock()
    models.current, models.stage = client, "vision"
    if failure:
        client._request.side_effect = RuntimeError("unload failed")
        with pytest.raises(RuntimeError, match="unload failed"):
            models.close()
    else:
        models.close()
        assert models.events == [dict(stage="vision", model_unloaded=True)]
    client._request.assert_called_once_with(
        "/api/generate", dict(model="qwen3.5:4b-q4_K_M", keep_alive=0, stream=False)
    )
    client.close.assert_called_once_with()
    assert models.current is None and models.stage == ""
    models.close()
    client._request.assert_called_once()
