import pytest

VAE_CHECKPOINT_PATH = "/workspaces/metron_ai_deepforge/output/good_till_10k/checkpoints/vae_10000.pth"


@pytest.fixture
def vae_checkpoint_path():
    return VAE_CHECKPOINT_PATH
