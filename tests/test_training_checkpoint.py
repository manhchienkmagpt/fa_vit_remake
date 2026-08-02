import random

import numpy as np
import torch

from train import capture_random_state, restore_random_state, save_checkpoint


def test_checkpoint_is_saved_atomically(tmp_path):
    checkpoint_path = tmp_path / "best.pt"

    save_checkpoint(checkpoint_path, {"epoch": 3, "best_celebdf_auc": 0.91})

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_celebdf_auc"] == 0.91
    assert not (tmp_path / "best.pt.tmp").exists()


def test_random_state_can_be_restored_for_resume():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_random_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_random_state(state)

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_missing_random_state_is_accepted_for_old_checkpoints():
    restore_random_state(None)
