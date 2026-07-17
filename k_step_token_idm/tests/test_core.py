from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from k_step_token_idm.dataset import EmbeddingStore, KStepEmbeddingDataset
from k_step_token_idm.eval_rollout import KStepTokenIDMPolicy
from k_step_token_idm.model import AutoregressiveKStepTokenIDM, KStepTokenIDMConfig
from k_step_token_idm.splits import create_episode_split
from single_step_token_idm.tokenization import ActionTokenizer, ActionTokenizerConfig


class KStepTokenIDMCoreTest(unittest.TestCase):
    def test_model_shapes_and_causality(self):
        torch.manual_seed(0)
        cfg = KStepTokenIDMConfig(
            embed_dim=8,
            action_horizon=3,
            hidden_dim=32,
            condition_layers=2,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=64,
            dropout=0.0,
            time_embed_dim=16,
        )
        model = AutoregressiveKStepTokenIDM(cfg).eval()
        z_t = torch.randn(2, 8)
        z_goal = torch.randn(2, 8)
        steps = torch.tensor([25, 20])
        tokens = torch.randint(0, cfg.n_bins, (2, 3, 2))
        logits = model(z_t, z_goal, steps, tokens)
        self.assertEqual(tuple(logits.shape), (2, 3, 2, cfg.n_bins))

        changed = tokens.clone()
        changed[:, 1] = torch.randint(0, cfg.n_bins, (2, 2))
        changed_logits = model(z_t, z_goal, steps, changed)
        self.assertTrue(torch.equal(logits[:, :2], changed_logits[:, :2]))
        self.assertEqual(tuple(model.generate(z_t, z_goal, steps).shape), (2, 1, 3, 2))

    def test_episode_split_and_dataset_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            episodes = np.repeat(np.arange(10), 30)
            rng = np.random.default_rng(0)
            embeddings = rng.normal(size=(len(episodes), 8)).astype(np.float32)
            actions = rng.normal(size=(len(episodes), 2)).astype(np.float32)
            np.savez(path, embeddings=embeddings, actions=actions, episode_ids=episodes)
            store = EmbeddingStore(path)
            manifest = create_episode_split(store.unique_episode_ids, split_seed=42)
            manifest.validate(store.unique_episode_ids)

            stats = store.action_stats(manifest.train_episode_ids)
            tokenizer = ActionTokenizer.from_stats(
                stats, ActionTokenizerConfig(action_dim=2, n_bins=32)
            )
            for partition in ("train", "val", "test"):
                dataset = KStepEmbeddingDataset(
                    store,
                    manifest,
                    partition,
                    action_horizon=3,
                    goal_offset=25,
                    tokenizer=tokenizer,
                )
                sample = dataset[0]
                self.assertEqual(tuple(sample["actions"].shape), (3, 2))
                self.assertEqual(tuple(sample["action_tokens"].shape), (3, 2))
                episode = int(sample["episode_id"])
                self.assertIn(episode, manifest.episode_ids(partition))
                start = int(sample["start_index"])
                goal = int(sample["goal_index"])
                self.assertEqual(goal - start, 25)
                self.assertTrue(np.all(store.episode_ids[start : goal + 1] == episode))

    def test_closed_loop_action_buffer_and_remaining_horizon(self):
        class Encoder:
            def __call__(self, pixels, interpolate_pos_encoding=True):
                return type("EncoderOutput", (), {"last_hidden_state": pixels.unsqueeze(1)})

        class JEPA:
            encoder = Encoder()

            @staticmethod
            def projector(value):
                return value

        class IDM:
            action_horizon = 3
            max_horizon = 50

            def __init__(self):
                self.seen_steps = []

            def generate(self, z_t, z_goal, steps, **kwargs):
                self.seen_steps.append(steps.detach().cpu().tolist())
                return torch.zeros(z_t.shape[0], 1, 3, 2, dtype=torch.long)

        class ActionSpace:
            shape = (2, 2)

        class Env:
            action_space = ActionSpace()

        tokenizer = ActionTokenizer(
            ActionTokenizerConfig(action_dim=2, n_bins=32),
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        idm = IDM()
        policy = KStepTokenIDMPolicy(
            JEPA(),
            idm,
            tokenizer,
            goal_offset=25,
            execute_horizon=2,
            process={},
            transform={},
            device=torch.device("cpu"),
            cache_goal=True,
            do_sample=False,
            temperature=1.0,
            top_k=None,
        )
        policy.set_env(Env())
        info = {
            "pixels": np.zeros((2, 1, 8), dtype=np.float32),
            "goal": np.ones((2, 1, 8), dtype=np.float32),
        }
        policy.get_action(info)
        policy.get_action(info)
        self.assertEqual(policy.num_replans, 1)
        policy.get_action(info)
        self.assertEqual(policy.num_replans, 2)
        self.assertEqual(idm.seen_steps, [[25, 25], [23, 23]])


if __name__ == "__main__":
    unittest.main()
