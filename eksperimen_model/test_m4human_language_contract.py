"""Offline contract checks. Tiny LM validates autograd; real Qwen gate remains separate."""
import tempfile
import json
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from eksperimen_model.models.m4human_projector import FrozenM4HumanLanguage, M4HumanPhysicalProjectorV1, answer_only_loss
from eksperimen_model.datasets.m4human_qa_dataset import M4HumanQADataset, canonical_answer, read_qa, validate_qa
from eksperimen_model.datasets import generate_m4human_qa
from eksperimen_model.datasets.generate_m4human_qa import generate_records, windows_from_h_cache
from eksperimen_model.m4human_evaluation import parse_response, score_rows, donor_pairs, score_donors, paired_bootstrap, verify_test_contract
from eksperimen_model.evaluate_m4human import create_test_lock
from eksperimen_model.utils.m4human_kinematics import DEFAULT_RECIPE, TASK_DEFINITIONS
from eksperimen_model.utils.m4human_runtime import CONTRACT_VERSION, canonical_hash, file_sha256, save_checkpoint, state_dict_hash, validate_run_artifacts


class TinyTokenizer:
    pad_token_id, eos_token_id = 0, 1
    all_special_tokens = ["<EOS>", "<USER>", "<ASSISTANT>"]
    chat_template = "tiny-test-template"

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            if text.startswith("<EOS>"):
                ids.append(1)
                text = text[5:]
            else:
                ids.append(ord(text[0]) + 2)
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token - 2) for token in ids if token > 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f'<{row["role"].upper()}>\n{row["content"]}' + ("<EOS>" if row["role"] == "assistant" else "\n") for row in messages)
        return rendered + ("<ASSISTANT>\n" if add_generation_prompt else "")


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embed = nn.Embedding(512, 12)
        self.head = nn.Linear(12, 512)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache=False):
        valid = attention_mask[..., None]
        hidden = (inputs_embeds * valid).cumsum(1) / valid.cumsum(1).clamp_min(1)
        return SimpleNamespace(logits=self.head(hidden))


def row(identifier, answer="speeding_up", subject="S1", recording="R1"):
    return dict(qa_id=identifier, window_id="W" + identifier, subject_id=subject, recording_id=recording, split="val", task="root_speed_trend", question="Same question", answer=answer, target_status="defined", validity="valid", recipe_hash="r", joint_map_hash="j", coordinate_hash="c", split_hash="s", action_id="walk", task_interval_s=[.5, 2.5], observation_interval_s=[0., 31/12], time_s=[i/12 for i in range(32)], label_support=list(range(6,32)))


def main():
    from eksperimen_model.evaluate_m4human import majority_predictions
    subgroup_rows=[dict(row('arms'),task='relative_limb_motion',body_part='arms',answer='faster_left'),
                   dict(row('legs'),task='relative_limb_motion',body_part='legs',answer='faster_right')]
    assert [json.loads(r['raw_text'])['answer'] for r in majority_predictions(subgroup_rows,subgroup_rows)]==['faster_left','faster_right']
    torch.manual_seed(11)
    tokenizer = TinyTokenizer()
    model = FrozenM4HumanLanguage(TinyCausalLM(), tokenizer, max_prefix_tokens=1024, max_total_tokens=1200)
    U = torch.randn(16, 256)
    sample = dict(U=U, token_mask=torch.tensor([True] * 15 + [False]), question="Physical question?", task="root_speed_trend", answers=TASK_DEFINITIONS["root_speed_trend"]["answers"], canonical_answer=canonical_answer("root_speed_trend", "speeding_up"))
    inputs, labels = model.assemble([sample], training=True)
    alternate = dict(sample, canonical_answer=canonical_answer("root_speed_trend", "slowing_down"))
    altered, labels2 = model.assemble([alternate], training=True)
    boundary = int((labels[0] != -100).nonzero()[0])
    assert torch.equal(inputs["inputs_embeds"][:, :boundary], altered["inputs_embeds"][:, :boundary])
    assert torch.equal(inputs["attention_mask"][:, :boundary], altered["attention_mask"][:, :boundary])
    assert torch.equal(inputs["position_ids"][:, :boundary], altered["position_ids"][:, :boundary])
    assert (labels[:, :boundary] == -100).all()
    assert labels[0, -1] == tokenizer.eos_token_id
    initial_llm, initial_projector = state_dict_hash(model.llm), state_dict_hash(model.projector)
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-3)
    loss = model([sample])
    loss.backward()
    for layer in (model.projector.net[1], model.projector.net[3]):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.llm.parameters())
    optimizer.step()
    assert state_dict_hash(model.llm) == initial_llm
    assert state_dict_hash(model.projector) != initial_projector
    assert sum(parameter.numel() for parameter in M4HumanPhysicalProjectorV1().parameters()) == 920064
    second = dict(sample, question="Q?")
    batch, _ = model.assemble([sample, second])
    assert batch["attention_mask"][1, 0] == 0
    individual, _ = model.assemble([second])
    a = model.llm(**batch).logits[1, -1]
    b = model.llm(**individual).logits[0, -1]
    assert torch.allclose(a, b, atol=1e-6)
    raw, first = model.reference_generate([sample], max_new_tokens=2)
    prefix, _ = model.assemble([sample])
    expected = model.llm(**prefix).logits[:, -1]
    assert torch.allclose(first, expected)
    assert raw[0]["raw_ids"][0] == expected.argmax(-1).item()
    reference = [row("1"), row("2")]
    valid = canonical_answer("root_speed_trend", "speeding_up")
    predictions = [{"qa_id": "1", "raw_text": valid}, {"qa_id": "2", "raw_text": "invalid"}]
    score = score_rows(reference, predictions, ["root_speed_trend"])
    assert abs(score["tasks"]["root_speed_trend"]["f1"]["speeding_up"] - 2 / 3) < 1e-12
    assert abs(score["score"] - 1 / 6) < 1e-12
    for invalid in ('{"task":"root_speed_trend","answer":"speeding_up","answer":"unknown"}', valid + " text", "```json\n" + valid + "\n```", '{"task":"other","answer":"speeding_up"}', '{"task":"root_speed_trend","answer":"speeding_up","extra":1}'):
        assert not parse_response(invalid, "root_speed_trend")["parse_valid"]
    pair_rows = [row("a"), row("b", "slowing_down", "S2", "R2")]
    panel = donor_pairs(pair_rows)
    assert len(panel["pairs"]) == 2
    pair_predictions = [{"qa_id": row_["qa_id"], "raw_text": canonical_answer(row_["task"], row_["answer"])} for row_ in pair_rows]
    grounding = score_donors(panel, pair_rows, pair_predictions)
    assert grounding["PairBothCorrect"] == grounding["CorrectChange"] == 1
    pair_rows[1]["answer"] = "speeding_up"
    assert not donor_pairs(pair_rows)["pairs"]
    bootstrap = paired_bootstrap(reference, predictions, predictions, ["root_speed_trend"], draws=20)
    assert bootstrap["delta"] == 0 and bootstrap["ci95"] is None
    unsupported = paired_bootstrap([row("s1", subject="S1"), row("s2", subject="S2"), row("s3", subject="S3")],
        [], [], ["root_speed_trend", "relative_limb_motion"], draws=20)
    assert unsupported["unsupported_resamples"] == 20 and unsupported["ci95"] is None
    validate_qa(reference[0])
    conflicting = dict(reference[0], validity="unknown")
    try:
        validate_qa(conflicting)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting QA status accepted")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        hroot = Path(directory) / "H"
        hroot.mkdir()
        times = torch.arange(32, dtype=torch.float64) / 12
        joint_map = {"audited": True, "task_groups": {}}
        joint_map_path = Path(directory) / "joint_map.json"
        joint_map_path.write_text(json.dumps(joint_map))
        hlineage = dict(joint_map_hash=file_sha256(joint_map_path), recipe_hash=canonical_hash(DEFAULT_RECIPE),
                        target_hash="audited-target", coordinate_hash="radar-m", split_hash="subject-split")
        target = {"P_relative_m": torch.zeros(32, 22, 3), "r_m": torch.zeros(32, 3),
                  "v_relative_mps": torch.zeros(32, 22, 3), "v_root_mps": torch.zeros(32, 3),
                  "position_valid_joint": torch.ones(32, 22, dtype=torch.bool),
                  "position_valid_root": torch.ones(32, dtype=torch.bool),
                  "velocity_valid_joint": torch.tensor([False] * 6 + [True] * 26)[:, None].expand(32, 22).clone(),
                  "velocity_valid_root": torch.tensor([False] * 6 + [True] * 26)}
        entries = []
        for number in range(2):
            tensor_path = hroot / f"window{number}.pt"
            torch.save({"time_s": times}, tensor_path)
            entry = dict(window_id=f"H{number}", subject_id=7, recording_id="pkg:7:0", split="val", action_id=0,
                         frame_uids=[f"F{number}_{i}" for i in range(32)], source_keys=[f"K{number}_{i}" for i in range(32)],
                         window_support_hash=f"support{number}", tensor_path=tensor_path.name, tensor_sha256=file_sha256(tensor_path))
            if number == 0:
                target_path = hroot / "target.pt"
                torch.save(target, target_path)
                entry.update(target_path=target_path.name, target_sha256=file_sha256(target_path))
            entries.append(entry)
        index_path = hroot / "index.jsonl"
        index_path.write_text("".join(json.dumps(item) + "\n" for item in entries))
        (hroot / "metadata.json").write_text(json.dumps({"contract_version": CONTRACT_VERSION, "complete": True,
            "representation": "full_H", "lineage": hlineage, "count": 2, "index_sha256": file_sha256(index_path)}))
        generated, _ = generate_records(windows_from_h_cache(hroot, hlineage["joint_map_hash"], hlineage["recipe_hash"]),
                                        joint_map, DEFAULT_RECIPE, ["root_speed_trend"])
        assert generated[0]["answer"] == "no_overall_trend" and generated[0]["label_support"] == list(range(6, 32))
        assert generated[1]["target_status"] == "undefined" and generated[1]["reason"] == "annotation_not_audited"
        assert generated[0]["h_cache_hash"] == canonical_hash(json.loads((hroot / "metadata.json").read_text()))
        generated_path = Path(directory) / "generated_qa.jsonl"
        with patch.object(sys, "argv", ["generate_m4human_qa", "--h-cache", str(hroot),
                                        "--joint-map", str(joint_map_path), "--output", str(generated_path),
                                        "--tasks", "root_speed_trend"]):
            generate_m4human_qa.main()
        assert read_qa(generated_path) == generated
        uroot = Path(directory) / "U"
        uroot.mkdir()
        u_tensor = uroot / "window.pt"
        torch.save({"U": torch.zeros(16, 256), "token_mask": torch.ones(16, dtype=torch.bool), "time_s": times}, u_tensor)
        u_index = [{**entries[0], "tensor_path": u_tensor.name, "tensor_sha256": file_sha256(u_tensor),
                    "K": 16, "condition": "C_base"}]
        (uroot / "index.jsonl").write_text(json.dumps(u_index[0]) + "\n")
        (uroot / "metadata.json").write_text(json.dumps({"contract_version": CONTRACT_VERSION, "complete": True,
            "representation": "exact_U", "condition": "C_base", "K": 16,
            "lineage": {**hlineage, "h_cache_hash": canonical_hash(json.loads((hroot / "metadata.json").read_text())),
                        "tokenizer_hash": "tokenizer"}, "index_sha256": file_sha256(uroot / "index.jsonl")}))
        joined = M4HumanQADataset(generated_path, uroot, "val", "C_base", hlineage)
        assert len(joined) == 1 and joined[0]["U"].shape == (16, 256) and joined.records[0]["subject_id"] == 7
        for field in ("recipe_hash", "joint_map_hash", "split_hash", "coordinate_hash", "gt_source_hash", "h_cache_hash", "window_support_hash", "source_keys", "frame_ids"):
            tampered = [dict(item) for item in generated]
            tampered[0][field] = "changed"
            changed_path = Path(directory) / f"tampered_{field}.jsonl"
            changed_path.write_text("".join(json.dumps(item) + "\n" for item in tampered))
            try:
                M4HumanQADataset(changed_path, uroot, "val", "C_base", hlineage)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Changed QA {field} accepted by exact-U join")
        lock = Path(directory) / "lock.json"
        lock.write_text("{}")
        try:
            verify_test_contract(lock)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete test lock accepted")
        lock.unlink()
        lineage = dict(encoder_hash="e", motion_hash="m", recipe_hash="r", joint_map_hash="j", split_hash="s")
        config = {"lineage": lineage, "primary_budget": 16, "selected_tasks": ["root_speed_trend"], "generation": {"max_new_tokens": 3}}
        (Path(directory) / "config.yaml").write_text(__import__("yaml").safe_dump(config))
        (Path(directory) / "effect.json").write_text('{"macro_f1":0.05}')
        (Path(directory) / "uncertainty.json").write_text('{"method":"paired_subject_cluster_row_weighted","draws":20,"seed":42}')
        test_rows = [dict(item, split="test") for item in [row("a"), row("b", "slowing_down", "S2", "R2")]]
        qa_path = Path(directory) / "qa.jsonl"
        qa_path.write_text("".join(json.dumps(item) + "\n" for item in test_rows))
        assert len(read_qa(qa_path)) == 2
        duplicate_qa = Path(directory) / "duplicate_qa.jsonl"
        duplicate_qa.write_text(qa_path.read_text().replace('"qa_id": "a"', '"qa_id": "a", "qa_id": "changed"'))
        try:
            read_qa(duplicate_qa)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate reference JSON field accepted")
        paths = {}
        for condition in ("C_base", "C_kin"):
            cache = Path(directory) / f"{condition}.json"
            cache.write_text(json.dumps({"contract_version": CONTRACT_VERSION, "lineage": lineage,
                "condition": condition, "representation": "exact_U", "complete": True, "K": 16,
                "data_kind": "m4human", "scientific_eligible": True, "tokenizer_hash": condition}))
            run = Path(directory) / condition
            run.mkdir()
            checkpoint = run / "best.pt"
            save_checkpoint(checkpoint, M4HumanPhysicalProjectorV1(12), dict(lineage, condition=condition, K=16,
                tokenizer_hash=condition, cache_sha256=file_sha256(cache), qa_sha256=file_sha256(qa_path),
                config_hash=canonical_hash(config),
                initial_sha256="same-initial"))
            (run / "metrics.json").write_text(json.dumps({"status": "complete", "analysis_ready": True,
                "exposure_hash": "same-exposures", "successful_updates": 2, "n_QA_exposures": 32}))
            paths[condition] = (cache, checkpoint)
        args = SimpleNamespace(config=str(Path(directory) / "config.yaml"), base_cache=str(paths["C_base"][0]),
            kin_cache=str(paths["C_kin"][0]), base_checkpoint=str(paths["C_base"][1]),
            kin_checkpoint=str(paths["C_kin"][1]), effect_tolerance_json=str(Path(directory) / "effect.json"),
            uncertainty_json=str(Path(directory) / "uncertainty.json"), qa=str(qa_path), output=str(lock))
        with patch("eksperimen_model.evaluate_m4human.validate_run_artifacts", return_value=[]):
            create_test_lock(args, test_rows)
        verified = verify_test_contract(lock, qa_path, paths["C_base"][0], config,
            checkpoint_paths={"C_base": paths["C_base"][1]}, condition="C_base")
        assert verified["donor_panel"]["pairs"] and verified["checkpoint_sha256"]["C_base"] == file_sha256(paths["C_base"][1])
        paths["C_base"][1].write_bytes(b"changed")
        try:
            verify_test_contract(lock, checkpoint_paths={"C_base": paths["C_base"][1]})
        except ValueError:
            pass
        else:
            raise AssertionError("Changed selected checkpoint accepted by test lock")
    with tempfile.TemporaryDirectory() as directory:
        from eksperimen_model import train_m4human_projector as trainer
        import yaml
        root = Path(directory)
        lineage = dict(encoder_hash="e", motion_hash="m", recipe_hash="r", joint_map_hash="j", split_hash="s")
        config = {"lineage": lineage, "data_kind": "synthetic", "llm": {"revision": "a" * 40}, "primary_budget": 16,
                  "selected_tasks": ["root_speed_trend"], "generation": {"max_new_tokens": 2},
                  "training": {"seed": 4, "micro_batch": 1, "accumulation": 1, "max_successful_updates": 2,
                               "epochs_cap": 2, "precision": "fp32", "weight_decay": 0.01, "learning_rate": 1e-3,
                               "warmup_fraction": 0.05, "clip_grad_norm": 1.0, "max_prefix_tokens": 1024,
                               "max_total_tokens": 1200}}
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        qa_path, cache_path, initial_path, run_path = (root / name for name in ("qa.jsonl", "cache.json", "initial.pt", "run"))
        qa_path.write_text(json.dumps(dict(row("train"), split="train")) + "\n" + json.dumps(row("val")) + "\n")
        cache_path.write_text("{}")
        from transformers import Qwen2Config, Qwen2ForCausalLM
        def new_tiny_qwen():
            torch.manual_seed(7)
            qconfig = Qwen2Config(vocab_size=512, hidden_size=24, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=2048, eos_token_id=1, pad_token_id=0,
                attention_dropout=0.0, bos_token_id=2)
            return FrozenM4HumanLanguage(Qwen2ForCausalLM(qconfig), TinyTokenizer(), 1024, 1200)
        save_checkpoint(initial_path, new_tiny_qwen().projector,
                        dict(lineage, llm_revision="a" * 40, seed=4, role="paired_initial"))
        class FakeDataset:
            fail_on_second_train_access = False
            train_accesses = 0
            def __init__(self, _qa, _cache, split, condition, _lineage):
                self.split = split
                self.records = [dict(row(split), split=split)]
                self.metadata = {"K": 16, "tokenizer_hash": "tok", "condition": condition,
                                 "data_kind": "m4human", "scientific_eligible": True}
            def __len__(self):
                return 1
            def __getitem__(self, index):
                if self.split == "train":
                    FakeDataset.train_accesses += 1
                    if FakeDataset.fail_on_second_train_access and FakeDataset.train_accesses == 2:
                        raise RuntimeError("intentional interruption after epoch boundary")
                return dict(U=U.clone(), token_mask=sample["token_mask"].clone(),
                            qa_id=self.records[index]["qa_id"], window_id=self.records[index]["window_id"],
                            question=sample["question"], task=sample["task"], answers=sample["answers"],
                            canonical_answer=sample["canonical_answer"])
        argv = ["train_m4human_projector", "--config", str(config_path), "--qa", str(qa_path), "--u-cache", str(cache_path),
                "--condition", "C_base", "--initial", str(initial_path), "--output", str(run_path), "--device", "cpu"]
        with patch.object(trainer, "M4HumanQADataset", FakeDataset), patch.object(trainer, "load_frozen_qwen", side_effect=lambda *_: new_tiny_qwen()):
            with patch.object(sys, "argv", argv):
                trainer.main()
            interrupted = root / "interrupted"
            FakeDataset.fail_on_second_train_access = True
            FakeDataset.train_accesses = 0
            with patch.object(sys, "argv", [*argv[:-4], "--output", str(interrupted), "--device", "cpu"]):
                try:
                    trainer.main()
                except RuntimeError as error:
                    assert "intentional interruption" in str(error)
                else:
                    raise AssertionError("Intentional interrupted run completed")
            assert (interrupted / "last.pt").exists() and (interrupted / "best.pt").exists()
            FakeDataset.fail_on_second_train_access = False
            resumed_run = root / "resumed"
            with patch.object(sys, "argv", [*argv, "--resume", str(interrupted / "last.pt")]):
                sys.argv[sys.argv.index("--output") + 1] = str(resumed_run)
                trainer.main()
        assert not validate_run_artifacts(run_path)
        assert json.loads((run_path / "metrics.json").read_text())["analysis_ready"] is True
        assert not validate_run_artifacts(resumed_run)
        assert state_dict_hash(torch.load(run_path / "last.pt", weights_only=True)["state_dict"]) == state_dict_hash(torch.load(resumed_run / "last.pt", weights_only=True)["state_dict"])
    # Exercise the actual pinned Transformers Qwen2 code without downloading assets.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(vocab_size=512, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, eos_token_id=1, pad_token_id=0,
        attention_dropout=0.0, bos_token_id=2)
    qwen = FrozenM4HumanLanguage(Qwen2ForCausalLM(config), tokenizer, 1024, 1200)
    qwen([sample]).backward()
    assert qwen.projector.net[1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in qwen.llm.parameters())
    batched, logits = qwen.generate([sample, second], max_new_tokens=3)
    alone, logits_alone = qwen.generate([second], max_new_tokens=3)
    assert torch.allclose(logits[1], logits_alone[0], atol=1e-5)
    assert batched[1]["raw_ids"] == alone[0]["raw_ids"]
    reference_ids, _ = qwen.reference_generate([sample, second], max_new_tokens=3)
    assert [row["raw_ids"] for row in reference_ids] == [row["raw_ids"] for row in batched]
    print("Pinned random tiny Qwen2 architecture: PASS (pretrained checkpoint gate not_run)")
    print("M4Human language contract: PASS (offline tiny LM; actual Qwen gate not_run)")


if __name__ == "__main__":
    main()
