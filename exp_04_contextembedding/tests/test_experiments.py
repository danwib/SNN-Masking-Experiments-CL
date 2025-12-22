from pathlib import Path

import torch
import torch.nn.functional as F

from masking_experiments.config import (
    DatasetConfig,
    ExperimentConfig,
    MaskingConfig,
    ModelConfig,
    SleepPhaseConfig,
    SoftColumnMaskConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
)
from masking_experiments.data import TaskDataset
from masking_experiments.experiments import run_experiment
from masking_experiments.masking import MaskController
from masking_experiments.snn import SparseSNN


def _base_tasks():
    return [
        TaskConfig(name="Task 1", prompt="Task 1", column_index=0, dataset_variant="task1"),
        TaskConfig(name="Task 2", prompt="Task 2", column_index=1, dataset_variant="task2"),
        TaskConfig(
            name="Task 1'", prompt="Task 1'", column_index=2, dataset_variant="task1_prime", held_out=True
        ),
        TaskConfig(
            name="Task 2'", prompt="Task 2'", column_index=3, dataset_variant="task2_prime", held_out=True
        ),
    ]


def _shared_column_tasks():
    return [
        TaskConfig(name="Task 1", prompt="Task 1", column_index=0, dataset_variant="task1"),
        TaskConfig(name="Task 2", prompt="Task 2", column_index=1, dataset_variant="task2"),
        TaskConfig(
            name="Task 1'", prompt="Task 1'", column_index=0, dataset_variant="task1_prime"
        ),
        TaskConfig(
            name="Task 2'", prompt="Task 2'", column_index=1, dataset_variant="task2_prime"
        ),
    ]


def _residual_tasks():
    return [
        TaskConfig(name="Task 1", prompt="Task 1", column_index=0, dataset_variant="task1"),
        TaskConfig(name="Task 2", prompt="Task 2", column_index=1, dataset_variant="task2"),
        TaskConfig(
            name="Task 1'",
            prompt="Task 1'",
            column_index=2,
            dataset_variant="task1_prime",
            residual_from="Task 1",
        ),
        TaskConfig(
            name="Task 2'",
            prompt="Task 2'",
            column_index=3,
            dataset_variant="task2_prime",
            residual_from="Task 2",
        ),
    ]


def _sleep_tasks():
    return [
        TaskConfig(name="Task 1", prompt="Task 1", column_index=0, dataset_variant="task1"),
        TaskConfig(name="Task 2", prompt="Task 2", column_index=1, dataset_variant="task2"),
        TaskConfig(
            name="Task 1'",
            prompt="Task 1'",
            column_index=2,
            dataset_variant="task1_prime",
            consolidate_into="Task 1",
        ),
        TaskConfig(
            name="Task 2'",
            prompt="Task 2'",
            column_index=3,
            dataset_variant="task2_prime",
            consolidate_into="Task 2",
        ),
    ]


def _mixed_mnist_fashion_tasks():
    return [
        TaskConfig(name="Digits", prompt="Digits", column_index=0, dataset_variant="task1"),
        TaskConfig(
            name="Clothing",
            prompt="Clothing",
            column_index=1,
            dataset_variant="fashion_task1",
            dataset_name="fashion_mnist",
        ),
    ]


def _make_config(experiment_type: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"test_{experiment_type}",
        experiment_type=experiment_type,
        tasks=_base_tasks(),
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=32,
            base_learning_rate=0.01,
            seed=3,
            task_schedule=["Task 1", "Task 2"],
        ),
        masking=MaskingConfig(learning_rate_scale=0.2, threshold_shift=0.3, prompt_vector_dim=8),
        model=ModelConfig(hidden_per_column=16, time_steps=3, beta=0.9),
    )


def test_parallel_experiment_runs():
    config = _make_config("parallel")
    result = run_experiment(config)
    assert result.stages[0].label == "parallel_training"
    for metric in result.stages[0].metrics:
        assert 0.0 <= metric.accuracy <= 1.0


def test_sequential_experiment_tracks_stages():
    config = _make_config("sequential")
    result = run_experiment(config)
    assert len(result.stages) == 3  # initial + two tasks
    assert result.stages[0].label == "initial"
    assert all(0.0 <= metric.accuracy <= 1.0 for metric in result.stages[-1].metrics)


def test_sequential_with_shared_columns():
    config = ExperimentConfig(
        name="shared_columns",
        experiment_type="sequential",
        tasks=_shared_column_tasks(),
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=32,
            base_learning_rate=0.01,
            seed=5,
            task_schedule=["Task 1", "Task 2", "Task 1'", "Task 2'"],
        ),
        masking=MaskingConfig(learning_rate_scale=0.2, threshold_shift=0.3, prompt_vector_dim=8),
        model=ModelConfig(hidden_per_column=16, time_steps=3, beta=0.9),
    )
    result = run_experiment(config)
    assert result.stages[-1].label == "after_Task 2'"
    for stage in result.stages:
        for metric in stage.metrics:
            assert 0.0 <= metric.accuracy <= 1.0


def test_residual_mask_allows_extra_column_activity():
    config = ExperimentConfig(
        name="residual_columns",
        experiment_type="sequential",
        tasks=_residual_tasks(),
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=32,
            base_learning_rate=0.01,
            seed=7,
            task_schedule=["Task 1", "Task 2", "Task 1'", "Task 2'"],
        ),
        masking=MaskingConfig(learning_rate_scale=0.2, threshold_shift=0.3, prompt_vector_dim=8),
        model=ModelConfig(hidden_per_column=16, time_steps=3, beta=0.9),
    )
    result = run_experiment(config)
    assert result.stages[-1].label == "after_Task 2'"
    assert len(result.stages) == 5  # initial + 4 tasks
    for stage in result.stages:
        for metric in stage.metrics:
            assert 0.0 <= metric.accuracy <= 1.0


def test_sleep_phase_aligns_prompts_and_tracks_errors():
    config = ExperimentConfig(
        name="sleep_phase",
        experiment_type="sequential",
        tasks=_sleep_tasks(),
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=32,
            base_learning_rate=0.01,
            seed=11,
            task_schedule=["Task 1", "Task 2", "Task 1'", "Task 2'"],
        ),
        masking=MaskingConfig(learning_rate_scale=0.2, threshold_shift=0.3, prompt_vector_dim=8),
        model=ModelConfig(hidden_per_column=16, time_steps=3, beta=0.9),
        sleep=SleepPhaseConfig(
            enabled=True,
            epochs=1,
            batch_size=32,
            label_weight=0.5,
            distillation_weight=0.5,
            temperature=1.0,
        ),
    )
    result = run_experiment(config)
    labels = [stage.label for stage in result.stages]
    assert "before_sleep" in labels
    assert labels[-1] == "after_sleep"

    mapping = {"Task 1'": "Task 1", "Task 2'": "Task 2"}
    for metric in result.stages[-1].metrics:
        assert 0.0 <= metric.accuracy <= 1.0
        if metric.task_name in mapping:
            assert metric.prompt_used == mapping[metric.task_name]
            assert metric.error_overlap is not None
            assert 0.0 <= metric.error_overlap <= 1.0
        else:
            assert metric.prompt_used == metric.task_name


def test_soft_column_masks_log_and_train():
    config = ExperimentConfig(
        name="soft_columns_seq",
        experiment_type="sequential",
        tasks=_base_tasks()[:2],
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=16,
            base_learning_rate=0.01,
            seed=19,
            task_schedule=["Task 1", "Task 2"],
        ),
        masking=MaskingConfig(
            learning_rate_scale=0.2,
            threshold_shift=0.3,
            prompt_vector_dim=8,
            mode="soft_columns",
            soft_columns=SoftColumnMaskConfig(
                total_columns=4,
                base_columns=4,
                base_tasks=["Task 1", "Task 2"],
                novel_tasks=[],
                temperature=0.5,
                lambda_entropy=0.01,
                lambda_balance=0.01,
                mask_learning_rate=0.05,
            ),
        ),
        model=ModelConfig(hidden_per_column=8, time_steps=2, beta=0.9),
    )
    result = run_experiment(config)
    final_stage = result.stages[-1]
    assert final_stage.mask_logs is not None
    assert "Task 1" in final_stage.mask_logs
    assert "Task 2" in final_stage.mask_logs
    assert any(stage.metric_deltas for stage in result.stages if stage.label.startswith("after_Task 2"))


def test_soft_column_sleep_reports_masks():
    tasks = [
        TaskConfig(name="Task 1", prompt="Task 1", column_index=0, dataset_variant="task1"),
        TaskConfig(name="Task 2", prompt="Task 2", column_index=1, dataset_variant="task2"),
        TaskConfig(
            name="Task 1'",
            prompt="Task 1'",
            column_index=2,
            dataset_variant="task1_prime",
            consolidate_into="Task 1",
        ),
        TaskConfig(
            name="Task 2'",
            prompt="Task 2'",
            column_index=3,
            dataset_variant="task2_prime",
            consolidate_into="Task 2",
        ),
    ]
    config = ExperimentConfig(
        name="soft_columns_seq_sleep",
        experiment_type="sequential",
        tasks=tasks,
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=16,
            base_learning_rate=0.01,
            seed=29,
            task_schedule=["Task 1", "Task 2", "Task 1'", "Task 2'"],
        ),
        masking=MaskingConfig(
            learning_rate_scale=0.2,
            threshold_shift=0.3,
            prompt_vector_dim=8,
            mode="soft_columns",
            soft_columns=SoftColumnMaskConfig(
                total_columns=4,
                base_columns=2,
                base_tasks=["Task 1", "Task 2"],
                novel_tasks=["Task 1'", "Task 2'"],
                temperature=0.5,
                lambda_entropy=0.01,
                lambda_balance=0.01,
                lambda_overlap_novel=0.01,
                mask_learning_rate=0.05,
            ),
        ),
        model=ModelConfig(hidden_per_column=8, time_steps=2, beta=0.9),
        sleep=SleepPhaseConfig(
            enabled=True,
            epochs=1,
            batch_size=16,
            label_weight=0.1,
            distillation_weight=0.9,
            temperature=1.5,
        ),
    )
    result = run_experiment(config)
    labels = [stage.label for stage in result.stages]
    assert "before_sleep" in labels
    assert labels[-1] == "after_sleep"
    before_stage = next(stage for stage in result.stages if stage.label == "before_sleep")
    after_stage = result.stages[-1]
    assert before_stage.mask_logs is not None and "_occupancy" in before_stage.mask_logs
    assert after_stage.mask_logs is not None and "_occupancy" in after_stage.mask_logs


def test_mixed_mnist_and_fashion_tasks_run():
    config = ExperimentConfig(
        name="mnist_fashion_mix",
        experiment_type="sequential",
        tasks=_mixed_mnist_fashion_tasks(),
        dataset=DatasetConfig(
            root="tests/.fake_data",
            held_out_fraction=0.1,
            max_train_samples=64,
            use_fake_data=True,
            download=False,
            default_dataset="mnist",
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=16,
            base_learning_rate=0.01,
            seed=37,
            task_schedule=["Digits", "Clothing"],
        ),
        masking=MaskingConfig(learning_rate_scale=0.2, threshold_shift=0.3, prompt_vector_dim=8),
        model=ModelConfig(hidden_per_column=16, time_steps=3, beta=0.9),
    )
    result = run_experiment(config)
    assert result.stages[-1].label == "after_Clothing"
    assert all(0.0 <= metric.accuracy <= 1.0 for metric in result.stages[-1].metrics)


def test_learned_prompt_embeddings_receive_gradients():
    tasks = [
        TaskConfig(name="taskA", prompt="taskA", column_index=0, dataset_variant="task1", task_id=0),
        TaskConfig(name="taskB", prompt="taskB", column_index=1, dataset_variant="task2", task_id=1),
    ]
    masking_cfg = MaskingConfig(
        learning_rate_scale=0.5,
        threshold_shift=0.1,
        prompt_vector_dim=4,
        prompt_mode="learned_task_id",
        prompt_learning_rate=0.05,
    )
    controller = MaskController(tasks, masking_cfg)
    model_cfg = ModelConfig(hidden_per_column=4, time_steps=2, beta=0.9)
    model = SparseSNN(
        input_dim=6,
        prompt_dim=masking_cfg.prompt_vector_dim,
        num_columns=controller.num_columns,
        model_cfg=model_cfg,
        base_learning_rate=0.01,
        seed=5,
    )
    features = torch.randn(4, 6)
    labels = torch.rand(4)
    mask = controller.build_mask("taskA")
    outputs = model.forward(features, mask)
    logits = model._select_logits(outputs, mask, 0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    provider = controller._prompt_vectors
    assert provider._embedding is not None
    grad = provider._embedding.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[0]).item() > 0


def test_stage5A1_config_runs_with_fake_data():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage5A1_sequential_X_Xprime_soft_sleep_learned_taskid.yaml"
    )
    config = load_config(config_path)
    config.dataset.root = "tests/.fake_data"
    config.dataset.use_fake_data = True
    config.dataset.download = False
    config.dataset.max_train_samples = 64
    config.training.epochs = 1
    config.training.batch_size = 16
    config.training.base_learning_rate = 0.01
    config.sleep.epochs = 1
    config.sleep.batch_size = 16
    result = run_experiment(config)
    labels = [stage.label for stage in result.stages]
    assert "after_task1_prime" in labels
    assert labels[-1] == "after_sleep"


def test_wake_training_changes_between_initial_and_after_task1(monkeypatch):
    features_zero = torch.zeros(16, 4)
    features_one = torch.ones(16, 4)
    labels_zero = torch.zeros(16)
    labels_one = torch.ones(16)
    base_features = torch.cat([features_zero, features_one], dim=0)
    base_labels = torch.cat([labels_zero, labels_one], dim=0)

    def fake_build_datasets(tasks, dataset_cfg, seed):
        datasets = {}
        for task in tasks:
            datasets[task.name] = TaskDataset(
                train_features=base_features.clone(),
                train_labels=base_labels.clone(),
            )
        return datasets

    monkeypatch.setattr("masking_experiments.experiments.build_datasets", fake_build_datasets)
    config = _make_config("sequential")
    config.training.epochs = 5
    config.training.batch_size = 8
    config.training.base_learning_rate = 0.05
    result = run_experiment(config)
    after_task1 = next(stage for stage in result.stages if stage.label == "after_Task 1")
    assert after_task1.metric_deltas is not None
    assert any(abs(delta) > 1e-6 for delta in after_task1.metric_deltas.values())
