from masking_experiments.config import (
    DatasetConfig,
    ExperimentConfig,
    MaskingConfig,
    ModelConfig,
    SleepPhaseConfig,
    TaskConfig,
    TrainingConfig,
)
from masking_experiments.experiments import run_experiment


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
