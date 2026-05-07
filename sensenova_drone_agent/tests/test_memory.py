from sensenova_drone.memory import MemoryEntry, RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.world_state import ActionSequence, KairosActionCondition, PredictedFuture


def test_real_observation_memory_rejects_predicted_future() -> None:
    memory = RealObservationMemory()
    future = PredictedFuture(
        action_condition=KairosActionCondition(
            action_sequence=ActionSequence(actions=[], commands=[], horizon_steps=0)
        )
    )

    try:
        memory.append(future)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("PredictedFuture should not be accepted into real observation memory.")


def test_real_observation_memory_accepts_real_entry() -> None:
    memory = RealObservationMemory()
    entry = MemoryEntry(observation=Observation(frame_rgb="frame"), metadata={"source": "real_gazebo_camera"})
    memory.append(entry)
    assert len(memory) == 1
