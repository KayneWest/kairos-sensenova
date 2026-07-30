#!/usr/bin/env python3
import argparse
import asyncio
import sys

try:
    from mavsdk import System
except ModuleNotFoundError as exc:
    if exc.name == "mavsdk":
        print(
            "mavsdk is not installed in this Python environment. "
            "Run this script via `docker compose -f docker-compose.yml run --rm tools ...` "
            "or activate the dedicated MAVSDK environment first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise


DEFAULT_CONNECTION = "udpin://0.0.0.0:14540"


async def first_value(async_iterable, timeout: float):
    return await asyncio.wait_for(anext(async_iterable), timeout=timeout)


async def wait_connected(drone: System, timeout: float) -> None:
    async def _wait() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_wait(), timeout=timeout)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    drone = System()
    print(f"Connecting to PX4 SITL via {args.connection}")
    try:
        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await wait_connected(drone, timeout=args.timeout)
    except asyncio.TimeoutError:
        print(
            f"Timed out while connecting to PX4 SITL at {args.connection}. "
            "Make sure the simulator is running and sending heartbeats.",
            file=sys.stderr,
        )
        return 1
    print("Connected to vehicle.")

    try:
        armed = await first_value(drone.telemetry.armed(), args.timeout)
        in_air = await first_value(drone.telemetry.in_air(), args.timeout)
        flight_mode = await first_value(drone.telemetry.flight_mode(), args.timeout)
    except asyncio.TimeoutError:
        print("Timed out while waiting for core telemetry.", file=sys.stderr)
        return 1

    print(f"armed={armed}")
    print(f"in_air={in_air}")
    print(f"flight_mode={getattr(flight_mode, 'name', flight_mode)}")

    try:
        position = await first_value(drone.telemetry.position(), 5.0)
        print(
            "position="
            f"lat={position.latitude_deg:.7f}, "
            f"lon={position.longitude_deg:.7f}, "
            f"abs_alt_m={position.absolute_altitude_m:.2f}, "
            f"rel_alt_m={position.relative_altitude_m:.2f}"
        )
    except asyncio.TimeoutError:
        print("position=unavailable")

    try:
        battery = await first_value(drone.telemetry.battery(), 5.0)
        print(f"battery_remaining_percent={battery.remaining_percent:.3f}")
    except asyncio.TimeoutError:
        print("battery=unavailable")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
