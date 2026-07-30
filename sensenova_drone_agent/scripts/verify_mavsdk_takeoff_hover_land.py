#!/usr/bin/env python3
import argparse
import asyncio
import re
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
SAFE_LOCAL_CONNECTION = re.compile(r"^udpin://(0\.0\.0\.0|127\.0\.0\.1|localhost):14540$")


async def wait_connected(drone: System, timeout: float) -> None:
    async def _wait() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_wait(), timeout=timeout)


async def wait_for_async_value(async_iterable, predicate, timeout: float) -> None:
    async def _wait() -> None:
        async for value in async_iterable:
            if predicate(value):
                return

    await asyncio.wait_for(_wait(), timeout=timeout)


def validate_sitl_guard(args: argparse.Namespace) -> None:
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")
    if not SAFE_LOCAL_CONNECTION.match(args.connection):
        raise SystemExit(
            "Refusing to run because the connection string is not the local PX4 SITL UDP endpoint."
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    parser.add_argument("--hover-seconds", type=float, default=5.0)
    parser.add_argument("--takeoff-altitude-m", type=float, default=2.5)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    args = parser.parse_args()

    validate_sitl_guard(args)

    drone = System()
    print(f"Connecting to PX4 SITL via {args.connection}")
    try:
        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await wait_connected(drone, timeout=args.timeout)
    except asyncio.TimeoutError:
        print(
            f"Timed out while connecting to PX4 SITL at {args.connection}. "
            "Start the local simulator before running this script.",
            file=sys.stderr,
        )
        return 1
    print("Connected to vehicle.")

    await drone.action.set_takeoff_altitude(args.takeoff_altitude_m)

    print("Arming.")
    await drone.action.arm()

    print("Takeoff.")
    await drone.action.takeoff()
    await wait_for_async_value(drone.telemetry.in_air(), lambda value: bool(value), timeout=args.timeout)

    print(f"Hovering for {args.hover_seconds:.1f} seconds.")
    await asyncio.sleep(args.hover_seconds)

    print("Landing.")
    await drone.action.land()
    await wait_for_async_value(drone.telemetry.in_air(), lambda value: not bool(value), timeout=args.timeout)

    try:
        await asyncio.wait_for(
            wait_for_async_value(drone.telemetry.armed(), lambda value: not bool(value), timeout=args.timeout),
            timeout=args.timeout,
        )
    except asyncio.TimeoutError:
        print("Vehicle still armed after landing. Sending disarm.")
        await drone.action.disarm()

    print("Takeoff, hover, and landing sequence completed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
