#!/usr/bin/env python3
import argparse
import asyncio
import re
import sys

try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
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


async def safe_land(drone: System, timeout: float) -> None:
    try:
        await drone.action.land()
        await wait_for_async_value(drone.telemetry.in_air(), lambda value: not bool(value), timeout=timeout)
    finally:
        try:
            await drone.action.disarm()
        except Exception:
            pass


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    parser.add_argument("--forward-m-s", type=float, default=0.0)
    parser.add_argument("--right-m-s", type=float, default=0.0)
    parser.add_argument("--down-m-s", type=float, default=0.0)
    parser.add_argument("--yawspeed-deg-s", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=2.0)
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
    await asyncio.sleep(3.0)

    zero_setpoint = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    nudge_setpoint = VelocityBodyYawspeed(
        args.forward_m_s,
        args.right_m_s,
        args.down_m_s,
        args.yawspeed_deg_s,
    )

    print("Sending initial zero body-velocity setpoint before starting Offboard.")
    await drone.offboard.set_velocity_body(zero_setpoint)

    try:
        print("Starting Offboard mode.")
        await drone.offboard.start()
        print(
            "Sending nudge: "
            f"forward_m_s={args.forward_m_s}, "
            f"right_m_s={args.right_m_s}, "
            f"down_m_s={args.down_m_s}, "
            f"yawspeed_deg_s={args.yawspeed_deg_s}, "
            f"duration={args.duration}"
        )
        await drone.offboard.set_velocity_body(nudge_setpoint)
        await asyncio.sleep(args.duration)
        await drone.offboard.set_velocity_body(zero_setpoint)
        print("Stopping Offboard mode.")
        await drone.offboard.stop()
    except OffboardError as exc:
        print(f"Offboard control failed: {exc}", file=sys.stderr)
        await safe_land(drone, timeout=args.timeout)
        return 1

    print("Landing.")
    await safe_land(drone, timeout=args.timeout)
    print("Offboard nudge sequence completed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
