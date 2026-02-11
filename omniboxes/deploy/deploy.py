"""
OmniBoxes Unified Deployment Script

Deploys OmniBoxes browser automation infrastructure on a single machine.
Supports both local development and production deployments with nginx.

Usage:
    Local dev:     python deploy.py 10
    Production:    python deploy.py 10 --nginx
    Custom ports:  python deploy.py 10 --nginx --master-port 7000
"""

import argparse
import sys
import time

from process_manager import OmniboxesLauncher
from nginx_manager import NginxManager


def main():
    parser = argparse.ArgumentParser(
        description="Unified OmniBoxes deployment for local dev and production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local dev:      python deploy.py 10
  Production:     python deploy.py 10 --nginx
  Custom ports:   python deploy.py 20 --nginx --master-port 7500

For more information, see omniboxes/deploy/README.md
        """
    )

    parser.add_argument(
        "instances",
        type=int,
        help="Number of browser instances to launch"
    )

    parser.add_argument(
        "--nginx",
        action="store_true",
        help="Setup nginx reverse proxy for external access (requires sudo)"
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=7000,
        help="Port for master server (default: 7000)"
    )

    parser.add_argument(
        "--node-port",
        type=int,
        default=8080,
        help="Port for node server (default: 8080)"
    )

    parser.add_argument(
        "--instance-start-port",
        type=int,
        default=9000,
        help="Starting port for instance servers (default: 9000)"
    )

    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Port for Redis server (default: 6379)"
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum parallel instance starts (default: auto-calculated)"
    )

    parser.add_argument(
        "--disable-recovery",
        action="store_true",
        help="Disable automatic process recovery"
    )

    args = parser.parse_args()

    if args.instances < 1:
        print("❌ Error: Number of instances must be at least 1")
        sys.exit(1)

    print("=" * 60)
    print("🚀 OMNIBOXES UNIFIED DEPLOYMENT")
    print("=" * 60)
    print(f"\nMode: {'Production (with nginx)' if args.nginx else 'Local Development'}")
    print(f"Instances: {args.instances}")
    print("=" * 60 + "\n")

    # Step 1: Setup nginx if requested
    if args.nginx:
        nginx_manager = NginxManager(args.master_port)

        if not nginx_manager.setup():
            print("\n❌ nginx setup failed. Exiting.")
            sys.exit(1)

        print("Waiting 2 seconds before starting OmniBoxes servers...\n")
        time.sleep(2)

    # Step 2: Launch OmniBoxes servers
    print("=" * 60)
    print("🎯 STARTING OMNIBOXES SERVERS")
    print("=" * 60 + "\n")

    # Create and configure launcher
    launcher = OmniboxesLauncher(args.instances)
    launcher.instance_start_port = args.instance_start_port
    launcher.node_port = args.node_port
    launcher.master_port = args.master_port
    launcher.redis_port = args.redis_port

    if args.max_parallel:
        launcher.max_parallel_starts = args.max_parallel

    if args.disable_recovery:
        launcher.recovery_enabled = False

    # Run the launcher
    launcher.run()


if __name__ == "__main__":
    main()
