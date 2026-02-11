#!/usr/bin/env python3
"""
OmniBoxes Multi-Node Deployment Script

Deploys OmniBoxes across multiple machines with automatic node discovery via Redis.

Deployment Modes:
-----------------
1. 'master' - Master node only (coordinates workers, no browsers)
   - Runs master server with Redis for node discovery
   - Use this on ONE machine that will coordinate all workers

2. 'worker' - Worker node only (browsers + node server)
   - Runs instance servers + node server
   - Automatically registers with master via Redis
   - Use this on ALL worker machines (can be many)

3. 'both' - Master + Worker on same machine
   - Runs everything: instance servers, node server, master server
   - Useful when you want master to also handle browsers

Architecture:
------------
One 'master' machine + Multiple 'worker' machines

Example with 3 machines:
  Machine 1 (10.0.0.10): --mode master --master-redis-host 10.0.0.10
  Machine 2 (10.0.0.11): --mode worker --master-redis-host 10.0.0.10
  Machine 3 (10.0.0.12): --mode worker --master-redis-host 10.0.0.10

Usage Examples:
--------------
# On master machine (IP: 192.168.1.100)
python deploy_multinode.py --mode master --master-redis-host 192.168.1.100

# On worker machine 1
python deploy_multinode.py 50 --mode worker --master-redis-host 192.168.1.100

# On worker machine 2
python deploy_multinode.py 50 --mode worker --master-redis-host 192.168.1.100

# Access master API at: http://192.168.1.100:7000
"""

import argparse
import sys
import socket

from multinode_manager import MultiNodeLauncher


def get_local_ip():
    """Get local IP address"""
    hostname = socket.gethostname()
    try:
        # Try to get IP by connecting to external address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        try:
            return socket.gethostbyname(hostname)
        except Exception:
            return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node OmniBoxes deployment with automatic node discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Examples:
--------------------

Setup with 3 machines (1 master + 2 workers):

  # Machine 1 (master): IP 192.168.1.100
  python deploy_multinode.py --mode master --master-redis-host 192.168.1.100

  # Machine 2 (worker): IP 192.168.1.101
  python deploy_multinode.py 50 --mode worker --master-redis-host 192.168.1.100

  # Machine 3 (worker): IP 192.168.1.102
  python deploy_multinode.py 50 --mode worker --master-redis-host 192.168.1.100

Access the master API at: http://192.168.1.100:7000

Node Discovery:
--------------
- Workers automatically register with master via Redis
- Master discovers workers every 10 seconds
- Workers send heartbeat every 30 seconds
- Dead workers are removed after 120 seconds

For full documentation, see: https://webgym.readthedocs.io/en/latest/multinode_deployment.html
        """
    )

    parser.add_argument(
        "instances",
        type=int,
        nargs='?',
        default=0,
        help="Number of browser instances (required for worker/both modes, ignored for master mode)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['master', 'worker', 'both'],
        default='both',
        help="Deployment mode: 'master' (coordinator only), 'worker' (browsers only), 'both' (combined)"
    )

    parser.add_argument(
        "--master-redis-host",
        type=str,
        default='localhost',
        help="Redis host on master node (master's IP address). Workers MUST specify master's IP."
    )

    parser.add_argument(
        "--master-redis-port",
        type=int,
        default=6379,
        help="Redis port on master (default: 6379)"
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=7000,
        help="Port for master server API (default: 7000)"
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
        help="Port for local Redis server (default: 6379, worker mode only)"
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum parallel instance starts (default: auto-calculated)"
    )

    parser.add_argument(
        "--advertise-ip",
        type=str,
        default=None,
        help="Public IP to register with master (default: auto-detect). Use when auto-detected IP is not reachable by master."
    )

    parser.add_argument(
        "--disable-recovery",
        action="store_true",
        help="Disable automatic process recovery"
    )

    args = parser.parse_args()

    # Validation
    if args.mode in ['worker', 'both'] and args.instances < 1:
        print(f"❌ Error: --mode {args.mode} requires specifying number of instances")
        print(f"   Example: python deploy_multinode.py 50 --mode {args.mode} --master-redis-host <master-ip>")
        sys.exit(1)

    if args.mode == 'worker' and args.master_redis_host == 'localhost':
        print("⚠️  WARNING: Worker mode with --master-redis-host localhost")
        print("   Worker nodes should connect to master's Redis!")
        print("   Did you mean: --master-redis-host <master-ip>")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Display configuration
    print("=" * 70)
    print("🚀 OMNIBOXES MULTI-NODE DEPLOYMENT")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   Master Redis: {args.master_redis_host}:{args.master_redis_port}")

    if args.mode != 'master':
        print(f"   Browser Instances: {args.instances}")

    local_ip = get_local_ip()
    print(f"   Local IP: {local_ip}")

    print()
    print("🔍 Role Description:")
    if args.mode == 'master':
        print("   This machine will be the MASTER node")
        print("   - Runs Redis for service discovery")
        print("   - Runs master API server")
        print("   - Coordinates all worker nodes")
        print("   - Does NOT run browser instances")
        print()
        print(f"   Workers should connect with: --master-redis-host {local_ip}")
        print(f"   Access API at: http://{local_ip}:{args.master_port}")
    elif args.mode == 'worker':
        print("   This machine will be a WORKER node")
        print(f"   - Runs {args.instances} browser instances")
        print("   - Registers with master via Redis")
        print("   - Sends heartbeat every 30 seconds")
        print(f"   - Connects to master at: {args.master_redis_host}")
    else:  # both
        print("   This machine will be MASTER + WORKER")
        print("   - Runs Redis for service discovery")
        print("   - Runs master API server")
        print(f"   - Runs {args.instances} browser instances locally")
        print("   - Can coordinate additional worker machines")
        print()
        print(f"   Additional workers can connect with: --master-redis-host {local_ip}")
        print(f"   Access API at: http://{local_ip}:{args.master_port}")

    print("=" * 70)
    print()

    # Create and configure launcher
    launcher = MultiNodeLauncher(
        num_instances=args.instances,
        mode=args.mode,
        master_redis_host=args.master_redis_host,
        master_redis_port=args.master_redis_port,
        advertise_ip=args.advertise_ip
    )

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
