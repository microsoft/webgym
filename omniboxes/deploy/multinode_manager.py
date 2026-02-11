"""
Multi-Node OmniBoxes Process Manager

Extends the single-machine launcher to support multi-machine deployments
with automatic node discovery via Redis registry.

Supports three deployment modes:
- 'master': Master server only (coordinates workers, no local browsers)
- 'worker': Worker node only (browsers + node server, registers with master)
- 'both': Master + Worker on same machine (default for single-machine)
"""

import subprocess
import threading
import time
import sys
from typing import Optional

from omniboxes.deploy.process_manager import OmniboxesLauncher, ProcessInfo
from omniboxes.common.redis_registry import RedisRegistry


class MultiNodeLauncher(OmniboxesLauncher):
    """Launcher for multi-node OmniBoxes deployments"""

    def __init__(self,
                 num_instances: int,
                 mode: str = 'both',
                 master_redis_host: str = 'localhost',
                 master_redis_port: int = 6379,
                 advertise_ip: Optional[str] = None):
        """
        Initialize multi-node launcher

        Args:
            num_instances: Number of browser instances
            mode: Deployment mode ('master', 'worker', 'both')
            master_redis_host: Redis host (master's IP for worker nodes)
            master_redis_port: Redis port on master
            advertise_ip: Public IP to register with master (overrides auto-detect)
        """
        super().__init__(num_instances)

        if mode not in ['master', 'worker', 'both']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'master', 'worker', or 'both'")

        self.mode = mode
        self.master_redis_host = master_redis_host
        self.master_redis_port = master_redis_port
        self.advertise_ip = advertise_ip
        # Instance/node servers use LOCAL Redis (DB 0) for instance pool management
        # Registry uses MASTER's Redis (DB 1) for node discovery
        # self.redis_host/redis_port stay as localhost:6379 (set by parent __init__)
        self.registry: Optional[RedisRegistry] = None
        self.heartbeat_thread: Optional[threading.Thread] = None

        print(f"🔧 Multi-node mode: {mode}")
        print(f"🔧 Master Redis: {master_redis_host}:{master_redis_port}")
        if advertise_ip:
            print(f"🔧 Advertise IP: {advertise_ip}")

    def start_redis(self) -> bool:
        """Start Redis server (all modes need local Redis for instance pool)"""
        return super().start_redis()

    def setup_registry(self) -> bool:
        """Setup Redis registry for node discovery"""
        print(f"🔍 Setting up Redis registry...")

        try:
            self.registry = RedisRegistry(
                redis_host=self.master_redis_host,
                redis_port=self.master_redis_port,
                registry_db=1,  # Use DB 1 for registry
                ttl=120,
                logger=self.logger if hasattr(self, 'logger') else None
            )
            print(f"✅ Redis registry connected")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Redis registry: {e}")
            return False

    def register_worker_node(self) -> bool:
        """Register this worker node in the registry"""
        if not self.registry:
            print("❌ Registry not initialized")
            return False

        ip = self.advertise_ip if self.advertise_ip else self.registry.get_local_ip()
        node_url = f"http://{ip}:{self.node_port}"

        print(f"📝 Registering worker node: {node_url}")

        if not self.registry.register_node(node_url):
            return False

        # Start heartbeat thread to keep registration alive
        self.heartbeat_thread = threading.Thread(
            target=self.registry.start_heartbeat,
            args=(node_url, 30),  # Refresh every 30s (TTL is 120s)
            daemon=True
        )
        self.heartbeat_thread.start()
        print(f"✅ Worker node registered with heartbeat")

        return True

    def start_master_server(self):
        """Start the master server with Redis-based node discovery"""
        if self.mode == 'worker':
            print("⏭️  Skipping master server (worker mode)")
            return True

        print(f"👑 Starting master server on port {self.master_port} with Redis discovery...")

        cmd = [
            "python", "-m", "omniboxes.master.server",
            "--port", str(self.master_port),
            "--workers", str(self.master_workers),
            "--redis-host", self.master_redis_host,
            "--redis-port", str(self.master_redis_port),
            "--redis-registry"  # Enable Redis-based discovery
        ]

        # Add local node if running in 'both' mode
        if self.mode == 'both':
            cmd.extend(["--nodes", f"http://localhost:{self.node_port}"])

        process = self.start_process(cmd, f"Master:{self.master_port}")

        if process:
            print(f"✅ Master server started with Redis-based node discovery")
            return True
        else:
            print("❌ Failed to start master server")
            return False

    def start_node_server(self):
        """Start the node server (worker and both modes)"""
        if self.mode == 'master':
            print("⏭️  Skipping node server (master-only mode)")
            return True

        return super().start_node_server()

    def start_instance_servers_parallel(self) -> bool:
        """Start instance servers (worker and both modes)"""
        if self.mode == 'master':
            print("⏭️  Skipping instance servers (master-only mode)")
            return True

        return super().start_instance_servers_parallel()

    def stop_redis_service(self):
        """Stop the systemd Redis service to prevent it from respawning."""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "--quiet", "redis-server"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print("🔴 Stopping systemd Redis service...")
                subprocess.run(
                    ["sudo", "systemctl", "stop", "redis-server"],
                    capture_output=True, text=True, timeout=10
                )
                subprocess.run(
                    ["sudo", "systemctl", "stop", "redis"],
                    capture_output=True, text=True, timeout=10
                )
                print("✅ Systemd Redis service stopped")
        except Exception as e:
            print(f"⚠️  Failed to stop systemd Redis service: {e}")

    def cleanup(self):
        """Cleanup and unregister from registry"""
        if self._cleaned_up:
            return

        # Unregister from registry if worker node
        if self.registry and self.mode in ['worker', 'both']:
            print("📝 Unregistering from registry...")
            try:
                self.registry.unregister_node()
            except Exception as e:
                print(f"⚠️  Failed to unregister from registry: {e}")

        super().cleanup()

        # Stop systemd-managed Redis after parent cleanup
        self.stop_redis_service()

    def run(self):
        """Main execution method for multi-node deployment"""
        print("=" * 60)
        print("🎯 OMNIBOXES MULTI-NODE LAUNCHER")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"   • Mode: {self.mode}")
        print(f"   • Master Redis: {self.master_redis_host}:{self.master_redis_port}")

        if self.mode != 'master':
            print(f"   • Redis server port: {self.redis_port}")
            print(f"   • Instance servers: {self.num_instances}")
            print(f"   • Instance ports: {self.instance_start_port}-{self.instance_start_port + self.num_instances - 1}")
            print(f"   • Node server port: {self.node_port}")

        if self.mode != 'worker':
            print(f"   • Master server port: {self.master_port}")

        print(f"   • Parallel startup workers: {self.max_parallel_starts}")
        print(f"   • Process recovery: {'Enabled' if self.recovery_enabled else 'Disabled'}")
        print("=" * 60)

        # Validate port conflicts (skip for master-only mode)
        if self.mode != 'master':
            instance_port_range = range(self.instance_start_port, self.instance_start_port + self.num_instances)
            reserved_ports = [self.redis_port, self.node_port, self.master_port]

            for port in reserved_ports:
                if port in instance_port_range:
                    print(f"❌ ERROR: Port conflict detected!")
                    print(f"   Instance servers will use ports {self.instance_start_port}-{self.instance_start_port + self.num_instances - 1}")
                    print(f"   This conflicts with reserved port {port}")
                    print(f"\n💡 Solution: Use a different instance start port or reduce number of instances")
                    sys.exit(1)

        self.kill_existing_processes()
        self.setup_signal_handlers()

        try:
            # Step 1: Start Redis (master and both modes only)
            if not self.start_redis():
                raise Exception("Failed to start Redis server")

            # Step 2: Setup registry connection
            if not self.setup_registry():
                raise Exception("Failed to setup Redis registry")

            # Step 3: Start instance servers (worker and both modes)
            if self.mode != 'master':
                if not self.start_instance_servers_parallel():
                    raise Exception("Failed to start instance servers")

                self.wait_for_startup()

                # Step 4: Start node server (worker and both modes)
                if not self.start_node_server():
                    raise Exception("Failed to start node server")

                time.sleep(1)

                # Step 5: Register worker node in registry (worker mode only)
                # In 'both' mode, master already adds localhost as a static node
                if self.mode == 'worker':
                    if not self.register_worker_node():
                        raise Exception("Failed to register worker node")

                time.sleep(1)

            # Step 6: Start master server (master and both modes)
            if not self.start_master_server():
                raise Exception("Failed to start master server")

            print("=" * 60)
            print("🎉 All servers started successfully!")
            print("📋 Server Status:")

            if self.mode != 'worker':
                print(f"   • Master server running on http://0.0.0.0:{self.master_port}")
                print(f"     - Discovering workers from Redis registry")

            if self.mode != 'master':
                print(f"   • Redis server running on port {self.redis_port}")
                print(f"   • {self.num_instances} instance server(s) running")
                print(f"   • Node server running on http://0.0.0.0:{self.node_port}")
                print(f"     - Registered in Redis registry with heartbeat")

            print("=" * 60)
            print("💡 Press Ctrl+C to stop all servers")
            print("🔄 Automatic process recovery is enabled")
            print("=" * 60)

            while not self.shutdown_event.is_set():
                time.sleep(5)
                self.check_and_restart_processes()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
