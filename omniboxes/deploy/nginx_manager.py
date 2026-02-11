"""
Nginx Manager Module

Handles nginx reverse proxy setup for production deployments.
"""

import subprocess
import socket


class NginxManager:
    """Manages nginx reverse proxy setup"""

    def __init__(self, master_port: int):
        self.master_port = master_port
        self.nginx_conf_path = "/etc/nginx/sites-available/omnibox-master"
        self.nginx_enabled_path = "/etc/nginx/sites-enabled/omnibox-master"

    def get_external_ip(self) -> str:
        """Get the external IP address of the machine"""
        try:
            # Get hostname's IP
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                if ips:
                    return ips[0]
        except Exception:
            pass

        # Fallback to socket method
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def is_nginx_installed(self) -> bool:
        """Check if nginx is installed"""
        try:
            result = subprocess.run(
                ["nginx", "-v"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def create_nginx_config(self) -> str:
        """Generate nginx configuration content"""
        return f"""server {{
    listen 80;
    server_name _;

    # Increase timeouts for long-running requests
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
    send_timeout 600s;

    # Increase buffer sizes for large responses
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;

    location / {{
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto http;
        proxy_pass http://127.0.0.1:{self.master_port};
    }}
}}
"""

    def setup(self) -> bool:
        """Setup nginx reverse proxy"""
        print("🌐 Setting up nginx reverse proxy...")
        print(f"   Master port: {self.master_port}")

        external_ip = self.get_external_ip()
        print(f"   External IP: {external_ip}")

        # Check if nginx is installed
        if not self.is_nginx_installed():
            print("\n❌ ERROR: nginx is not installed!")
            print("Please install nginx first:")
            print("  sudo apt-get update && sudo apt-get install -y nginx")
            return False

        print("   nginx is installed ✓")

        try:
            # Create nginx configuration
            print(f"   Creating nginx configuration at {self.nginx_conf_path}...")
            nginx_config = self.create_nginx_config()

            # Write config using sudo tee
            proc = subprocess.Popen(
                ["sudo", "tee", self.nginx_conf_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            _, stderr = proc.communicate(input=nginx_config)

            if proc.returncode != 0:
                print(f"   ❌ Failed to write nginx config: {stderr}")
                return False

            # Enable the site
            print("   Enabling nginx site...")
            subprocess.run(
                ["sudo", "ln", "-sf", self.nginx_conf_path, self.nginx_enabled_path],
                check=True
            )

            # Remove default site
            subprocess.run(
                ["sudo", "rm", "-f", "/etc/nginx/sites-enabled/default"],
                check=False
            )

            # Test nginx configuration
            print("   Testing nginx configuration...")
            result = subprocess.run(
                ["sudo", "nginx", "-t"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"   ❌ nginx configuration test failed:")
                print(result.stderr)
                return False

            # Restart nginx
            print("   Restarting nginx...")
            subprocess.run(["sudo", "systemctl", "restart", "nginx"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "nginx"], check=True)

            print("\n" + "=" * 60)
            print("✅ nginx setup completed successfully!")
            print("=" * 60)
            print(f"\nYour OmniBox Master server will be accessible at:")
            print(f"  http://{external_ip}")
            print(f"\nLocally accessible at:")
            print(f"  http://localhost:{self.master_port}")
            print(f"\nTest the connection with:")
            print(f'  curl -H "x-api-key: default_key" http://{external_ip}/info')
            print("=" * 60 + "\n")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during nginx setup: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during nginx setup: {e}")
            return False
