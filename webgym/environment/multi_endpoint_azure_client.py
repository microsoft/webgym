#!/usr/bin/env python3
"""
Multi-endpoint Azure OpenAI client for WebGym reward evaluation.
Simple synchronous implementation with endpoint failover.

Loads endpoint configurations from JSON config files (same format as agento/endpoint_configs/).
"""

import time
import random
import os
import json
import glob
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openai import AzureOpenAI


@dataclass
class EndpointConfig:
    azure_endpoint: str
    deployment: str
    api_version: str = "2025-01-01-preview"
    token_provider_type: str = "default"  # "default" (AzureCliCredential) or "uami" (ManagedIdentityCredential)
    model: str = "gpt-4o"
    last_failure_time: float = 0
    consecutive_failures: int = 0


class MultiEndpointAzureOpenAI:
    def __init__(self, config_dir: str, max_concurrent: int = 50):
        """
        Initialize multi-endpoint Azure OpenAI client from a directory of JSON config files.

        Args:
            config_dir: Path to directory containing endpoint JSON config files
                        (e.g. /path/to/endpoint_configs/gpt4o/)
            max_concurrent: Max concurrent requests (unused, kept for interface compat)
        """
        self.config_dir = config_dir
        self.endpoints = self._load_endpoint_configs(config_dir)
        self.clients: Dict[str, AzureOpenAI] = {}

        # Token caching for "default" (AzureCliCredential) mode
        self._token_cache_time = 0
        self._token_max_age = 300  # Re-read from disk every 5 minutes
        self._cached_token = None

        # Initialize clients immediately
        self._initialize_clients()

    def _load_endpoint_configs(self, config_dir: str) -> List[EndpointConfig]:
        """Load endpoint configurations from JSON files in a directory."""
        config_dir = os.path.abspath(config_dir)
        if not os.path.isdir(config_dir):
            raise ValueError(f"Endpoint config directory does not exist: {config_dir}")

        config_files = sorted(glob.glob(os.path.join(config_dir, "*.json")))
        if not config_files:
            raise ValueError(f"No JSON config files found in: {config_dir}")

        endpoints = []
        for filepath in config_files:
            with open(filepath) as f:
                raw = json.load(f)

            kwargs = raw.get("CHAT_COMPLETION_KWARGS_JSON", {})
            endpoints.append(EndpointConfig(
                azure_endpoint=kwargs["azure_endpoint"],
                deployment=kwargs["azure_deployment"],
                api_version=kwargs.get("api_version", "2025-01-01-preview"),
                token_provider_type=kwargs.get("azure_ad_token_provider", "default"),
                model=kwargs.get("model", "gpt-4o"),
            ))

        print(f"✅ Loaded {len(endpoints)} endpoint configs from {config_dir}")
        return endpoints

    def _read_token_from_cache_file(self) -> Optional[str]:
        """
        Read Azure token directly from ~/.azure/msal_token_cache.json.
        Avoids calling 'az' CLI which fails in multi-process environments.
        User must run 'az login' beforehand to populate the cache.
        """
        cache_paths = [
            "/root/.azure/msal_token_cache.json",
            os.path.expanduser("~/.azure/msal_token_cache.json"),
        ]

        cache_path = None
        for path in cache_paths:
            if os.path.exists(path):
                cache_path = path
                break

        if not cache_path:
            print(f"❌ Token cache not found. Tried: {cache_paths}")
            print(f"💡 Please run: az login")
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            access_tokens = cache_data.get("AccessToken", {})
            best_token = None
            best_expiry = 0

            for key, token_data in access_tokens.items():
                if "cognitiveservices.azure.com" in key.lower():
                    expires_on = int(token_data.get("expires_on", 0))
                    if expires_on > (time.time() + 300):  # 5 min buffer
                        if expires_on > best_expiry:
                            best_token = token_data.get("secret")
                            best_expiry = expires_on

            if best_token:
                time_remaining = (best_expiry - time.time()) / 3600
                print(f"✅ Using Azure token from {cache_path} (expires in {time_remaining:.1f}h)")
                return best_token
            else:
                print(f"❌ No valid cognitive services token found in {cache_path}")
                print(f"💡 Please run: az login")
                return None

        except Exception as e:
            print(f"❌ Failed to read token from cache: {str(e)[:200]}")
            print(f"💡 Please run: az login")
            return None

    def _make_disk_token_provider(self):
        """
        Token provider that reads directly from disk cache.
        Safe for multi-process environments (no 'az' CLI subprocess calls).
        """
        def token_provider():
            current_time = time.time()
            needs_refresh = (
                self._cached_token is None or
                (current_time - self._token_cache_time) > self._token_max_age
            )
            if needs_refresh:
                token = self._read_token_from_cache_file()
                if token:
                    self._cached_token = token
                    self._token_cache_time = current_time
                else:
                    self._cached_token = None
                    self._token_cache_time = 0
            return self._cached_token

        return token_provider

    def _resolve_token_provider(self, provider_type: str):
        """Resolve token provider string to a callable.

        "default" -> read token directly from ~/.azure/msal_token_cache.json
                     (safe for multi-process environments, no 'az' CLI calls)
        "uami"    -> ManagedIdentityCredential (for prod Azure compute)
        """
        if provider_type == "uami":
            from azure.identity import get_bearer_token_provider, ManagedIdentityCredential
            return get_bearer_token_provider(
                ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"]),
                "https://cognitiveservices.azure.com/.default",
            )
        else:
            # "default" — read token directly from disk cache
            return self._make_disk_token_provider()

    def _initialize_clients(self, force_refresh=False):
        """Create AzureOpenAI clients, one per unique (azure_endpoint, token_provider_type) pair."""
        if force_refresh:
            print("🔄 Force refreshing Azure clients...")

        if not force_refresh and len(self.clients) > 0:
            return

        self.clients.clear()

        # Group endpoints by (azure_endpoint, token_provider_type) to avoid duplicate clients
        seen = {}
        for endpoint in self.endpoints:
            key = (endpoint.azure_endpoint, endpoint.token_provider_type)
            if key not in seen:
                token_provider = self._resolve_token_provider(endpoint.token_provider_type)
                client = AzureOpenAI(
                    azure_endpoint=endpoint.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=endpoint.api_version,
                    timeout=90.0,
                    max_retries=0,  # Handle retries manually
                )
                seen[key] = client
            self.clients[endpoint.deployment] = seen[key]

        print(f"✅ Created Azure OpenAI clients for {len(self.endpoints)} deployments ({len(seen)} unique endpoints)")

    def _get_available_endpoints(self) -> List[EndpointConfig]:
        """Get endpoints that haven't failed recently"""
        current_time = time.time()
        available = []

        for endpoint in self.endpoints:
            # Skip endpoints that have failed recently (exponential backoff)
            if endpoint.consecutive_failures > 0:
                backoff_time = min(60 * (2 ** endpoint.consecutive_failures), 300)  # Max 5 minutes
                if current_time - endpoint.last_failure_time < backoff_time:
                    continue

            available.append(endpoint)

        # If no endpoints available, reset all failures and try again
        if not available:
            for endpoint in self.endpoints:
                endpoint.consecutive_failures = 0
                endpoint.last_failure_time = 0
            available = self.endpoints.copy()

        return available

    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Make chat completion with infinite retry and intelligent endpoint rotation"""
        last_exception = None
        tried_endpoints = set()  # Track deployments tried in this call
        retry_round = 0
        token_refreshed_this_call = False

        while True:
            available = self._get_available_endpoints()
            untried = [e for e in available if e.deployment not in tried_endpoints]

            # If we've tried all endpoints, reset and start a new round
            if not untried:
                retry_round += 1
                tried_endpoints.clear()

                if retry_round > 0:
                    backoff_delay = min(5 * retry_round, 30)  # Max 30 seconds between rounds
                    print(f"⏳ Tried all {len(available)} endpoints. Round {retry_round} - waiting {backoff_delay}s before retry...")
                    time.sleep(backoff_delay)

                for endpoint in self.endpoints:
                    endpoint.consecutive_failures = 0
                    endpoint.last_failure_time = 0

                untried = self._get_available_endpoints()

            endpoint = random.choice(untried)
            tried_endpoints.add(endpoint.deployment)

            client = self.clients.get(endpoint.deployment)
            if not client:
                continue

            try:
                call_start = time.time()
                response = client.chat.completions.create(
                    model=endpoint.deployment,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', 800),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.95)
                )
                call_duration = time.time() - call_start

                # Success - reset failure count
                endpoint.consecutive_failures = 0
                endpoint.last_failure_time = 0

                content = response.choices[0].message.content
                usage = getattr(response, 'usage', None)
                prompt_tokens = getattr(usage, 'prompt_tokens', '?') if usage else '?'
                completion_tokens = getattr(usage, 'completion_tokens', '?') if usage else '?'
                finish_reason = getattr(response.choices[0], 'finish_reason', '?')

                if retry_round > 0:
                    print(f"✅ Succeeded on {endpoint.deployment} after {retry_round} retry rounds")

                print(f"🔍 [{endpoint.deployment}] {call_duration:.1f}s | {prompt_tokens}/{completion_tokens} tokens | finish={finish_reason} | resp_len={len(content) if content else 0}")

                return {
                    'choices': [{
                        'message': {
                            'content': content
                        }
                    }]
                }

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check for authentication/token expiration errors
                is_auth_error = any(indicator in error_str for indicator in [
                    'aadsts70043',
                    'aadsts50058',
                    'aadsts700082',
                    'refresh token has expired',
                    'token has expired',
                    'authentication failed',
                    'unauthorized',
                    '401'
                ])

                if is_auth_error and not token_refreshed_this_call:
                    print(f"🔑 Authentication error on {endpoint.deployment}: {str(e)[:200]}")
                    print(f"🔄 Refreshing Azure clients...")

                    self._initialize_clients(force_refresh=True)
                    token_refreshed_this_call = True
                    tried_endpoints.clear()

                    print(f"✅ Clients refreshed, retrying...")
                    print(f"💡 If errors persist, run: az login")
                    continue

                # Content filter violations are about the request content, not the endpoint.
                # Retrying on another endpoint will give the same result. Raise immediately.
                if 'content_filter' in error_str or 'responsibleaipolicyviolation' in error_str:
                    print(f"⚠️  Content filter on {endpoint.deployment}: {str(e)[:150]}")
                    raise

                # Invalid image data errors are about the request payload, not the endpoint.
                # Every endpoint will reject the same bad image. Raise immediately.
                if any(s in error_str for s in ['invalid image', 'unsupported image', 'invalid base64']):
                    print(f"⚠️  Invalid image on {endpoint.deployment}: {str(e)[:150]}")
                    raise

                # Record failure
                endpoint.consecutive_failures += 1
                endpoint.last_failure_time = time.time()

                if 'rate limit' in error_str or '429' in error_str:
                    print(f"⚠️  Rate limited on {endpoint.deployment} (tried {len(tried_endpoints)}/{len(available)} endpoints)")
                    continue
                elif any(code in error_str for code in ['500', '502', '503', '504']):
                    print(f"⚠️  Server error on {endpoint.deployment}: {str(e)[:100]}")
                    continue
                else:
                    print(f"⚠️  Error on {endpoint.deployment}: {str(e)[:100]}")
                    continue
