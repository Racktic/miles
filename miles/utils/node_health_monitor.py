"""Per-node health monitoring for diagnosing node failures in distributed training.

Deploys one lightweight Ray actor per cluster node that collects disk, GPU, CPU,
process, and kernel-log metrics and writes them to per-node log files on shared
filesystem (Weka).  Designed so that when a node dies, its last metrics are
already flushed to shared storage for post-mortem analysis.
"""

import concurrent.futures
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)

DEFAULT_ALERT_THRESHOLDS = {
    "tmp_disk_percent": 80.0,
    "disk_percent": 90.0,
    "gpu_mem_percent": 95.0,
    "cpu_mem_percent": 95.0,
}


@ray.remote(num_cpus=0, max_restarts=3)
class NodeHealthProbeActor:
    """Per-node health monitoring actor that writes to shared filesystem."""

    def __init__(
        self,
        node_id: str,
        node_ip: str,
        log_dir: str,
        interval: float = 30.0,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.node_id = node_id
        self.node_ip = node_ip
        self.hostname = os.uname().nodename
        self.log_dir = log_dir
        self.interval = interval
        self.alert_thresholds = alert_thresholds or dict(DEFAULT_ALERT_THRESHOLDS)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sample_count = 0
        self._latest_alerts: List[str] = []
        self._log_file_path = self._init_log_file()
        self._nvml_initialized = False

    def _init_log_file(self) -> str:
        safe_hostname = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.hostname)
        safe_ip = self.node_ip.replace(":", "_")
        log_path = Path(self.log_dir) / "node_health" / f"{safe_hostname}_{safe_ip}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    def _try_init_nvml(self) -> bool:
        if self._nvml_initialized:
            return True
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception:
            return False

    def start(self) -> str:
        """Start internal sampling loop. Returns log file path."""
        if self._thread is not None and self._thread.is_alive():
            return self._log_file_path
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"health-probe-{self.hostname}",
        )
        self._thread.start()
        return self._log_file_path

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
            self._thread = None

    def is_alive(self) -> bool:
        """Heartbeat check used by the supervisor."""
        return True

    def get_latest_alerts(self) -> List[str]:
        return list(self._latest_alerts)

    # ------------------------------------------------------------------
    # Internal sampling loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        self._write_line(
            f"=== Node Health Monitor Started on {self.hostname} ({self.node_ip}) ==="
        )
        self._write_line(
            f"=== Interval: {self.interval}s | Thresholds: {self.alert_thresholds} ==="
        )

        while not self._stop_event.is_set():
            try:
                self._collect_and_log()
            except Exception as exc:
                self._write_line(f"  ERROR: sampling failed: {exc}")
            if self._stop_event.wait(timeout=self.interval):
                break

        self._write_line("=== Node Health Monitor Stopped ===")

    def _collect_and_log(self) -> None:
        self._sample_count += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        alerts: List[str] = []
        lines = [
            f"[{ts}] [node={self.hostname}] [ip={self.node_ip}] [sample={self._sample_count}]"
        ]

        self._collect_disk(lines, alerts)
        self._collect_gpu(lines, alerts)
        self._collect_cpu_mem(lines, alerts)
        self._collect_procs(lines)
        self._collect_dmesg(lines, alerts)

        if alerts:
            lines.append(f"  *** ALERTS ({len(alerts)}): {'; '.join(alerts)} ***")

        self._latest_alerts = alerts

        for line in lines:
            self._write_line(line)
        self._write_line("")  # blank separator

    # ------------------------------------------------------------------
    # Metric collectors
    # ------------------------------------------------------------------

    def _collect_disk(self, lines: List[str], alerts: List[str]) -> None:
        try:
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                except (PermissionError, OSError):
                    continue
                pct = usage.percent
                line = (
                    f"  DISK  {part.mountpoint:<20s}: {pct:5.1f}% used "
                    f"({_fmt_bytes(usage.used)}/{_fmt_bytes(usage.total)}) "
                    f"free={_fmt_bytes(usage.free)}"
                )
                is_tmp = part.mountpoint in ("/tmp", "/var/tmp")
                threshold_key = "tmp_disk_percent" if is_tmp else "disk_percent"
                threshold = self.alert_thresholds.get(threshold_key, 90.0)
                if pct > threshold:
                    alert = f"{part.mountpoint} at {pct:.1f}% (threshold {threshold:.0f}%)"
                    alerts.append(alert)
                    line += f"  *** ALERT: {alert} ***"
                lines.append(line)
        except Exception as exc:
            lines.append(f"  DISK  ERROR: {exc}")

    def _collect_gpu(self, lines: List[str], alerts: List[str]) -> None:
        if not self._try_init_nvml():
            lines.append("  GPU   UNAVAILABLE (pynvml not found)")
            return
        try:
            import pynvml

            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_procs: List[str] = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = -1

                mem_pct = (mem.used / mem.total * 100) if mem.total > 0 else 0
                line = (
                    f"  GPU {i}: mem={_fmt_bytes(mem.used)}/{_fmt_bytes(mem.total)} "
                    f"({mem_pct:.1f}%) util={util.gpu}% temp={temp}C"
                )
                if mem_pct > self.alert_thresholds.get("gpu_mem_percent", 95.0):
                    alert = f"GPU {i} mem at {mem_pct:.1f}%"
                    alerts.append(alert)
                    line += f"  *** ALERT: {alert} ***"
                lines.append(line)

                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in procs:
                        gpu_procs.append(
                            f"pid={proc.pid} gpu{i}={_fmt_bytes(proc.usedGpuMemory)}"
                        )
                except Exception:
                    pass

            if gpu_procs:
                lines.append(f"  GPU_PROCS: {', '.join(gpu_procs[:20])}")
        except Exception as exc:
            lines.append(f"  GPU   ERROR: {exc}")

    def _collect_cpu_mem(self, lines: List[str], alerts: List[str]) -> None:
        try:
            vm = psutil.virtual_memory()
            line = (
                f"  CPU_MEM: {vm.percent:.1f}% used "
                f"({_fmt_bytes(vm.used)}/{_fmt_bytes(vm.total)}) "
                f"avail={_fmt_bytes(vm.available)}"
            )
            if hasattr(vm, "cached"):
                line += f" cached={_fmt_bytes(vm.cached)}"
            if hasattr(vm, "buffers"):
                line += f" buffers={_fmt_bytes(vm.buffers)}"
            if vm.percent > self.alert_thresholds.get("cpu_mem_percent", 95.0):
                alert = f"CPU mem at {vm.percent:.1f}%"
                alerts.append(alert)
                line += f"  *** ALERT: {alert} ***"
            lines.append(line)
        except Exception as exc:
            lines.append(f"  CPU_MEM ERROR: {exc}")

    def _collect_procs(self, lines: List[str]) -> None:
        try:
            python_count = 0
            ray_worker_count = 0
            for p in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = p.info.get("name", "") or ""
                    if "python" in name.lower():
                        python_count += 1
                    cmdline = p.info.get("cmdline", []) or []
                    if any("ray" in str(c) for c in cmdline):
                        ray_worker_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            enroot_count = "N/A"
            try:
                result = subprocess.run(
                    ["enroot", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    container_lines = [
                        l for l in result.stdout.strip().split("\n") if l.strip()
                    ]
                    enroot_count = str(len(container_lines))
            except (FileNotFoundError, subprocess.SubprocessError):
                pass

            lines.append(
                f"  PROCS: python={python_count} ray_workers={ray_worker_count} "
                f"enroot_containers={enroot_count}"
            )
        except Exception as exc:
            lines.append(f"  PROCS ERROR: {exc}")

    def _collect_dmesg(self, lines: List[str], alerts: List[str]) -> None:
        try:
            result = subprocess.run(
                ["dmesg", "-T", "--level=err,crit,alert,emerg"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                oom_lines = [
                    l.strip()
                    for l in result.stdout.split("\n")
                    if "oom" in l.lower()
                    or "killed process" in l.lower()
                    or "out of memory" in l.lower()
                ]
                if oom_lines:
                    for oom_line in oom_lines[-3:]:
                        lines.append(f"  DMESG_OOM: {oom_line}")
                    alerts.append(
                        f"{len(oom_lines)} OOM killer messages found in dmesg"
                    )
                else:
                    lines.append("  DMESG: (no OOM messages)")
            else:
                lines.append("  DMESG: (access denied or unavailable)")
        except Exception:
            lines.append("  DMESG: (unavailable)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_line(self, line: str) -> None:
        try:
            with open(self._log_file_path, "a") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass  # Never crash the monitor due to write failure


# ======================================================================
# Cluster-level orchestrator
# ======================================================================


@dataclass
class ClusterHealthMonitor:
    """Orchestrator that maintains one NodeHealthProbeActor per cluster node."""

    log_dir: str
    interval: float = 30.0
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ALERT_THRESHOLDS)
    )
    supervisor_interval: float = 60.0

    _actors: Dict[str, Any] = field(default_factory=dict, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)

    def start(self) -> None:
        logger.info(
            "Starting ClusterHealthMonitor (log_dir=%s, interval=%ss)",
            self.log_dir,
            self.interval,
        )
        os.makedirs(os.path.join(self.log_dir, "node_health"), exist_ok=True)

        self._sync_actors()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._supervisor_loop,
            daemon=True,
            name="cluster-health-supervisor",
        )
        self._thread.start()
        logger.info(
            "ClusterHealthMonitor started with %d probe actors", len(self._actors)
        )

    def stop(self) -> None:
        logger.info("Stopping ClusterHealthMonitor...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.supervisor_interval + 5)
            self._thread = None

        for actor in self._actors.values():
            try:
                ray.get(actor.stop.remote(), timeout=5)
            except Exception:
                pass
        self._actors.clear()
        logger.info("ClusterHealthMonitor stopped")

    # ------------------------------------------------------------------
    # Supervisor loop
    # ------------------------------------------------------------------

    def _supervisor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sync_actors()
                self._check_actor_liveness()
            except Exception:
                logger.warning(
                    "ClusterHealthMonitor supervisor error", exc_info=True
                )
            if self._stop_event.wait(timeout=self.supervisor_interval):
                break

    def _sync_actors(self) -> None:
        """Discover nodes and ensure one probe actor per node."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw_nodes = pool.submit(ray.nodes).result(timeout=10)
        except Exception:
            logger.warning(
                "ClusterHealthMonitor: ray.nodes() failed, skipping sync"
            )
            return

        alive: Dict[str, str] = {}
        for node in raw_nodes:
            if not node.get("Alive"):
                continue
            node_id = str(node.get("NodeID", ""))
            node_ip = str(
                node.get("NodeManagerAddress") or node.get("NodeName") or ""
            )
            if node_id and node_ip:
                alive[node_id] = node_ip

        # Remove actors for dead nodes
        for node_id in list(self._actors.keys()):
            if node_id not in alive:
                logger.info(
                    "Node %s no longer alive, removing probe actor", node_id[:8]
                )
                self._actors.pop(node_id, None)

        # Create actors for new nodes
        for node_id, node_ip in alive.items():
            if node_id not in self._actors:
                self._create_actor(node_id, node_ip)

    def _create_actor(self, node_id: str, node_ip: str) -> None:
        try:
            scheduling = NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=True
            )
            actor = NodeHealthProbeActor.options(
                scheduling_strategy=scheduling,
                name=f"node-health-{node_id[:8]}",
                get_if_exists=True,
            ).remote(
                node_id=node_id,
                node_ip=node_ip,
                log_dir=self.log_dir,
                interval=self.interval,
                alert_thresholds=self.alert_thresholds,
            )
            log_path = ray.get(actor.start.remote(), timeout=10)
            self._actors[node_id] = actor
            logger.info(
                "Created health probe for node %s (%s), logging to %s",
                node_id[:8],
                node_ip,
                log_path,
            )
        except Exception:
            logger.warning(
                "Failed to create health probe for node %s",
                node_id[:8],
                exc_info=True,
            )

    def _check_actor_liveness(self) -> None:
        """Ping each actor and recreate dead ones."""
        dead: List[str] = []
        refs = {
            node_id: actor.is_alive.remote()
            for node_id, actor in self._actors.items()
        }
        for node_id, ref in refs.items():
            try:
                ray.get(ref, timeout=5)
            except Exception:
                dead.append(node_id)

        for node_id in dead:
            logger.warning(
                "Health probe actor for node %s is dead, will recreate on next sync",
                node_id[:8],
            )
            self._actors.pop(node_id, None)


# ======================================================================
# Utility
# ======================================================================


def _fmt_bytes(n: int) -> str:
    if n >= 1024**4:
        return f"{n / 1024**4:.1f}TB"
    elif n >= 1024**3:
        return f"{n / 1024**3:.1f}GB"
    elif n >= 1024**2:
        return f"{n / 1024**2:.1f}MB"
    else:
        return f"{n / 1024:.1f}KB"
