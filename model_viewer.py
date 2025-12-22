import psutil, time, subprocess, json
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class OllamaMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.data = {"t": [], "cpu": [], "cpw": [], "gpu": [], "gpw": [], "ram": [], "oram": []}
        self.proc = None
        
    def get_proc(self):
        for p in psutil.process_iter(['name']):
            try:
                if 'ollama' in p.info['name'].lower(): return p
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        return None
    
    def get_stats(self):
        cpu = psutil.cpu_percent(interval=0.1)
        cpw = cpu * 0.5
        ram = psutil.virtual_memory().percent
        gpu, gpw, oram = 0, 0, 0
        
        if GPU_AVAILABLE:
            try:
                g = GPUtil.getGPUs()[0]
                gpu, gpw = g.load * 100, getattr(g, 'powerDraw', 0) or 0
            except: pass
            
        if not self.proc or not self.proc.is_running(): self.proc = self.get_proc()
        if self.proc:
            try: oram = self.proc.memory_info().rss / (1024 * 1024)
            except: pass
            
        return cpu, cpw, gpu, gpw, ram, oram

    def monitor(self, duration=60):
        start = time.time()
        try:
            while (time.time() - start) < duration:
                now = time.time() - start
                stats = self.get_stats()
                
                self.data["t"].append(now)
                for i, k in enumerate(["cpu", "cpw", "gpu", "gpw", "ram", "oram"]):
                    self.data[k].append(stats[i])
                
                print(f"\r{now:.1f}s | CPU:{stats[0]}% | GPU:{stats[2]}% | Ollama:{stats[5]:.0f}MB", end="")
                time.sleep(self.interval)
        except KeyboardInterrupt: pass
        self.plot()

    def plot(self):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        d = self.data
        ax[0].plot(d["t"], d["cpu"], label="CPU %")
        if GPU_AVAILABLE: ax[0].plot(d["t"], d["gpu"], label="GPU %")
        ax[1].plot(d["t"], d["cpw"], label="CPU W")
        if GPU_AVAILABLE: ax[1].plot(d["t"], d["gpw"], label="GPU W")
        ax[2].plot(d["t"], d["ram"], label="System RAM %")
        ax[2].plot(d["t"], d["oram"], label="Ollama MB")
        
        for a in ax: a.legend(); a.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"ollama_{datetime.now().strftime('%H%M%S')}.png")
        plt.show()

if __name__ == "__main__":
    d = int(input("Duration (s) [60]: ") or 60)
    OllamaMonitor(interval=1.0).monitor(duration=d)
