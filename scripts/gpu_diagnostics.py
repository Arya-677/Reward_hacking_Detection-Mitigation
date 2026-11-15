"""GPU Diagnostic CLI Utility for V3.

Displays GPU specifications, current VRAM allocation, temperature,
and the processor placement (100% GPU / Partial / CPU) of models running in Ollama.
"""

from reward_hacking.gpu_monitor import GPUMonitor


def main():
    monitor = GPUMonitor()
    hw = monitor.get_hardware_profile()

    print("================================================================================")
    print("NVIDIA GPU & HARDWARE DIAGNOSTICS")
    print("================================================================================")
    print(f"GPU Model:          {hw.gpu_name}")
    print(f"GPU UUID:           {hw.gpu_uuid}")
    print(f"Driver Version:     {hw.driver_version}")
    print(f"CUDA Compatibility: {hw.cuda_version}")
    print(f"Total VRAM:         {hw.total_vram_mb:.0f} MB ({hw.total_vram_mb / 1024:.1f} GB)")
    print(f"Available VRAM:     {hw.free_vram_mb:.0f} MB ({hw.free_vram_mb / 1024:.1f} GB)")
    print(f"System RAM:         {hw.system_ram_gb:.1f} GB Total ({hw.free_ram_gb:.1f} GB Free)")
    print(f"CPU:                {hw.cpu_name}")
    print(f"Operating System:   {hw.os_version}")
    print(f"Ollama Version:     {hw.ollama_version}")
    print("--------------------------------------------------------------------------------")

    # Current snapshot
    snap = monitor.get_gpu_snapshot()
    if snap:
        print(f"Current VRAM Usage: {snap.memory_used_mb:.0f} MB / {snap.memory_total_mb:.0f} MB ({snap.memory_used_mb / snap.memory_total_mb * 100:.1f}%)")
        print(f"GPU Utilization:    {snap.utilization_gpu_pct:.1f}%")
        print(f"Temperature:        {snap.temperature_c:.0f} C")
    else:
        print("Warning: Could not read instantaneous GPU metrics.")

    print("\n--------------------------------------------------------------------------------")
    print("OLLAMA ACTIVE MODELS & PROCESSOR PLACEMENT")
    print("--------------------------------------------------------------------------------")
    loaded = monitor.get_ollama_loaded_models()
    if not loaded:
        print("No models currently loaded in Ollama memory (idle).")
    else:
        for m in loaded:
            status = " [OK - 100% GPU ACCELERATED]" if m.is_fully_gpu else " [WARNING - CPU FALLBACK DETECTED]"
            print(f"- Model:     {m.name}")
            print(f"  Placement: {m.processor_string}{status}")
            print(f"  VRAM:      {m.vram_mb:.1f} MB (Total Size: {m.total_size_mb:.1f} MB)")
            print(f"  ID:        {m.model_id}")

    print("================================================================================\n")


if __name__ == "__main__":
    main()
