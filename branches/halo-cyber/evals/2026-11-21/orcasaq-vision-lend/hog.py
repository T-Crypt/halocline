"""Test helper for run3.sh only: holds all but KEEP_MIB of the card's free memory until killed, so the engine's next
cudaMalloc for the image encoder's arena really fails. One context, one allocation, freed on exit."""
import ctypes, glob, os, signal, sys, time
lib = ctypes.CDLL("/opt/cuda/targets/x86_64-linux/lib/libcudart.so.13")   # the runtime the engine links
keep = int(os.environ.get("KEEP_MIB", "200")) << 20
free, total = ctypes.c_size_t(), ctypes.c_size_t()
assert lib.cudaSetDevice(0) == 0 and lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
want = max(0, free.value - keep)
p = ctypes.c_void_p()
rc = lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(want))
lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
print(f"hog rc={rc} held {want >> 20} MiB, free now {free.value >> 20} MiB", flush=True)
signal.signal(signal.SIGTERM, lambda *a: (lib.cudaFree(p), sys.exit(0)))
time.sleep(int(os.environ.get("HOLD_S", "120")))
lib.cudaFree(p)
