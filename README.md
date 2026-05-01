# monkeylater

Run selected functions later, in background threads.

`monkeylater` is an experimental Python library for making explicit function calls asynchronous by monkey-patching them. It is aimed at experiment scripts and logging/output paths where writes such as `numpy.save` or `cv2.imwrite` should not block the main loop.

![teaser](docs/monkeylater-illustration.png)

> [!CAUTION]
> Patched functions no longer run synchronously. They return `Future` objects, exceptions are delayed until `flush()` or `Future.result()`, and mutable inputs may be read after your code has modified them. Use this only for experiments, logging, and non-critical output paths.

## Global patching

Use global patching for scripts and experiment code where you want selected functions to run in the background across a wider section of your program.

```python
import numpy as np
import monkeylater

monkeylater.patch_global((np, "save"))

larger_array = np.random.rand(5000, 5000)
np.save("large_array.npy", larger_array)

# ...
# The save runs in the background, and the main thread can continue doing other work.
```

Patch multiple functions by passing more explicit `(owner, "attribute")` tuples:

```python
monkeylater.patch_global((np, "save"), (cv2, "imwrite"))
```

## Advanced usage

### Scoped patching

Use scoped patching when you want a safer, temporary patch around a specific block. The original function is restored when the `with` block exits, and pending background work is flushed before exit completes.

```python
import numpy as np
import monkeylater

arr = np.arange(9)

with monkeylater.patch((np, "save")):
    np.save("array.npy", arr)
```

### Flushing and restoring

Global mode automatically flushes pending work at process exit. You only need `flush()` or `restore_all()` when you want control earlier.

Call `flush()` when you need outputs to exist now, want exceptions now, or are writing tests:

```python
monkeylater.flush()
```

Call `restore_all()` in long-running processes when you want to stop global patching before process exit:

```python
monkeylater.restore_all()
```

### Future usage

Patched functions return `concurrent.futures.Future` objects instead of their original return values. Keep the future when you want to wait for one call or inspect its result directly.

```python
with monkeylater.patch((np, "save")):
    save_job = np.save("array.npy", np.arange(9))

save_job.result()
```

Exceptions raised by the original function are deferred until `Future.result()`, `flush()`, or scoped patch exit.

### Mutable inputs

`monkeylater` does not copy arguments. If you pass a mutable object and modify it immediately after calling the patched function, the background function may see the modified object.

Copy explicitly when needed:

```python
with monkeylater.patch((np, "save")):
    np.save("array.npy", arr.copy())
```
