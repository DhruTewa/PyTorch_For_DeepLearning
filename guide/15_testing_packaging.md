# Chapter 15: Testing and Packaging

| | |
|---|---|
| **Level** | Intermediate |
| **You will learn** | What to test in a computer vision project, how to write fast automated tests, and how to turn your code into a package others can install |
| **Prerequisites** | Chapters 0, 2, 4, 6, 14 |
| **Ready-made code** | Patterns only; adapt to your project |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!TIP]
> **🟢 Extra: this whole chapter.** Real projects need their own automated checks.

**In this chapter**
[15.1 The idea](#151-the-idea) · [15.2 What to test](#152-what-to-test) · [15.3 Model tests](#153-model-tests) · [15.4 Data tests](#154-data-tests) · [15.5 Running tests](#155-running-tests) · [15.6 Packaging](#156-packaging) · [15.7 Automation](#157-automation) · [15.8 Common problems](#158-common-problems-and-how-to-spot-them) · [Key takeaways](#key-takeaways)

---

## 15.1 The idea

Machine learning code fails *quietly*. A bug rarely crashes; it just gives a slightly worse model, and you attribute it to bad luck. Automated tests turn quiet failures into loud ones, in seconds, before a long training run or a deployment.

You are not testing "is the model accurate?" (that is evaluation, Chapter 10). You are testing "is the **machinery** correct?".

---

## 15.2 What to test

| Area | Example check | Catches |
|---|---|---|
| **Shapes** | Output is `[batch, num_classes]` for batch sizes 1 and 4 | Wrong layer sizes, hard-coded batch size |
| **Learning** | The model can overfit one tiny batch | Broken loss, frozen layers, wrong labels |
| **Gradients** | Every parameter gets a non-zero gradient | Detached graph, accidental freezing |
| **Determinism** | In `eval()` mode, same input gives same output | Forgotten `eval()`, random layers |
| **Save and load** | Reloaded model gives identical outputs | Mismatched architecture, missing keys |
| **Preprocessing** | Training and serving use the same transform | The most common production bug |
| **Data** | No duplicates across splits; images open; labels in range | Leakage, corrupt files, label errors |
| **Prediction API** | A known input returns the expected fields and class names | Broken serving code |

Keep tests **fast** (seconds): tiny models, tiny fake data, a handful of images. Slow tests do not get run.

---

## 15.3 Model tests

```python
# tests/test_model.py
import pytest, torch, torch.nn as nn

def build_model(num_classes=3):                     # in a real project: import it from your package
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, num_classes))

@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(8, 3, 32, 32), torch.randint(0, 3, (8,))

def test_output_shape(batch):
    x, _ = batch
    assert build_model(3)(x).shape == (8, 3)

@pytest.mark.parametrize("batch_size", [1, 4])
def test_any_batch_size(batch_size):
    assert build_model()(torch.randn(batch_size, 3, 32, 32)).shape[0] == batch_size

def test_eval_is_deterministic(batch):
    x, _ = batch
    model = build_model().eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))

def test_can_overfit_one_batch(batch):
    x, y = batch
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        loss = nn.functional.cross_entropy(model(x), y)
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * first                # relative: a small model may not reach exactly zero

def test_gradients_flow(batch):
    x, y = batch
    model = build_model()
    nn.functional.cross_entropy(model(x), y).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

def test_save_load_roundtrip(batch, tmp_path):
    x, _ = batch
    a = build_model().eval()
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b = build_model().eval()
    b.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))
```

> [!WARNING]
> **Watch out: the "overfit one batch" threshold.** A very small model cannot memorise random labels as fast as a big one. A fixed target such as "loss below 0.1" makes the test fail for the wrong reason. Compare against the **starting loss** (as above), and allow enough steps.

**Tests that need randomness** should set the seed (`torch.manual_seed(0)`), so a failure is reproducible. `tmp_path` is a built-in fixture giving each test a fresh temporary folder.

---

## 15.4 Data tests

Test the data **assumptions** you rely on, ideally against a small sample committed to the repository (or generated in the test):

```python
# tests/test_data.py
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def _digest(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

def test_no_leakage_between_splits(tmp_path):
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        for split in ("train", "val"):
            Image.fromarray((rng.random((8, 8, 3)) * 255).astype("uint8")).save(tmp_path / split / f"{i}.png")
    train = {_digest(p) for p in (tmp_path / "train").glob("*.png")}
    val = {_digest(p) for p in (tmp_path / "val").glob("*.png")}
    assert not train & val
```

**More checks worth adding for real data:**
- Every image opens and has the expected number of channels.
- Labels are within `0 .. num_classes - 1`; class folders match the class list.
- One batch from the loader has the expected shape, dtype and value range (a normalised image has values roughly between -3 and 3, not 0 to 255).
- The training transform is random but the evaluation transform is not (run it twice on one image and compare).
- A fixed **preprocessing parity test**: load an image, preprocess it with the training code *and* with the serving code, assert the tensors match.

> [!TIP]
> **🟢 Extra: "golden" prediction test.** Keep one or two real images with their expected top class and approximate probability. A test that loads the saved model and checks them catches a broken export or a wrong class order immediately.

---

## 15.5 Running tests

```
pip install pytest
pytest -q                 # run everything, short output
pytest -x                 # stop at the first failure
pytest -k overfit         # run only tests whose name contains "overfit"
```

Put `tests/` next to your package. A minimal configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run them **before every training run you care about** and **in automation on every change** (15.7).

---

## 15.6 Packaging

Turning your code into an installable package means `import my_project` works from anywhere, and a teammate (or a server) can install it with one command.

**Layout:**

```
my_project/
├── pyproject.toml
├── src/my_project/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── predict.py
└── tests/
```

**`pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my_project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "pillow",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
serve = ["fastapi", "uvicorn", "python-multipart"]

[project.scripts]
my-predict = "my_project.predict:main"          # creates a command-line tool
```

```
pip install -e .            # editable install: edits to the code take effect immediately
pip install -e ".[dev]"     # plus the development tools
```

| Habit | Why |
|---|---|
| `src/` layout | Prevents tests accidentally importing the local folder instead of the installed package |
| Dependencies in `pyproject.toml` with sensible minimum versions | One place; easy to reinstall |
| **Pin exact versions** for deployment (`pip freeze > requirements.lock`, or a lock file) | The same code with a new library version can behave differently |
| Optional extras (`dev`, `serve`) | Light installs for users who only need prediction |
| `__init__.py` exposes the few functions users need | A clear public surface |

> [!CAUTION]
> **Not tested.** The Dockerfile and the GitHub Actions file below were not built or run. Adapt them to your project and test them before relying on them.

> [!NOTE]
> **🔵 Advanced alternative: containers and modern tooling.**
>
> **What changes:** ship a **Docker image** containing the code, the pinned dependencies and the model file, instead of asking others to rebuild the environment.
>
> **Why teams do it:** "works on my machine" becomes "works in the image". Tools like `uv` or `poetry` manage environments and lock files; `ruff` handles linting and formatting; `mypy` checks type hints.
>
> ```dockerfile
> FROM python:3.11-slim
> WORKDIR /app
> COPY requirements.lock .
> RUN pip install --no-cache-dir -r requirements.lock
> COPY src/ src/
> COPY model_bundle.pt .
> CMD ["uvicorn", "my_project.serve:app", "--host", "0.0.0.0", "--port", "8000"]
> ```
>
> Use the CPU-only PyTorch build in containers when you do not need a GPU: the images are much smaller.

---

## 15.7 Automation

Run tests automatically on every change so nobody has to remember.

> [!NOTE]
> **🔵 Advanced alternative: continuous integration.**
>
> ```yaml
> # .github/workflows/tests.yml   (GitHub Actions)
> name: tests
> on: [push, pull_request]
> jobs:
>   test:
>     runs-on: ubuntu-latest
>     steps:
>       - uses: actions/checkout@v4
>       - uses: actions/setup-python@v5
>         with: { python-version: "3.11" }
>       - run: pip install -e ".[dev]"
>       - run: ruff check .
>       - run: pytest -q
> ```
>
> **What changes:** every push triggers the checks on a clean machine.
>
> **Why teams do it:** it catches "works on my laptop" problems and keeps the main branch always working. Keep the tests fast, or CI becomes too slow to be useful. Also useful: **pre-commit hooks** that run `ruff` and formatting before each commit.

**What not to put in CI:** full training runs. Test the machinery on tiny data; run real training elsewhere.

---

## 15.8 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Tests too slow | Nobody runs them | Tiny models and data, a few seconds total |
> | Unseeded randomness in tests | A test passes sometimes and fails sometimes | Seed every test |
> | Overfit test with a fixed threshold | Fails for small models | Compare against the starting loss |
> | Testing accuracy instead of machinery | Tests depend on real data and take hours | Keep tests about shapes, flow, determinism, parity |
> | Preprocessing not tested | Live accuracy far below validation | Parity test between training and serving code |
> | Tests importing local files by accident | Pass locally, fail after install | `src/` layout and `pip install -e .` |
> | Unpinned dependencies | Same code, different results later | Lock file or pinned requirements |
> | Large files committed to git | Repository becomes huge | Keep data and weights out of git; use storage or a data versioning tool |
> | No golden prediction test | A broken export is not noticed | One or two known images with expected outputs |

---

## Key takeaways

1. ML bugs are **quiet**. Tests make them loud.
2. Test the **machinery**: shapes, overfit-one-batch, gradients, determinism, save/load, preprocessing parity, data assumptions.
3. Keep tests **fast and seeded**, using tiny models and data.
4. Package with `pyproject.toml` and a `src/` layout, **pin versions** for deployment, and consider a container.
5. Run tests automatically on every change, but never full training.

**Previous:** [Chapter 14: Export and serving](14_deployment.md) &nbsp;|&nbsp; **Next:** [Chapter 16: Monitoring and maintenance](16_monitoring.md)
