# Appendix A: Generating Synthetic Images with Diffusion Models

| | |
|---|---|
| **Level** | Optional, advanced |
| **You will learn** | What text-to-image diffusion models can and cannot do for a vision project, and how to use one to create extra training images responsibly |
| **Prerequisites** | Chapters 2, 10 |
| **Ready-made code** | Snippets in this appendix (no toolkit module) |

> **How to read the colored boxes**
> 🟢 **Extra**: goes beyond the core recipe &nbsp;|&nbsp; 🟡 **Watch out**: a common problem and how to spot it &nbsp;|&nbsp; 🟣 **Outdated vs current**: older practice and what replaced it &nbsp;|&nbsp; 🔵 **Advanced**: a production alternative

> [!CAUTION]
> **Not tested.** The code in this appendix was never run. The ideas and cautions are the useful part.

> [!NOTE]
> Generative models change faster than almost anything in this guide. The **ideas and cautions** below are stable. Specific model names, class names and arguments are not: read the model card of whichever model you choose.

---

## A.1 The idea

**Data scarcity** is a common bottleneck: you need examples of a rare defect (a specific insect bite, a hairline crack), and you have twelve. A **text-to-image diffusion model** can generate pictures from a text description ("a mango with a small wormhole, close-up photo"), which you might add to your training set.

It starts from random noise and removes the noise step by step, guided by your text, until an image appears.

---

## A.2 When it helps, and when it does not

| Helps | Does not help |
|---|---|
| Rare cases that are visually simple and well described in words | Subtle, specialised appearances the model never saw (medical findings, fine industrial defects) |
| Adding variety (backgrounds, lighting) around real examples | Replacing real data entirely |
| Bootstrapping a first model before real data arrives | Evaluating your model (see below) |
| Illustrating or stress-testing a model | Anything where a wrong-looking image would teach the wrong thing |

> [!WARNING]
> **The cautions that matter most**
> 1. **Never evaluate on generated images.** Test only on **real** images (Chapter 10). A model can score well on fakes and fail on reality.
> 2. **Check for a "tell".** If your fake images share a texture, style or artefact that real images lack, and the fakes all belong to one class, the model learns the tell instead of the defect: a shortcut (Chapter 12). Look at heatmaps and test on real data.
> 3. **Generated images are not guaranteed to show what you asked for.** Review them by eye and discard bad ones.
> 4. **Licence and policy:** check the model's licence and your organisation's rules on synthetic data.

---

## A.3 The basic workflow

```python
# pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",             # an older model; check the hub for current ones
    torch_dtype=torch.float16,                         # 16-bit: GPU only; use float32 on CPU (very slow)
).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)      # makes the result repeatable
image = pipe(
    prompt="a mango with a small wormhole, close-up photo",
    negative_prompt="cartoon, drawing, text, watermark",         # what to avoid
    num_inference_steps=30,
    generator=generator,
).images[0]
image.save("mango_wormhole_seed42.png")
```

| Setting | What it does | Typical |
|---|---|---|
| **Prompt** | What you want | Specific and photographic ("close-up photo", the lighting, the surface) |
| **Negative prompt** | What to avoid | "cartoon, drawing, blurry, text, watermark" |
| **Steps** | How many denoising steps: more is slower, with diminishing gains | 20 to 50 |
| **Seed** | Fixes the starting noise | Record it: the same seed, prompt and steps reproduce the image |
| **Guidance scale** | How strictly to follow the prompt (a higher value means closer to the prompt, with less variety and sometimes harsher images) | The library default is a good start |

**Workflow for a project:**
1. Write several prompts for the rare case, varying object, background and lighting.
2. Generate in batches with different seeds; **save the prompt and seed** with each image.
3. **Review every image**, keeping only convincing ones.
4. Add them to the **training set only**, labelled with the correct class, and keep a record that they are synthetic.
5. Retrain, then evaluate on **real** validation and test images, and compare with the model trained without the fakes. If real-data accuracy does not improve, the fakes are not helping.

> [!TIP]
> **🟢 Extra: a stronger use.** Use generation to *vary real images* (image-to-image or inpainting: change the background, add a defect to a real photo) rather than creating images from nothing. The result stays closer to your real data.

---

## A.4 Common problems and how to spot them

> [!WARNING]
> | Problem | How to spot it | Fix |
> |---|---|---|
> | Evaluated on fakes | Great scores, poor real-world accuracy | Real images only for validation and test |
> | A shortcut learned from fakes | Heatmaps focus on texture or style; real-image recall for that class is low | Mix more real data; make fakes less uniform; test on real |
> | Out-of-memory error | Crash on load or generate | Use half precision, a smaller image, or `pipe.enable_attention_slicing()` |
> | Slow on CPU | Minutes per image | Use a GPU or a hosted service |
> | Cannot reproduce an image | Different output each run | Record and reuse the seed, prompt, steps and library versions |
> | Unusable images | Distorted objects, wrong content | Review and filter; refine prompts; consider image-to-image |
> | Licence problems | Unclear right to use outputs commercially | Read the model licence first |

---

## Key takeaways

1. Diffusion models can add variety or bootstrap rare cases, but **do not replace real data**.
2. **Evaluate only on real images.**
3. Watch for a **tell** the model can use as a shortcut.
4. Record **prompt, seed, steps and model version**, and review every image.
5. Measure whether it **actually improves real-image accuracy**. If not, drop it.
