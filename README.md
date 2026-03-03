# CV Research 2026

Goal (Week 1): build a minimal training loop that can run, log (W&B/TensorBoard), and save checkpoints.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train --epochs 3 --batch_size 128 --lr 0.1 --seed 42
```


## Week 1: Minimal training loop (CIFAR10 / ResNet18)

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```


### Train

<pre class="overflow-visible! px-0!" data-start="985" data-end="1020"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>scripts\train_cifar.bat</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### TensorBoard

<pre class="overflow-visible! px-0!" data-start="1038" data-end="1112"><div class="relative w-full my-4"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>tensorboard </span><span class="ͼu">--logdir</span><span> outputs/runs</span><br/><span class="ͼl"># open http://localhost:6006</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### Outputs

* Logs: `outputs/runs/cifar_resnet18/`
* Checkpoints: `outputs/checkpoints/cifar_resnet18/`
* Figure: `outputs/figures/tensorboard_train.png`


---
## 5）把该忽略的大文件加入 .gitignore（非常重要）
确认 `.gitignore` 里有这些（没有就加）：

```gitignore
.venv/
data/
outputs/runs/
---
