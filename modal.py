import os
import subprocess
import itertools
import time
import queue

from typing import Optional
from pathlib import Path

from modal import Image, Stub, Secret, method, asgi_app, Mount, Function, gpu, Volume

model = "codellama/CodeLlama-7b-Python-hf"

def prepare_model():
    subprocess.run(
        [
            "bash",
            "./scripts/prepare.sh",
            model,
            "--hf_token=" + os.environ["HUGGINGFACE_TOKEN"],
        ],
        check=True,
        cwd="/gpt-fast",
    )

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/nightly/cu118"
    )
    .pip_install(
        # Use the barebones hf-transfer package for maximum download speeds. No
        # progress bar, but expect 700MB/s.
        "hf-transfer~=0.1",
        "huggingface-hub",
        "sentencepiece",
    )
    .apt_install("git")
    .run_commands("git clone https://github.com/pytorch-labs/gpt-fast")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        prepare_model,
        secret=Secret.from_name("huggingface"),
        timeout=20 * 60, # 20 minutes
    )
)

stub = Stub("gpt-fast", image=image)
volume = Volume.from_name("gpt-fast-vol")

@stub.cls(
    gpu=gpu.A100(memory=80),
    timeout=40 * 60,
    keep_warm=1,
    volumes={"/volume": volume},
    container_idle_timeout=20 * 60,
)
class Model:
    def __init__(
            self,
            checkpoint_path: Path = Path(f"/gpt-fast/checkpoints/{model}/model.pth"),
            compile_model: bool = True,
            compile_prefill: bool = False,
            draft_checkpoint_path: Optional[Path] = None,
            # draft_checkpoint_path: Optional[Path] = Path(f"/gpt-fast/checkpoints/{model}/model_int8.pth"),
            profile: Optional[Path] = None,
        ):
        self.checkpoint_path = checkpoint_path
        self.compile_model = compile_model
        self.compile_prefill = compile_prefill
        self.profile = profile
        self.draft_checkpoint_path = draft_checkpoint_path

    def __enter__(self):
        import torch

        from . import generate
        from .tp import maybe_init_dist
        from .generate import _load_model

        assert self.checkpoint_path.is_file(), self.checkpoint_path

        global print
        rank = maybe_init_dist()
        use_tp = rank is not None
        if use_tp:
            torch.cuda.set_device(rank)
            if rank != 0:
                # only print on rank 0
                print = lambda *args, **kwargs: None

        self.device = 'cuda'
        precision = torch.bfloat16
        is_speculative = self.draft_checkpoint_path is not None


        t0 = time.time()
        print("Loading model weights ...")
        model = _load_model(self.checkpoint_path, self.device, precision, use_tp)

        if is_speculative:
            draft_model = _load_model(self.draft_checkpoint_path, self.device, precision, use_tp)
        else:
            draft_model = None

        torch.cuda.synchronize()
        print(f"Loading model weights took {time.time() - t0:.02f} seconds")

        torch.manual_seed(1234)
        if self.compile_model:
            if is_speculative and use_tp:
                torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

            if is_speculative:
                self.model_forward = torch.compile(generate.model_forward, mode="reduce-overhead", fullgraph=True)

            self.decode_one_token = torch.compile(generate.decode_one_token, mode="reduce-overhead", fullgraph=True)

            if self.compile_prefill:
                self.prefill = torch.compile(generate.prefill, fullgraph=True, dynamic=True)

        self.model = model
        self.draft_model = draft_model

        volume.reload()
        if False and Path("/volume/model3.pth").is_file(): # TODO(irfansharif): Figure out how to actually use this.
            print("Loading model from /volume/model3.pth ...")
            t0 = time.time()
            self.binary_model = torch.jit.load("/volume/model3.pth")
            print(f"Loaded compiled model from /volume/model3.pth in {time.time() - t0:.02f} seconds")
            print(f"Type: {type(self.binary_model)}")
        else:
            print("Running warmup inference ...")
            t0 = time.time()
            self.binary_model = None
            self.generate_inner(
                "What does 'lambda x: x + 1' do?",
                num_samples=0,
                max_new_tokens=100,
                speculate_k=5,
                temperature=0.8,
                top_k=200,
                interactive=False,
                q=queue.Queue(),
                sentinel=object(),
            )
            print(f"Warmup inference took {time.time() - t0:.02f} seconds")


    def __exit__(self, _exc_type, _exc_value, _traceback):
        print("Terminating instance!")

    @method()
    def generate(
            self,
            prompt: str,
            num_samples: int = 1,
            max_new_tokens: int = 500,
            speculate_k: int = 5,
            temperature: float = 0.8,
            top_k: int = 200,
            interactive: bool = False,
        ):
        import queue
        import threading

        q = queue.Queue()
        sentinel = object()

        threading.Thread(
            target=self.generate_inner,
            args=(prompt, num_samples, max_new_tokens, speculate_k, temperature, top_k, interactive, q, sentinel),
        ).start()

        while True:
            data = q.get()
            if data is sentinel:
                break
            yield data


    def generate_inner(
            self,
            prompt: str,
            num_samples: int,
            max_new_tokens: int,
            speculate_k: int,
            temperature: float,
            top_k: int,
            interactive: bool,
            q: queue.Queue,
            sentinel: object,
        ):
        import torch
        import contextlib

        from sentencepiece import SentencePieceProcessor

        from . import generate
        from .generate import (
            encode_tokens,
            B_INST,
            E_INST,
        )

        tokenizer_path = self.checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        is_chat = "chat" in str(self.checkpoint_path)
        if is_chat:
            prompt = f"{B_INST} {prompt.strip()} {E_INST}"

        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=self.device)
        prompt_length = encoded.size(0)

        model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(self.model.parameters(), self.model.buffers())])
        aggregate_metrics = {
            'tokens_per_sec': [],
            'accept_counts': [],
        }

        is_speculative = self.draft_checkpoint_path is not None
        if self.compile_model:
            if is_speculative:
                generate.model_forward = self.model_forward

            generate.decode_one_token = self.decode_one_token

            if self.compile_prefill:
                generate.prefill = self.prefill

        start = -1 if num_samples == 0 else 0 # used to initialize the model
        for i in range(start, num_samples):
            torch.cuda.synchronize()

            if i == 0:
                print(f"Starting inference for prompt = '{prompt}'")

            if interactive and i >= 0:
                buffer = []
                period_id = tokenizer.encode('.')[0]
                done_generating = False

                def callback(x):
                    nonlocal done_generating
                    if done_generating:
                        return

                    buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                    if x.item() == tokenizer.eos_id():
                        done_generating = True

                    if len(buffer) == 4 or done_generating:
                        q.put(''.join(buffer))
                        print(''.join(buffer), end='', flush=True)
                        buffer.clear()
                    # print(, end='', flush=True)
            else:
                callback = lambda x: x

            t0 = time.perf_counter()

            if (i != num_samples - 1 or not self.profile) or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y, metrics = generate.generate(
                    self.model,
                    encoded,
                    max_new_tokens,
                    draft_model=self.draft_model,
                    speculate_k=speculate_k,
                    interactive=interactive,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                )
                aggregate_metrics['accept_counts'].append(metrics['accept_counts'])

            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                try:
                    torch.jit.save(torch.jit.script(self.model), "/volume/model3.pth")
                    volume.commit()
                    print("Saved model to /volume/model3.pth")
                except Exception as e:
                    print(f"Failed to save model to /volume/model3.pth: {e}")

                continue

            if hasattr(prof, "export_chrome_trace"):
                if use_tp:
                    prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
                else:
                    prof.export_chrome_trace(f"{profile}.json")

            torch.cuda.synchronize()
            t = time.perf_counter() - t0

            if not interactive:
                generated = tokenizer.decode(y.tolist())
                q.put(generated)
                print(generated)
            else:
                print()

            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            aggregate_metrics['tokens_per_sec'].append(tokens_sec)
            print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
            print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        print("==========")

        if is_speculative:
            counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
            acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

        print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        q.put(sentinel)



# @stub.local_entrypoint()
@stub.function(
    gpu=gpu.A100(memory=80),
    timeout=40 * 60,
    volumes={"/volume": volume},
)
def main(lookup: bool = False):
    if not lookup:
        model = Model()
        fn = model.generate
    else:
        fn = Function.lookup("gpt-fast", "Model.generate")

    questions = [
        "Implement fibonacci in python.",
        # "Write a Rust function that performs binary exponentiation.",
        # "How do I allocate memory in C?",
    ]

    for question in questions:
        for generated in fn.remote_gen(prompt=question, max_new_tokens=500, interactive=False):
            print(generated, end='')



@stub.function(
    mounts=[
        Mount.from_local_dir(Path(__file__).parent / "llm-frontend", remote_path="/assets"),
    ],
    allow_concurrent_inputs=10,
    timeout=60 * 10,
)
@asgi_app(label="gpt-fast-app")
def app():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Function.lookup("gpt-fast", "Model.generate").get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            fn = Function.lookup("gpt-fast", "Model.generate")
            for generated in fn.remote_gen(unquote(question)):
                yield f"data: {json.dumps(dict(text=generated), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app

