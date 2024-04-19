import subprocess as subp
import os, yaml
from pathlib import Path

outdir = Path("out/")
outdir.mkdir(exist_ok=True)

os.environ["WANDB_MODE"] = "disabled"

if os.environ.get("debug"):
    cmd = "debugpy-run"
else:
    cmd = "python"

args = [
    # # ws=2
    # # dict(ws=2, dp=1, tp=2, ne=1),
    # dict(ws=2, dp=1, tp=2, ne=2, edp=1, ep=2, etp=1, mt="ds"),
    dict(ws=2, dp=1, tp=2, ne=2, edp=1, ep=1, etp=2, mt="ds"),
    # dict(ws=2, dp=1, tp=2, ne=2, edp=1, ep=1, etp=2, mt="mb"),
    # # just dp
    # # dict(ws=2, dp=1, tp=2, edp=2, ep=1, etp=1, mt="ds"),
    # # dict(ws=2, dp=2, tp=1, edp=2, ep=1, etp=1, mt="ds"),
    # # dict(ws=2, dp=2, tp=1, edp=1, ep=2, etp=1, mt="ds"),
    # # ws=4
    # # from the paper
    # dict(ws=4, dp=2, tp=2, ne=2, edp=1, ep=2, etp=2, mt="ds"),
    # dict(ws=4, dp=2, tp=2, ne=2, edp=1, ep=2, etp=2, mt="mb"),
]

mts = dict(ds="deepspeed", mb="megablocks")

for arg in args:
    assert arg["ws"] == arg["dp"] * arg["tp"]
    if arg["ne"] > 1:
        assert arg["edp"] * arg["ep"] * arg["etp"] == arg["ws"]
        assert arg["etp"] in [1, arg["tp"]]

for arg in args:
    gencfg = "configs/generated.yml"
    with open("configs/template.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["model_parallel_size"] = arg["tp"]
    cfg["moe_num_experts"] = arg["ne"]
    if arg["ne"] > 1:
        cfg["enable_expert_tensor_parallelism"] = arg["etp"] > 1
        cfg["moe_type"] = mts[arg["mt"]]
        cfg["moe_expert_parallel_size"] = arg["ep"]
    with open(gencfg, "w") as f:
        yaml.dump(cfg, f)

    logpath = outdir / ("-".join([f"{k}={v}" for k, v in arg.items()]) + ".log")
    print("Running", logpath)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(arg["ws"]))
    with open(logpath, "w") as f:
        subp.check_call(
            f"{cmd} deepy.py train.py {gencfg} configs/local_setup.yml",
            shell=True,
            stdout=f,
            stderr=f,
        )
