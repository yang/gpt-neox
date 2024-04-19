import subprocess as subp
import os, yaml, re, sys
from pathlib import Path

printlines = lambda xs: print("\n".join(xs))

outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/")
logs = outdir.glob("*.log")


def grep(lines, pat):
    experts = [
        l
        for l in lines
        if re.search(
            pat,
            l,
        )
    ]

    return experts


for log in logs:
    print()
    print("==", log, "==")
    ws = int(re.search(r"ws=(\d+)", str(log)).group(1))
    with open(log) as f:
        lines = [l.rstrip() for l in f.readlines()]
    content = "".join(lines)

    ne = len(set(re.findall(r"deepspeed_experts...dense_4h_to_h", content))) or 1

    experts = grep(
        lines,
        "experts...dense_4h_to_h.weight|moe.experts.mlp.w1|2.(module.)?mlp.dense_4h_to_h.weight",
    )
    print("experts:")
    printlines(experts[: ne * ws])
    print("..")
    printlines(experts[-ne * ws :])

    router = grep(lines, "gate.wg.weight|router.layer.weight")
    print("router:")
    printlines(router[:ws])
    print("..")
    printlines(router[-ws:])

    # for m in re.finditer(
    #     r"""^!! rank=(?P<rank>\d+) model .*?dense_4h_to_h.weight': '(.*?)'""",
    #     content,
    #     re.MULTILINE | re.DOTALL,
    # ):
    #     print(m.group(2))
