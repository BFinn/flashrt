# Clean-room policy

flashrt is Apache-2.0. Everything in this repository must be either written here or
vendored from a project whose license allows it, with that license kept alongside.

## Allowed sources

| Source | License | May we copy code? |
|---|---|---|
| ggml / llama.cpp | MIT | Yes: keep the MIT notice in the vendored directory |
| ik_llama.cpp | MIT | Yes, same |
| CUTLASS | BSD-3-Clause | Yes, same |
| vLLM, SGLang | Apache-2.0 | Yes: keep their NOTICE entries and mark modified files |
| Papers, blog posts, public benchmarks | n/a | Ideas and algorithms only; cite them |

## Not allowed: Strata

[Strata](https://github.com/Niko1221/Strata) has no license file, so its code is all
rights reserved. **We may use its ideas**, from its README, `docs/DETAILS.md`, the paper
(`docs/paper/Strata-Paper.pdf`) and our own measurements. **We may not copy, translate
or paraphrase its source code.**

In practice:

- Do not open Strata's `src/` or `tools/` while writing flashrt code. The design
  notes in `docs/design.md` are the interface between the two.
- If Strata gains an OSI license later, this policy is revisited and recorded here with
  the license and date.
- Reviews ask for the origin of any non-trivial kernel or algorithm (paper, our
  derivation, or a vendored permissive file).

## Modified third-party files

Apache-2.0 section 4(b): modified files carry a prominent notice that they were changed.
Use a header line such as:

```
// Modified for flashrt (2026-xx-xx): <what changed>. Original: <project>, <license>.
```
