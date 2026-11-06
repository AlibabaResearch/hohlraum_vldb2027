# Bundled SGLang and HiSim

`sglang/` contains the complete tracked source snapshot used by this artifact,
including the Hohlraum integration changes and `tools/sglang-simulator/`.
There is no nested Git repository or submodule, and no patch step is required.
See `UPSTREAM.json` for the comparison base and changed native source files,
and `../hohlraum/docs/UPSTREAM_CHANGES.md` for their purpose.

Install through `../scripts/install.sh` so that the bundled source is selected.
Internal profiles, tokenizer/model assets, experiment outputs and local virtual
environments are not part of this source snapshot. Original licenses and notices
are retained in `sglang/`.
