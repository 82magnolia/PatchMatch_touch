# Vendored ObjectFolder runtime

This directory contains the minimal files required for the
`baselines/objectfolder_inr` tactile renderer:

- `taxim_render.py`;
- `basics/sensorParams.py` and `basics/CalibData.py`;
- the matching legacy Taxim calibration in `calibs/`; and
- the upstream ObjectFolder license.

The files are copied from the ObjectFolder 2.0 source used by this project.
They are vendored so the PatchMatch baseline does not depend on a separate
`baselines/ObjectFolder` checkout. Do not mix `taxim_render.py` with a
different calibration without validating the optical model.

Upstream project: <https://github.com/rhgao/ObjectFolder>
