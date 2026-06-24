# Normal Frame Analysis for Tactile Sensor Renderings

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{n}_B$ | Surface normal in the object body frame (fixed to the object) |
| $R_{CB,0}$ | Rotation: object body → camera, at Stage 1 (original turntable pose) |
| $R_{CB,1} = R_{turn}\,R_{CB,0}$ | Rotation: object body → camera, after turntable rotation $R_{turn}$ |
| $R_{CS}$ | Rotation: GelSight sensor → camera, from ArUCo at press time |
| $R_{CS}'$ | Rotation: GelSight sensor → camera, at new turntable pose |

ZED's `sl.MEASURE.NORMALS` always returns normals in the **camera frame**.
`ortho_project_raw` rotates them into the **sensor frame** using the ArUCo-derived $R_{CS}$.

---

## Stage 1 — Initial capture (both implementations identical)

User presses `'c'` to freeze the view and segment the object.

$$\texttt{normals\_cached} = \mathbf{n}_C = R_{CB,0}\,\mathbf{n}_B$$

Touch saved at this pose:

$$\mathbf{n}_{saved} = R_{CS}^T\,R_{CB,0}\,\mathbf{n}_B$$

---

## Stage 2 — After pressing `'t'` (turntable rotated)

### Old implementation (with rotation correction)

The rotation correction tried to map fresh ZED normals back to the initial camera frame:

$$\texttt{normals\_cached} = R_{CB,0}\,R_{CB,1}^T \cdot R_{CB,1}\,\mathbf{n}_B = R_{CB,0}\,\mathbf{n}_B$$

Touch saved at new pose:

$$\mathbf{n}_{saved}' = R_{CS}'^T\,\mathbf{\color{red}{R_{CB,0}}}\,\mathbf{n}_B$$

### New implementation (no rotation correction)

`normals_cached` is updated directly from the current ZED frame:

$$\texttt{normals\_cached} = R_{CB,1}\,\mathbf{n}_B = R_{turn}\,R_{CB,0}\,\mathbf{n}_B$$

Touch saved at new pose:

$$\mathbf{n}_{saved}' = R_{CS}'^T\,R_{CB,1}\,\mathbf{n}_B = R_{CS}'^T\,R_{turn}\,R_{CB,0}\,\mathbf{n}_B$$

---

## Comparison table

| | **Old (with rotation correction)** | **New (current)** |
|---|---|---|
| `normals_cached` after Stage 1 | $R_{CB,0}\,\mathbf{n}_B$ | $R_{CB,0}\,\mathbf{n}_B$ |
| Saved normal, Stage 1 touch | $R_{CS}^T\,R_{CB,0}\,\mathbf{n}_B$ | $R_{CS}^T\,R_{CB,0}\,\mathbf{n}_B$ |
| `normals_cached` after `'t'` | $R_{CB,0}\,\mathbf{n}_B$ (rotated back to view 0) | $R_{CB,1}\,\mathbf{n}_B = R_{turn}\,R_{CB,0}\,\mathbf{n}_B$ |
| Saved normal, Stage 2 touch | $R_{CS}'^T\,\mathbf{\color{red}{R_{CB,0}}}\,\mathbf{n}_B$ | $R_{CS}'^T\,R_{CB,1}\,\mathbf{n}_B$ |
| Physically consistent? | **No** | **Yes** |

---

## Why the old implementation was wrong

The old correction stored $R_{CB,0}\,\mathbf{n}_B$ in `normals_cached` regardless of turntable angle.
This paired the **sensor pose at the new angle** ($R_{CS}'$) with the **object orientation from view 0** ($R_{CB,0}$) — a combination that has no physical interpretation.

The new implementation always pairs the sensor pose and object orientation from the same moment in time, producing normals that correctly represent what the GelSight physically measures when pressed against the surface.
