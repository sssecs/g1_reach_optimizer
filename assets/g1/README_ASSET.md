# G1 model assets

Place the exact Unitree MJLab model here:

```text
assets/g1/g1.xml
assets/g1/assets/*.STL
```

Source:
`unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/`

The package does not convert the model to URDF because the optimizer relies on
the MJCF collision geoms and sites.
