aletheia/
├── core/
│   ├── atomic.py               # Atomic substrate simulation (AtomicState, SparseAtomicGrid, AtomicPhysiologyEngine)
│   ├── voxel.py                # Voxel-based physiology (VoxelState, VoxelPhysiologyMapping)
│   ├── memory.py               # Temporal Memory and Pattern Recognition
│   ├── mythos.py               # Mythos Forge (Narrative generation)
│   ├── aspiration.py           # Aspiration Engine
│   ├── reflection.py           # Reflective Engine
│   ├── volition.py             # Volition Transform Lattice
│   └── utils.py                # Shared utilities (randomness, logging config, etc.)
│
├── runtime/
│   ├── observer.py             # Observer interface (for external inputs / simulated senses)
│   ├── runtime_loop.py         # Main runtime loop activating the full agent
│
├── data/
│   ├── logs/                   # Logging outputs, state dumps
│   ├── models/                 # (Future) learned models or pattern archives
│
├── visualizations/
│   ├── plotter.py              # (Optional) Visualize atomic/voxel state & actions
│
├── tests/
│   ├── test_atomic.py
│   ├── test_voxel.py
│   ├── test_memory.py
│   └── ... (unit tests for core modules)
│
├── main.py                     # Entry point script to launch ALETHEIA
├── requirements.txt            # Dependencies
└── README.md                   # Overview & developer notes
