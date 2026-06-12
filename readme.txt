Project-19(DS)/
├── data/
│   ├── India_Eyes/      # Existing conjunctiva images (1, 2, 3...)
│   ├── India_Tongue/    # NEW: Tongue images (1, 2, 3...)
│   ├── India_Nails/     # NEW: Nail bed images (1, 2, 3...)
│   └── anemia_metadata.csv





Project-19(DS)/
├── data/
│   ├── India_Eyes/             # Folders 1, 2, 3...
│   └── Nail_Disease_Dataset/   # THE NEW DATASET
│       ├── healthy_nail/       # Label: 0 (Normal)
│       ├── blue_finger/        # Label: 1 (Anemic/Oxygen Risk)
│       ├── clubbing/           # Label: 1 (Anemic/Oxygen Risk)
│       └── ...
├── src/
│   └── nail_engine.py          # NEW: Processes category-based images
└── app.py                      # Updated to include nail classification