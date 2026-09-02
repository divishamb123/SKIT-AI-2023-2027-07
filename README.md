# Sprint 1: Data Preparation & Authentication Interface

**Student:** Divisha Manak Bohra (`23ESKCA038`)  
**Department:** Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur  
**Academic Year:** 2026–27 (Phase-II, 7th Semester)  
**Sprint Window:** 03-August-2026 to 20-September-2026  
**User Story:** *Preparing image data and authentication interface*

---

## 1. Form-2 Task Alignment

| S.No. | Form-2 Sprint 1 Task | Implementation | Status |
|---|---|---|---|
| 1 | **Collecting and organising image datasets for real/AI-generated classification** | Curated 132,000 real and synthetic images across CIFAKE and GenImage datasets. | Completed |
| 2 | **Creating balanced train, validation and test splits** | Generated 90k train, 10k val, 20k test, and 12k holdout splits with exact 50/50 class balance (`seed=42`). | Verified (0% leakage) |
| 3 | **Designing login and registration interface components** | Designed reusable UI primitives (`Input`, `Button`, `AuthCard`, `PasswordRequirements`). | Completed |
| 4 | **Building responsive authentication screens and handling representative form states** | Built responsive `/login` and `/register` pages with validation, loading, error, and session states. | Verified |

---

## 2. Dataset Partitioning & Stratification

All partition metadata is stored in [`datasets/splits/manifest.csv`](datasets/splits/manifest.csv) (132,000 records).

### Split Distribution

| Split | Partition Role | Total Images | Real ($y=0$) | AI Synthetic ($y=1$) | Class Balance | Sources |
|---|---|---|---|---|---|---|
| `train` | Training | 90,000 | 45,000 | 45,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `val` | Validation | 10,000 | 5,000 | 5,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `test` | Intact Test | 20,000 | 10,000 | 10,000 | 50.0% / 50.0% | CIFAKE (CIFAR-10 / SD-v1.4) |
| `holdout` | Out-of-Domain Holdout | 12,000 | 6,000 | 6,000 | 50.0% / 50.0% | GenImage (Real, SD, Midjourney, BigGAN) |
| **Total** | **All Splits** | **132,000** | **66,000** | **66,000** | **50.0% / 50.0%** | **Multi-Generator Corpus** |

### Generators & Sources

- `cifar10`: 60,000 samples (Real baseline)
- `stable-diffusion-1.4`: 60,000 samples (Latent diffusion)
- `real`: 6,000 samples (High-resolution photography)
- `stable-diffusion`: 5,000 samples (Diffusion)
- `midjourney`: 500 samples (Zero-shot holdout)
- `biggan`: 500 samples (Zero-shot holdout)

### Partition Integrity

- **Class Balance:** Exact 50.0% real and 50.0% fake across all 4 partitions.
- **Zero Leakage:** Confirmed 0 duplicate file paths across train, val, test, and holdout splits.

---

## 3. Authentication Interface

The web interface is built with **Next.js 14 (App Router)** and **Tailwind CSS**.

### Available Pages

- **Login (`/login`)**:
  - Email and password inputs with field-level validation.
  - Password visibility toggle (show/hide).
  - "Fill Demo Account" helper (`divisha.bohra@skit.ac.in`).
  - Active session view with user details and Sign Out option.
- **Registration (`/register`)**:
  - Full name, institutional email, password, and confirmation inputs.
  - Interactive password strength meter checking: minimum 8 characters, uppercase letter, number, and special character.
  - Redirects to `/login?registered=1` on success.
- **Root (`/`)**: Redirects automatically to `/login`.

---

## 4. Repository Structure

```text
Sprint-1_Divisha_Manak_Bohra/
├── README.md                           # Sprint 1 documentation
├── requirements.txt                    # Python dependencies
├── run_demo.sh                         # Single-command verification & run script
├── datasets/
│   ├── cifake/                         # CIFAKE image collection (120,000 images)
│   ├── genimage/                       # GenImage holdout collection (12,000 images)
│   └── splits/
│       └── manifest.csv                # Partition manifest (132,000 rows, seed=42)
├── scripts/
│   ├── prepare_image_datasets.py       # Dataset stratification and manifest generation
│   └── verify_sprint1_splits.py        # Split balance, zero-leakage, and path validation
├── tests/
│   ├── test_dataset_splits.py          # Pytest suite for dataset partition integrity
│   └── test_auth_validation.py         # Pytest suite for authentication form validation
└── frontend/                           # Next.js authentication frontend
    ├── package.json
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── src/
        ├── app/
        │   ├── layout.tsx              # Root HTML shell & dark theme
        │   ├── page.tsx                # Redirects to /login
        │   ├── login/page.tsx          # Login page & active session display
        │   └── register/page.tsx       # Registration page with validation meter
        ├── components/
        │   ├── auth/                   # AuthCard & PasswordRequirements meter
        │   ├── layout/Navbar.tsx       # Navigation bar (Login, Register, Sign Out)
        │   └── ui/                     # Input and Button primitives
        ├── lib/api/                    # Auth client, token storage & session handling
        └── types/                      # TypeScript auth and API interfaces
```

---

## 5. How to Run

### Option A: Run Everything (One Command)
```bash
./run_demo.sh
```

### Option B: Step-by-Step

1. **Verify Dataset Splits & Zero Leakage:**
   ```bash
   python3 scripts/verify_sprint1_splits.py
   ```

2. **Run Automated Test Suite:**
   ```bash
   pytest tests/
   ```

3. **Start the Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   Open **`http://localhost:3000`** (redirects to `http://localhost:3000/login`).
