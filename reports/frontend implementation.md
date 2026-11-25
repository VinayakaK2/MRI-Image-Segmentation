# **Cardiac MRI Segmentation — Frontend UI/UX & Integration Specification**

Planning approved

Integrated development

Adaptive theme

No coding yet

**Document type:** Product UI/UX Specification + Frontend Architecture + Backend Integration Contract

**Project:** Robustness Evaluation of a Lightweight U-Net for Cardiac MRI Segmentation Under Controlled Image-Quality Variations.

**Selected approach:** Integrated frontend/backend development.

**Theme:** Dark and light modes, with user-controlled switching.

## **1. Existing project inspection — findings**

Maine available project documents inspect kiye hain, particularly:

- `Project Implementation Plan.txt`
- `Research Explain Karo.txt`
- `Branch · Project Implementation Plan.txt`
- `English presentation script.txt`
- Phase 5 locked experiment specifications
- Latest research evidence-audit instructions

Ek important limitation: current project ke uploaded documents inspect kiye ja sakte hain, lekin actual local repository, running backend, source code aur API endpoints is conversation mein accessible nahi hain. Isliye main unke existence ya implementation status ka false claim nahi karunga.

### **1.1 Confirmed research scope**

| Component        | Existing project specification                    |
| ---------------- | ------------------------------------------------- |
| Dataset          | ACDC Cardiac MRI                                  |
| Cardiac phase    | End-Diastole (ED)                                 |
| Model            | `LightweightUNet2D`                               |
| Input            | 256 × 256, 2D MRI slice                           |
| Output           | 4-class segmentation                              |
| Classes          | Background, RV, MYO, LV                           |
| Parameters       | Approximately 1.96 million                        |
| Test cohort      | 15 held-out patients                              |
| Test images      | 131 ED slices, as reported                        |
| Robustness tests | Gaussian noise, Gaussian blur, contrast reduction |
| Mitigation       | CLAHE, median filtering, morphological processing |
| Evaluation       | Dice, IoU, statistical comparisons                |

The model uses a frozen checkpoint during the primary robustness experiments. The original research objective is to evaluate robustness, not introduce a new architecture or perform clinical diagnosis.

### **1.2 Important scientific constraints**

Frontend design ko existing research findings accurately represent karne honge.

| Research finding                                             | Frontend implication                                  |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| Severe blur caused substantial degradation                   | Blur experiment visualization important hai           |
| Contrast reduction affected performance                      | Contrast experiment separately display hoga           |
| Noise had comparatively small impact                         | Noise results ko exaggerate nahi karenge              |
| CLAHE attribution was challenged by normalization invariance | CLAHE ko proven successful mitigation nahi dikhayenge |
| Morphological improvements were modest/inconclusive          | Statistical significance separately display hogi      |
| Median filtering did not establish meaningful improvement    | Negative results preserve honge                       |

Ek aur crucial distinction: project documents mein patient-level Dice, slice-level Dice aur volume-level aggregation ki terminology use hui hai. Frontend mein metric aggregation definition ke bina koi generic `Dice Score` card nahi banega.

### **1.3 Existing implementation versus proposed architecture**

**Verified in project documents**

Documented

- Python / PyTorch research pipeline
- NiBabel, NumPy, SciPy
- Research scripts and experiment configurations
- Model checkpoints and evaluation outputs are part of the project workflow

  **Not yet verified**

  Inspection required
- Existing FastAPI/Flask server
- REST endpoints
- Database and artifact storage implementation
- Inference job execution mechanism
- Browser image-viewer implementation
- Actual repository directory structure

**Decision:** Frontend specification will use a React + TypeScript architecture, but the API contract below is a proposed contract, not a description of existing endpoints.

# **2. Product definition**

## **2.1 What are we building?**

We are building a **Cardiac MRI Research Workstation**.

It has three primary responsibilities:

1. Allow users to perform and inspect cardiac MRI segmentation.
2. Allow users to explore controlled image-quality and mitigation experiments.
3. Allow users to understand and compare scientific results with traceable evidence.

It is not:

- A hospital management system.
- A patient diagnosis platform.
- A treatment recommendation system.
- A replacement for a radiologist.
- A general-purpose medical imaging platform.

## **2.2 Primary users and their goals**

| User                | Main goal                                 |
| ------------------- | ----------------------------------------- |
| Researcher          | Explore model performance                 |
| Student             | Demonstrate research implementation       |
| Evaluator / Faculty | Inspect methodology and results           |
| Technical reviewer  | Verify model and experiment configuration |

We will initially optimize for a single research workspace rather than building a complex multi-hospital SaaS product.

## **2.3 Main user journeys**

## **Journey A — Run segmentation**

1. Open Segmentation Studio
2. Upload a supported MRI image or volume
3. Validate image and metadata
4. Select the verified model checkpoint
5. Execute inference
6. Inspect original image and predicted mask
7. Export segmentation artifacts

## **Journey B — Explore robustness**

1. Open Robustness Lab
2. Select noise, blur or contrast
3. Select an existing tested severity
4. Inspect corrupted MRI and segmentation
5. Compare against clean baseline
6. Inspect patient-wise and class-wise metrics

## **Journey C — Compare mitigation**

1. Open Mitigation Lab
2. Select a registered experiment family
3. Compare unmitigated and processed results
4. Inspect statistical evidence
5. Review conclusions and limitations

# **3. Global application structure**

## **3.1 Navigation architecture**

Seven primary navigation destinations rahenge.

| Route            | Page                | Purpose                 |
| ---------------- | ------------------- | ----------------------- |
| `/overview`      | Overview            | Research summary        |
| `/segmentation`  | Segmentation Studio | Image inference         |
| `/robustness`    | Robustness Lab      | Degradation experiments |
| `/mitigation`    | Mitigation Lab      | Processing comparisons  |
| `/analytics`     | Research Analytics  | Statistical exploration |
| `/reports`       | Reports             | Research exports        |
| `/model-dataset` | Model & Dataset     | Technical information   |

Experiment aur result detail ke liye nested routes:

```
/robustness/:experimentId
/mitigation/:experimentId
/segmentation/jobs/:jobId
/reports/:reportId
```

Ye routes proposed hain; final routing actual backend resource model aur user workflow inspect karne ke baad freeze hogi.

## **3.2 Application shell**

**CardioAI Research**

System ready

WORKSPACE

Overview

Segmentation

Robustness

Mitigation

Analytics

Reports

Settings

**Research Overview**

Cardiac MRI segmentation and robustness

Baseline Dice

# **—**

Awaiting verified artifact

Test patients

# **15**

Reported cohort

**Research performance**

**Recent experiments**

Noise · Blur · Contrast

Structural wireframe. The actual theme, spacing and components will be implemented later.

### **Global shell behaviour**

- Sidebar collapses on smaller screens.
- Current route receives a clear active state.
- Theme selection persists across sessions.
- Header shows backend connectivity status.
- Long-running operations have a persistent job indicator.
- Page navigation should not accidentally discard an active job.
- Errors and warnings use consistent notification patterns.

# **4. Adaptive visual theme specification**

Adaptive theme ka matlab sirf background colour change karna nahi hai. Dono themes mein readability, image contrast, charts, dialogs aur controls consistent hone chahiye.

## **4.1 Theme modes**

Three modes:

- Light
- Dark
- System (follow operating system preference)

Default: System.

User manual selection kare toh system preference override hogi.

## **4.2 Design tokens**

| Token             | Light          | Dark             |
| ----------------- | -------------- | ---------------- |
| Main background   | Cool off-white | Deep charcoal    |
| Surface           | White          | Dark slate       |
| Secondary surface | Light grey     | Elevated slate   |
| Primary text      | Dark navy      | Near-white       |
| Secondary text    | Slate          | Muted grey       |
| Accent            | Teal           | Bright teal      |
| Success           | Green          | Accessible green |
| Warning           | Amber          | Accessible amber |
| Error             | Red            | Accessible red   |
| Borders           | Light grey     | Subtle slate     |

Exact colour values design implementation ke waqt lock honge.

## **4.3 Typography and spacing**

- Font: Inter or equivalent.
- Page title: 24–28 px.
- Section title: 17–20 px.
- Body: 14–16 px.
- Secondary metadata: 12–13 px.
- Card radius: 12–16 px.
- Standard spacing scale: 4, 8, 12, 16, 24, 32 px.

## **4.4 Medical-image viewer theme exception**

Ek important design decision:

**MRI image viewer ka image canvas neutral black/dark background par rahega, even when the application uses light theme.**

Reason: Medical image intensity visualization application theme se independent honi chahiye.

Similarly:

- Class colors must remain identical in both themes.
- Overlay opacity must be theme-independent.
- Image grayscale values must never be altered just because the theme changes.
- Chart colours must remain distinguishable in both themes.

## **4.5 Interaction design**

Our interaction principles:

1. Buttons should have clear hover, focus, pressed and disabled states.
2. No unnecessary animation during image processing.
3. Short transitions for navigation and panels.
4. No fake loading percentages.
5. Destructive actions require confirmation.
6. Tooltips for technical metrics and unfamiliar controls.
7. Keyboard accessibility for major controls.

# **5. Detailed page-wise UI/UX specification**

## **Page 01 — Overview Dashboard**

**Route:** `/overview`

**Purpose:** Give a reliable snapshot of the complete research project.

### **Layout**

[Cardiac MRI Scan Analysis by Alfikri Djati for Callour Studio on Dribbble](https://images.openai.com/static-rsc-4/tIR9wAuPkahu47BiuZagtgChWjjG3LdXNYdhfsDm63KVxJVIT00n_VFTgPiNmRp3h10cnHvQkrTsueKjzskrt9ZMWpdZu17azVgU5nOVSnhR-zcktMPxq2QDErsY9sRrvSZVW40OqJoscybzLdLN2f9Dv7hvkQc6Hj8F5fav3ak?purpose=inline)

[dribbble.com](https://dribbble.com/shots/27050894-Cardiac-MRI-Scan-Analysis?utm_source=chatgpt.com)

| Section            | Components                           |
| ------------------ | ------------------------------------ |
| Header             | Page title, selected model, date     |
| Summary cards      | Baseline Dice, IoU, patients, slices |
| Research overview  | Dataset and model metadata           |
| Robustness summary | Noise, blur, contrast                |
| Mitigation summary | Processing methods and findings      |
| Performance graph  | Clean vs corrupted conditions        |
| Recent experiments | Experiment history                   |
| Quick actions      | Segmentation, experiments, reports   |

### **Component tree**

```
OverviewPage
├── PageHeader
├── ResearchStatusBanner
├── BaselineMetricGrid
│   ├── MetricCard
│   ├── MetricCard
│   ├── MetricCard
│   └── MetricCard
├── ResearchSummary
├── RobustnessSummary
│   ├── CorruptionSummaryCard
│   └── CorruptionSummaryCard
├── PerformanceComparisonChart
├── MitigationSummary
└── RecentExperimentsTable
```

### **Functional behaviour**

- Clicking a metric opens its detailed analysis.
- Clicking a corruption card opens the corresponding robustness experiment.
- Recent experiment rows open their details.
- Empty states explain when data is unavailable.
- Metric cards show the aggregation method and cohort context.

### **Critical requirement**

Dashboard must not hardcode experimental values.

It must retrieve data from the canonical research result artifacts.

If a result is not verified or unavailable, display `Not available` or `Not verified`, not a guessed value.

## **Page 02 — Segmentation Studio**

**Route:** `/segmentation`

Highest priority

This is the primary interactive workspace.

### **5.2.1 Workspace layout**

[syngo.via for MRI - Siemens Healthineers](https://images.openai.com/static-rsc-4/JEgn54EhEfA32UM9EPaPrSDREuwlA9fRiKWhJWNPNc17oIAtBzRyxxunB3RqmmnxwhgI0vxToQINeF573rASTgA03Q3mUBlEVJ--siIn163XyBqvGK5bh36HcDzulSYREez5fb0Ns_FUC3-EU2egkrvUwwl5om_PGDUcwPudkug?purpose=inline)

[siemens-healthineers.com](https://www.siemens-healthineers.com/magnetic-resonance-imaging/advanced-imaging-applications/syngo-via?utm_source=chatgpt.com)

Recommended three-panel layout:

| Panel  | Width guideline | Responsibility              |
| ------ | --------------- | --------------------------- |
| Left   | 260–300 px      | Upload and configuration    |
| Centre | Flexible        | MRI image viewer            |
| Right  | 280–320 px      | Result details and controls |

These widths are desktop design targets, not fixed implementation constraints.

### **5.2.2 Upload component**

Component: `MRIUploadPanel`

Supported input types must come from the actual backend loader.

The project currently documents NIfTI-based MRI processing. We should initially prioritize:

- `.nii`
- `.nii.gz`

Other formats, including PNG and JPEG, should only be enabled if the inference pipeline explicitly supports them.

Upload states:

| State      | UI behaviour           |
| ---------- | ---------------------- |
| Idle       | Dropzone               |
| Selecting  | File selected          |
| Uploading  | Progress if measurable |
| Validating | Validation status      |
| Valid      | Show file metadata     |
| Invalid    | Explain error          |
| Failed     | Retry or remove        |

Validation requirements:

- File size
- Extension and actual file signature
- Corrupted file
- Image dimensions
- Data type
- Non-finite pixel values
- Image-mask geometry compatibility, when applicable
- Spatial metadata availability

**NIfTI metadata must not be discarded.**

Especially important:

- Voxel spacing
- Affine matrix
- Orientation
- Dimensions
- Spatial units

### **5.2.3 MRI image viewer**

Component: `MRIViewer`

[MRI heart or Cardiac MRI in short axis view showing cross-sections of the left and right ventricle for diagnosis heart disease.](https://images.openai.com/static-rsc-4/Mlnh20B8LKLjC5WtVgLVXZiVqOx5ssVT5qpK8aDRWNbfpssB_ZrezP3c0uB5BNjSMA0hNRC0H3r4FMoHxSDK8g-cjCyk2zv2Da_UARxs2how5iF1v7sw0jN9DsMPb8w_3QEP_eyD-GJTn-kv7oUl25TUlL8mwl2qgu3EUrWu36lR0B3Fiw6oa6giZj7EMcGB?purpose=inline)

Viewer controls:

| Control      | Function                      |
| ------------ | ----------------------------- |
| Zoom         | Magnification                 |
| Pan          | Move image                    |
| Reset        | Restore default view          |
| Window/Level | Adjust display contrast       |
| Slice slider | Navigate volume slices        |
| Slice index  | Current slice / total slices  |
| Crosshair    | Optional spatial inspection   |
| Metadata     | Image geometry and dimensions |

For initial 2D-only inference, slice navigation is unnecessary unless the user supplies a volume and the backend supports slice extraction.

If volume inference is supported, the viewer should preserve physical-space orientation and show the selected slice index.

### **5.2.4 Segmentation configuration**

Component: `SegmentationConfigPanel`

Fields:

| Field              | Behaviour                                    |
| ------------------ | -------------------------------------------- |
| Model architecture | Display supported architecture               |
| Checkpoint         | Select available verified checkpoint         |
| Inference mode     | Supported 2D slice mode                      |
| Preprocessing      | Display actual configured pipeline           |
| Device             | Server-selected CPU/GPU, read-only initially |
| Ground truth       | Optional matching mask                       |

We should not allow arbitrary model architecture configuration through the frontend in the MVP.

### **5.2.5 Segmentation execution**

Component: `InferenceControl`

Flow:

```
Select image
    ↓
Validate input
    ↓
Choose checkpoint
    ↓
Review configuration
    ↓
Start inference
    ↓
Track job status
    ↓
Retrieve results
```

Button states:

- Run Segmentation
- Starting
- Queued
- Processing
- Completed
- Failed
- Cancelled, if cancellation is supported

A retry should not silently create a duplicate experiment record.

### **5.2.6 Result viewer**

Component: `SegmentationResultViewer`

[IMA205 Challenge 2023 | Kaggle](https://images.openai.com/static-rsc-4/q3EIDhFdIP02v-6CrYSWVKRJdjyDgw-pdiKnbygCDFyWNt2xuy05cPdBdp8EVCBY6kkKqJr5YUKHoMsFoBgea626FGo0ZGRhHcWM6_cbNQSQDiyXuxLfGqTgy5Xam5GZEkcrQ2VKkluNIkaDDJNPFqyJlmbDUb0xvfhqW_rQU6U?purpose=inline)

[kaggle.com](https://www.kaggle.com/competitions/ima205-challenge-2023?utm_source=chatgpt.com)

Viewing modes:

1. Original MRI
2. Ground Truth
3. Predicted Mask
4. Overlay
5. Side-by-side comparison

Class controls:

| Class      | Colour                          | Toggle   |
| ---------- | ------------------------------- | -------- |
| Background | Transparent / hidden by default | Optional |
| RV         | Cyan                            | Yes      |
| MYO        | Green                           | Yes      |
| LV         | Orange                          | Yes      |

Colours are illustrative until locked in the design system.

Additional controls:

- Overlay opacity
- Individual class visibility
- Reset visibility
- Compare ground truth/prediction
- Export current view

### **5.2.7 Metrics panel**

Component: `SegmentationMetricsPanel`

Display:

- Dice
- IoU
- Precision
- Recall
- HD95, if valid
- Class-wise metrics

Each metric must contain:

- Value
- Metric definition
- Aggregation method
- Evaluation cohort
- Ground-truth availability
- Metric computation status

If ground truth is absent:

**Evaluation unavailable**

A reference ground-truth segmentation mask was not provided. Prediction visualization is available, but Dice and IoU cannot be calculated.

## **Page 03 — Robustness Lab**

**Route:** `/robustness`

[When medical AI meets messy reality - MBZUAI](https://images.openai.com/static-rsc-4/xReia6ym8goZtuWKZvOCtf7ua8JG7q9dsd0mXsHZ_Wycx6tPXsCtRJMXjJQ_Ttg27Oyw0aRWrdapUbMs4nzl2kNfd2a_nNDrMWcMZuq--DcZwCZx-sGU5_XVO8CGWT2C9TIpgLHNHEIEA2KJuJKb-1wWBhC9CijIWLVqbhQlxGg?purpose=inline)

[mbzuai.ac.ae](https://mbzuai.ac.ae/news/when-medical-ai-meets-messy-reality/?utm_source=chatgpt.com)

### **Layout**

```
RobustnessLabPage
├── PageHeader
├── ExperimentSelector
├── CorruptionCategoryTabs
│   ├── NoiseTab
│   ├── BlurTab
│   └── ContrastTab
├── SeveritySelector
├── ImageComparisonViewer
│   ├── CleanImage
│   └── CorruptedImage
├── PredictionComparison
├── PerformanceDegradationChart
├── ClassWiseMetricTable
├── PatientWiseResults
└── ExperimentMetadata
```

### **Corruption categories**

**A. Gaussian Noise**

Display:

- Corruption parameters
- Random seed
- Severity
- Original MRI
- Corrupted MRI
- Predicted mask
- Dice/IoU difference

**B. Gaussian Blur**

Display:

- Blur parameters
- Kernel configuration
- Severity
- Image comparison
- Prediction changes
- Boundary degradation

**C. Contrast Reduction**

Display:

- Contrast transformation parameters
- Original intensity distribution
- Transformed intensity distribution
- Prediction comparison
- Normalization pipeline information

### **Severity controls**

Only pre-existing registered levels will appear by default.

If we later support custom corruption levels, those runs will be marked exploratory and kept separate from the original locked experiment.

### **Graphs**

- Severity vs Dice
- Severity vs IoU
- Class-wise degradation
- Patient-wise degradation
- Clean vs corrupted comparison

The UI should distinguish absolute and relative performance changes.

For example:

\\[ \Delta Dice = Dice\_{corrupted}-Dice\_{clean} \\]

\\[ Relative\ degradation = \frac{Dice\_{clean}-Dice\_{corrupted}}{Dice\_{clean}}\times100 \\]

### **Important feature: Same-patient comparison**

Every comparison must use matching patient identifiers and corresponding evaluation conditions.

We should never compare a clean result from one patient against a corrupted result from another patient and present it as a paired experiment.

## **Page 04 — Mitigation Lab**

**Route:** `/mitigation`

[Publications - Homepage](https://images.openai.com/static-rsc-4/aFwTwJ5V9Ewy8f_mspoYrxOSO73MGzvrCEZI1UNsiu8_5OKTFrIUAops7Oj8VmZM22cBByjNFoK00RV8dHqNgOLXOb5tfTAFWu8iiL4ObZEfhyK54STdjGwJRHPYPh_7lt5Qrw4JjvUuoVvQFiw6pXUIsgsO-SLSAyKdrZmD6qM?purpose=inline)

[wangziblake.github.io](https://wangziblake.github.io/publications/?utm_source=chatgpt.com)

### **Component architecture**

```
MitigationLabPage
├── PageHeader
├── MitigationExperimentSelector
├── ExperimentFamilyTabs
├── ProcessingConfiguration
├── InputImageComparison
├── PredictionComparison
├── ClassWiseMetrics
├── StatisticalResults
├── NegativeControlResults
└── ScientificInterpretation
```

### **Experiment families**

Based on the locked Phase 5 specification:

| Family                   | Comparison            | Primary hypotheses |
| ------------------------ | --------------------- | ------------------ |
| Post-processing alone    | All 10 conditions     | 10                 |
| CLAHE alone              | 3 contrast conditions | 3                  |
| CLAHE + post-processing  | 3 contrast conditions | 3                  |
| Median + post-processing | 3 noise conditions    | 3                  |
| **Total**                |                       | **19**             |

The fixed parameters are:

| Technique            | Configuration                               |
| -------------------- | ------------------------------------------- |
| CLAHE                | `clip_limit=0.01`, `tile_grid_size=(16,16)` |
| Median filtering     | `kernel_size=3`                             |
| RV LCC               | Minimum 30 pixels                           |
| MYO LCC              | Minimum 50 pixels                           |
| LV LCC               | Minimum 30 pixels                           |
| Myocardium safeguard | `max_hole_size=16 px`                       |

These are documented experiment specifications, not user-editable settings.

### **Statistical result component**

Component: `StatisticalComparisonPanel`

Display:

- Paired Student's t-test
- Wilcoxon signed-rank test
- Raw p-value
- Holm-adjusted p-value
- Effect size
- Sample size
- Mean difference
- Confidence interval, where available

Holm correction must be shown separately for each registered hypothesis family.

The UI must not imply that four family-local Holm corrections automatically provide global family-wise error control over all 19 hypotheses.

### **Scientific interpretation component**

Component: `EvidenceInterpretationPanel`

Every finding will receive one of these classifications:

| Status                 | Meaning                                        |
| ---------------------- | ---------------------------------------------- |
| Verified               | Supported by verified evidence                 |
| Observed               | Measured but not statistically established     |
| Mathematically derived | Supported by a checked mathematical derivation |
| Not tested             | No corresponding experiment                    |
| Contradicted           | Evidence conflicts with claim                  |
| Unverifiable           | Required artifacts unavailable                 |

This is particularly important for the CLAHE normalization-invariance investigation.

## **Page 05 — Research Analytics**

**Route:** `/analytics`

[Building an Interactive EDA Dashboard with Plotly | by Pradum Shukla | Accredian | Medium](https://images.openai.com/static-rsc-4/kGKMvDNgoAdseO0ZeQ-8H_ppk0B4ESlTa213zrm0p0Kui4w-740XBM6zaRy90L5AxCknEajJmWm83rNedxanyXIfwtnrddscwT6Phkfmys8-tQM_iq2OV9DD-t7oXKWk506e0BFWxmK9VEr8lIELC6Efuo6bSwp3Hq6gD2gyd-0?purpose=inline)

[medium.com](https://medium.com/accredian/building-an-interactive-eda-dashboard-with-plotly-a40b2005f1b6?utm_source=chatgpt.com)

### **Component tree**

```
ResearchAnalyticsPage
├── PageHeader
├── DatasetFilters
├── MetricDefinitionSelector
├── OverallPerformance
├── ClassWiseAnalysis
├── CorruptionAnalysis
├── PatientWiseAnalysis
├── StatisticalAnalysis
└── FailureCaseExplorer
```

### **Required analytical views**

| Visualization          | Purpose                           |
| ---------------------- | --------------------------------- |
| Bar chart              | Condition-wise comparison         |
| Line chart             | Severity vs performance           |
| Box plot               | Patient distribution              |
| Heatmap                | Class-wise degradation            |
| Scatter plot           | Patient-level metric relationship |
| Paired difference plot | Individual patient changes        |

### **Filters**

- Experiment
- Corruption type
- Severity
- Model checkpoint
- Anatomical class
- Evaluation metric
- Patient ID

Filters must be applied consistently to the data and every visualization.

### **Failure-case explorer**

This is one of the most useful components for research.

User can select a patient and inspect:

- Original MRI
- Ground-truth mask
- Predicted mask
- Difference mask
- Class-wise metrics
- Corruption condition

It should support identifying false positives, false negatives and boundary mismatches.

## **Page 06 — Reports**

**Route:** `/reports`

### **Components**

```
ReportsPage
├── ReportHeader
├── ReportTypeSelector
├── ReportConfiguration
├── ReportPreview
├── ExportControls
├── GeneratedReportsTable
└── ExportHistory
```

### **Report types**

1. Complete research report.
2. Baseline evaluation report.
3. Robustness experiment report.
4. Mitigation comparison report.
5. Individual segmentation report.

### **Export formats**

- PDF
- CSV
- JSON
- PNG overlays

JSON export will be useful for reproducibility and programmatic analysis.

### **Report integrity**

Every report should record:

- Report ID
- Generation timestamp
- Experiment identifiers
- Checkpoint identifier
- Dataset split
- Metric definitions
- Source artifact references
- Scientific limitations

A generated report should be a snapshot of the selected experiment data, not a dynamically changing document whose values silently change later.

## **Page 07 — Model & Dataset**

**Route:** `/model-dataset`

### **Components**

```
ModelDatasetPage
├── ModelOverview
├── CheckpointRegistry
├── ArchitectureInformation
├── DatasetOverview
├── CohortInformation
├── PreprocessingConfiguration
├── LabelMapping
└── ReproducibilityMetadata
```

### **Model section**

- Model architecture
- Input shape
- Output classes
- Parameter count
- Checkpoint version
- SHA-256 hash
- Training configuration
- Validation checkpoint information

### **Dataset section**

- ACDC dataset
- ED phase
- Patient-level split
- Patient counts
- Slice counts
- Label mapping
- Image spacing and geometry

### **Reproducibility section**

- Python version
- PyTorch version
- Dependencies
- Random seeds
- Configuration hash
- Experiment manifest

Checkpoint comparison should preserve the distinction between the historical 3-epoch and separate 50-epoch training runs. The latter must not be represented as a continuation of the former.

# **6. Frontend component architecture**

Ab hum pages se neeche aate hain.

Humein component structure aisa banana hai jahan common functionality repeat na karni pade aur image-viewer jaise complex components independently test ho sakein.

## **6.1 Proposed folder structure**

```
frontend/
│
├── public/
│
├── src/
│   ├── app/
│   │   ├── App.tsx
│   │   ├── router.tsx
│   │   └── providers.tsx
│   │
│   ├── layouts/
│   │   ├── AppShell.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Topbar.tsx
│   │   └── PageContainer.tsx
│   │
│   ├── pages/
│   │   ├── overview/
│   │   ├── segmentation/
│   │   ├── robustness/
│   │   ├── mitigation/
│   │   ├── analytics/
│   │   ├── reports/
│   │   └── model-dataset/
│   │
│   ├── components/
│   │   ├── ui/
│   │   ├── charts/
│   │   ├── image-viewer/
│   │   ├── metrics/
│   │   ├── experiments/
│   │   ├── tables/
│   │   ├── feedback/
│   │   └── common/
│   │
│   ├── features/
│   │   ├── upload/
│   │   ├── inference/
│   │   ├── segmentation/
│   │   ├── robustness/
│   │   ├── mitigation/
│   │   ├── analytics/
│   │   └── reports/
│   │
│   ├── services/
│   │   ├── api-client.ts
│   │   ├── upload-api.ts
│   │   ├── inference-api.ts
│   │   ├── experiment-api.ts
│   │   └── report-api.ts
│   │
│   ├── hooks/
│   │
│   ├── stores/
│   │
│   ├── types/
│   │
│   ├── utils/
│   │
│   └── styles/
│       ├── globals.css
│       ├── themes.css
│       └── tokens.css
│
├── tests/
│
├── package.json
└── vite.config.ts
```

This is a proposed frontend structure. Existing repository integration may change its location and naming.

## **6.2 Component responsibility boundaries**

| Component                | Responsibility                   |
| ------------------------ | -------------------------------- |
| `MRIUploadPanel`         | File selection and validation UI |
| `MRIViewer`              | Image display and navigation     |
| `SegmentationOverlay`    | Render mask over MRI             |
| `ClassVisibilityControl` | Toggle segmentation classes      |
| `MetricCard`             | Display a typed metric           |
| `ExperimentSelector`     | Choose experiment                |
| `ComparisonViewer`       | Side-by-side image comparison    |
| `StatisticalTable`       | Statistical results              |
| `JobStatusIndicator`     | Asynchronous job state           |
| `EvidenceBadge`          | Evidence classification          |
| `ThemeProvider`          | Theme management                 |

## **6.3 State management**

We should not put the entire application into one global state.

Recommended separation:

| State type         | Suggested owner       |
| ------------------ | --------------------- |
| Theme              | Global settings store |
| Sidebar state      | App shell             |
| Current MRI        | Segmentation feature  |
| Viewer zoom/pan    | MRI viewer            |
| Overlay visibility | Segmentation feature  |
| Active job         | Inference feature     |
| Experiment filters | Individual page       |
| API response cache | Query/cache layer     |
| Report generation  | Reports feature       |

Suggested technologies:

- Zustand for genuinely shared client state.
- TanStack Query for server-state caching, loading and invalidation.
- React component state for local interactions.

**Important:** MRI pixel arrays and large segmentation volumes should not be copied into global state repeatedly. Keep large image data in dedicated viewer/data-management layers.

## **6.4 Shared TypeScript domain types**

Illustrative domain contract:

```
type SegmentationClass =
  | "background"
  | "rv"
  | "myo"
  | "lv";

type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

type EvidenceStatus =
  | "verified"
  | "observed"
  | "mathematically_derived"
  | "not_tested"
  | "contradicted"
  | "unverifiable";

interface ImageMetadata {
  width: number;
  height: number;
  depth?: number;
  voxelSpacing?: number[];
  orientation?: string;
}

interface SegmentationJob {
  id: string;
  status: JobStatus;
  createdAt: string;
  error?: {
    code: string;
    message: string;
  };
}
```

These are proposed frontend types. Final definitions must match the backend's actual response schemas.

# **7. Backend integration architecture**

This is the most important engineering section.

Frontend ko existing research implementation ke saath integrate karna hai. Iska matlab hum existing Python research code ko unnecessarily rewrite nahi karenge.

## **7.1 Proposed architecture**

**React Frontend**

Pages · Viewer · Charts · Theme · User interactions

HTTPS / REST API

**Python API Layer**

FastAPI — proposed choice

Validation · Job orchestration · Serialization

**Inference Engine**

PyTorch / U-Net

**Research Engine**

Evaluation / experiments

**Data and Artifact Layer**

MRI · Masks · Checkpoints · Metrics · Experiment manifests · Reports

### **Responsibility boundaries**

**Frontend:**

- User interactions
- Image display
- Visualization
- Filtering
- Form validation for usability
- Result presentation

**API layer:**

- Request validation
- File handling
- Job creation
- Job status
- Access to research functions
- Response serialization

**Research engine:**

- Image preprocessing
- Model loading
- Inference
- Corruption generation
- Post-processing
- Metric computation
- Statistical analysis

**Storage layer:**

- Input files
- Output masks
- Experiment artifacts
- Checkpoints
- Metadata
- Reports

## **7.2 Backend integration principle: Adapter pattern**

One important decision: frontend endpoints should not directly depend on internal research script names.

For example, our current research documents refer to scripts such as:

```
scripts/evaluate_mitigation_test.py
scripts/audit_phase5_metrics.py
```

The frontend should not execute these scripts directly.

Instead:

```
Frontend
   ↓
API endpoint
   ↓
Application service
   ↓
Research adapter
   ↓
Existing Python implementation
```

Why?

Suppose tomorrow we change the implementation from one Python script to a reusable module.

If frontend directly depends on the script, multiple frontend integrations may break.

With an adapter, frontend contract remains stable.

The same principle applies to the model:

```
InferenceService
      ↓
ModelAdapter
      ↓
LightweightUNet2D
```

The model-loading and preprocessing logic should remain inside the research/backend layer.

# **8. Proposed API contract**

The following is the API specification we should validate and finalize after inspecting the actual Python implementation.

## **8.1 System endpoints**

| Method | Endpoint                | Purpose              |
| ------ | ----------------------- | -------------------- |
| GET    | `/api/v1/health`        | Backend availability |
| GET    | `/api/v1/system/status` | System readiness     |
| GET    | `/api/v1/models`        | Available models     |
| GET    | `/api/v1/models/{id}`   | Model metadata       |
| GET    | `/api/v1/datasets`      | Dataset information  |
| GET    | `/api/v1/experiments`   | Experiment registry  |

Health status and model readiness should be separate.

A backend can be online while the model checkpoint is unavailable.

## **8.2 Upload endpoints**

| Method | Endpoint                      | Purpose                              |
| ------ | ----------------------------- | ------------------------------------ |
| POST   | `/api/v1/images`              | Upload MRI                           |
| GET    | `/api/v1/images/{id}`         | Retrieve image metadata              |
| DELETE | `/api/v1/images/{id}`         | Delete temporary input, if permitted |
| GET    | `/api/v1/images/{id}/preview` | Retrieve preview                     |

Example request:

```
POST /api/v1/images
Content-Type: multipart/form-data
```

Example response:

```
{
  "image_id": "img_001",
  "status": "validated",
  "metadata": {
    "width": 256,
    "height": 256,
    "depth": 10
  }
}
```

Illustrative only. Actual metadata must be read from the MRI file, not invented.

## **8.3 Segmentation endpoints**

| Method | Endpoint                                 | Purpose             |
| ------ | ---------------------------------------- | ------------------- |
| POST   | `/api/v1/segmentation/jobs`              | Start inference     |
| GET    | `/api/v1/segmentation/jobs/{id}`         | Get job status      |
| POST   | `/api/v1/segmentation/jobs/{id}/cancel`  | Cancel if supported |
| GET    | `/api/v1/segmentation/jobs/{id}/result`  | Retrieve result     |
| GET    | `/api/v1/segmentation/jobs/{id}/metrics` | Retrieve metrics    |

### **Job creation**

Example:

```
{
  "image_id": "img_001",
  "model_id": "unet_50_epoch",
  "mode": "2d_slice",
  "slice_index": 4,
  "ground_truth_id": null
}
```

The backend must validate that the selected image and slice are supported.

### **Job response**

```
{
  "job_id": "job_001",
  "status": "queued",
  "created_at": "2026-11-21T14:00:00Z"
}
```

### **Job lifecycle**

**QUEUED**

**RUNNING**

SUCCEEDED

FAILED

CANCELLED

Terminal states should never transition back to running.

If the user retries a failed job, a new job ID should be created.

## **8.4 Experiment endpoints**

| Method | Endpoint                             | Purpose            |
| ------ | ------------------------------------ | ------------------ |
| GET    | `/api/v1/experiments`                | List experiments   |
| GET    | `/api/v1/experiments/{id}`           | Experiment details |
| GET    | `/api/v1/experiments/{id}/results`   | Metrics            |
| GET    | `/api/v1/experiments/{id}/artifacts` | Output files       |
| GET    | `/api/v1/experiments/{id}/patients`  | Patient-level data |

Robustness experiment filters:

```
GET /api/v1/experiments?type=robustness&corruption=blur
```

Mitigation experiment filters:

```
GET /api/v1/experiments?type=mitigation&method=clahe
```

These are proposed query parameters.

## **8.5 Analytics endpoints**

| Method | Endpoint                       | Purpose              |
| ------ | ------------------------------ | -------------------- |
| GET    | `/api/v1/analytics/overview`   | Summary              |
| GET    | `/api/v1/analytics/baseline`   | Baseline metrics     |
| GET    | `/api/v1/analytics/robustness` | Corruption metrics   |
| GET    | `/api/v1/analytics/mitigation` | Mitigation metrics   |
| GET    | `/api/v1/analytics/patients`   | Patient-wise metrics |
| GET    | `/api/v1/analytics/statistics` | Statistical results  |

The backend should calculate authoritative research metrics. Frontend may calculate display-only transformations, but must not independently recompute official Dice or statistical significance.

## **8.6 Report endpoints**

| Method | Endpoint                        | Purpose         |
| ------ | ------------------------------- | --------------- |
| POST   | `/api/v1/reports`               | Generate report |
| GET    | `/api/v1/reports`               | List reports    |
| GET    | `/api/v1/reports/{id}`          | Report metadata |
| GET    | `/api/v1/reports/{id}/download` | Download report |

Long report generation should use a job mechanism rather than block the HTTP request.

# **9. Backend response data contracts**

A stable API response schema is more important than the number of endpoints.

## **9.1 Standard response envelope**

Proposed:

```
{
  "data": {},
  "meta": {
    "request_id": "req_001"
  },
  "error": null
}
```

Error response:

```
{
  "data": null,
  "meta": {
    "request_id": "req_002"
  },
  "error": {
    "code": "INVALID_MRI_FILE",
    "message": "The uploaded MRI file is invalid."
  }
}
```

HTTP status codes must still correctly represent success or failure.

## **9.2 Metric schema**

This is particularly important.

Instead of:

```
{
  "dice": 0.8958
}
```

Prefer:

```
{
  "metric": "dice",
  "value": 0.8958,
  "class": "foreground",
  "aggregation": "patient_level",
  "cohort": "held_out_test",
  "sample_count": 15,
  "condition": "clean",
  "evidence_status": "reported"
}
```

The value above is an illustration using the reported baseline, not a newly verified result.

Metric schema should also support:

- Mean
- Standard deviation
- Median, where applicable
- Confidence interval
- Individual patient observations
- Slice-level observations
- Volume-level observations
- Metric computation version

Do not combine different aggregation methods into one chart without explicitly labelling them.

## **9.3 Artifact schema**

MRI predictions and masks should not be transmitted as huge JSON arrays.

Use artifact references.

```
{
  "artifact_id": "artifact_001",
  "type": "predicted_mask",
  "format": "nifti",
  "uri": "/api/v1/artifacts/artifact_001",
  "metadata": {
    "dimensions": [256, 256, 10],
    "class_count": 4
  }
}
```

The format and dimensions are examples.

For visualization, the backend may provide optimized 2D slice previews separately while retaining the original full-resolution output artifact.

**Critical requirement:** Resizing an image for browser preview must not modify the canonical inference input or mask.

# **10. Asynchronous execution and communication**

MRI inference and experiment execution can take longer than an ordinary UI request.

We need a proper job lifecycle.

## **Recommended initial architecture**

- HTTP for upload and job creation.
- HTTP polling for job status initially.
- Server-Sent Events or WebSocket only if a real need emerges.
- Durable result artifacts.
- Background worker if execution time or concurrency requires it.

Don't introduce Redis, Celery or Kubernetes automatically.

For a small research workstation, a simpler background-job implementation may be enough.

### **Frontend polling strategy**

```
Create job
   ↓
Receive job ID
   ↓
Poll job status
   ↓
Update UI
   ↓
Job completes
   ↓
Fetch result
   ↓
Invalidate related queries
```

Polling should:

- Stop on terminal status.
- Back off during long waits.
- Avoid duplicate requests.
- Recover from temporary network failures.
- Preserve the job ID on page navigation.
- Not pretend that polling progress is actual model progress.

### **Job cancellation**

Cancellation is optional and must only appear if the backend can safely implement it.

Stopping a browser request does not necessarily stop a running PyTorch inference operation.

# **11. Storage and persistence requirements**

We need to separate different kinds of data.

| Data                     | Storage responsibility    |
| ------------------------ | ------------------------- |
| Uploaded MRI             | File/object storage       |
| Ground truth             | Original dataset storage  |
| Model checkpoint         | Checkpoint storage        |
| Predicted masks          | Artifact storage          |
| Experiment configuration | Structured metadata       |
| Metrics                  | Structured result records |
| Statistical outputs      | Experiment artifacts      |
| Reports                  | File storage              |
| Logs                     | Application logging       |

### **Suggested logical entities**

```
Model
  └── Checkpoint

Dataset
  └── DatasetSplit
       └── Patient
            └── MRI Volume
                 └── Slice

Experiment
  ├── Configuration
  ├── ModelCheckpoint
  ├── DatasetSplit
  ├── Results
  └── Artifacts

InferenceJob
  ├── InputImage
  ├── ModelCheckpoint
  ├── Prediction
  └── Metrics

Report
  └── SourceExperiments
```

This is a logical domain model, not a commitment to a particular SQL schema.

### **Do we need a database immediately?**

Not necessarily.

For a single-user academic research application, structured JSON/YAML experiment manifests and a lightweight metadata database may be sufficient.

PostgreSQL becomes useful if we need:

- Multiple users
- Persistent experiment history
- Complex filtering
- Concurrent jobs
- Relational integrity
- Long-term audit history

The storage choice should follow actual usage, not architecture fashion.

# **12. Error handling, security and observability**

## **12.1 Error contract**

| Error                  | Expected frontend response  |
| ---------------------- | --------------------------- |
| `INVALID_MRI_FILE`     | Explain invalid input       |
| `UNSUPPORTED_FORMAT`   | Explain supported formats   |
| `INVALID_DIMENSIONS`   | Explain required dimensions |
| `MODEL_NOT_FOUND`      | Checkpoint unavailable      |
| `MODEL_LOAD_FAILED`    | Model initialization failed |
| `INFERENCE_FAILED`     | Execution failed            |
| `GROUND_TRUTH_MISSING` | Metrics unavailable         |
| `ARTIFACT_NOT_FOUND`   | Result file unavailable     |
| `JOB_NOT_FOUND`        | Job unavailable             |
| `STORAGE_ERROR`        | Storage operation failed    |

The frontend should show a safe, understandable message. Internal Python stack traces should remain in server logs.

## **12.2 Security requirements**

- Validate uploaded files server-side.
- Enforce upload size limits.
- Generate server-controlled file names.
- Prevent path traversal.
- Avoid accepting arbitrary filesystem paths.
- Never expose checkpoint storage paths.
- Protect file downloads.
- Apply appropriate CORS rules.
- Restrict allowed operations.
- Use temporary-file cleanup policies.
- Avoid exposing patient identifiers in unnecessary logs.

For this academic project, authentication can remain optional initially, but it should not be confused with secure storage or safe file handling.

## **12.3 Observability**

Every important operation should have:

- Request ID
- Job ID
- Experiment ID
- Model checkpoint ID
- Processing duration
- Success/failure status
- Error code
- Artifact references

This will help us debug the system when frontend and backend are integrated.

# **13. Frontend/backend integration requirements matrix**

This is our main coordination document.

| Feature       | Frontend requirement           | Backend requirement               |
| ------------- | ------------------------------ | --------------------------------- |
| Dashboard     | Display research summary       | Overview API                      |
| Upload        | Upload UI and metadata         | File validation and storage       |
| Viewer        | Display image and mask         | Preview/artifact delivery         |
| Inference     | Execute and track job          | Inference service                 |
| Metrics       | Display class-wise results     | Authoritative metric calculations |
| Robustness    | Filter and compare experiments | Experiment registry               |
| Mitigation    | Compare processing methods     | Locked experiment outputs         |
| Analytics     | Charts and filtering           | Aggregated research data          |
| Reports       | Generate and download          | Report generation                 |
| Model details | Display model metadata         | Checkpoint registry               |
| Theme         | Light/dark switching           | Not required                      |
| Job tracking  | Preserve job status            | Durable job status                |

### **Dependency order**

Some features cannot be implemented properly before their backend contracts exist.

## **Integration dependency map**

**API foundation**

Health · Schemas · Error handling

**Upload service**

MRI validation and storage

**Research registry**

Models and experiment data

**Core integration**

Inference · Results · Artifact retrieval

**Advanced research interface**

Robustness · Mitigation · Analytics · Reports

# **14. Frontend technology decisions**

Recommended stack:

| Layer         | Technology                              | Decision         |
| ------------- | --------------------------------------- | ---------------- |
| Framework     | React                                   | Use              |
| Language      | TypeScript                              | Use              |
| Build         | Vite                                    | Use              |
| Styling       | Tailwind CSS                            | Use              |
| UI components | shadcn/ui                               | Use selectively  |
| Charts        | Recharts                                | Use              |
| Routing       | React Router                            | Use              |
| Server state  | TanStack Query                          | Use              |
| Client state  | Zustand                                 | Use where needed |
| Forms         | React Hook Form + Zod                   | Use              |
| Image viewer  | OHIF/Cornerstone3D ecosystem evaluation | Evaluate         |
| Testing       | Vitest + React Testing Library          | Use              |
| E2E           | Playwright                              | Use              |

### **One important image-viewer decision**

We should not immediately build our own medical imaging renderer.

Evaluate two approaches:

**Option A — Lightweight custom 2D viewer**

Suitable for:

- 256 × 256 MRI slices
- Zoom and pan
- Segmentation overlays
- Class visibility
- Side-by-side comparisons

**Option B — Medical imaging viewer framework**

Suitable if we need:

- NIfTI volume navigation
- Physical-space orientation
- Advanced window/level
- More sophisticated medical imaging tools

My initial recommendation is a lightweight viewer behind a reusable viewer interface, but we should test NIfTI volume support before freezing this decision.

A full OHIF installation may be unnecessary for our current 2D research scope.

# **15. Non-functional requirements**

These requirements determine whether the application works reliably beyond a screenshot or demo.

| Category        | Requirement                                       |
| --------------- | ------------------------------------------------- |
| Usability       | Clear workflows and predictable interactions      |
| Responsiveness  | Desktop-first, tablet-compatible                  |
| Accessibility   | Keyboard navigation and readable contrast         |
| Performance     | Avoid unnecessary rendering of large image arrays |
| Reliability     | Recover from temporary API failures               |
| Integrity       | Never overwrite original research artifacts       |
| Reproducibility | Show checkpoint and experiment identifiers        |
| Security        | Validate files and protect storage                |
| Maintainability | Reusable feature-based components                 |
| Testing         | Unit, integration and end-to-end tests            |

### **Responsive behaviour**

Desktop:

- Full sidebar
- Three-panel segmentation workspace
- Multi-column analytics

Tablet:

- Collapsible sidebar
- Two-panel workspace
- Stacked charts

Mobile:

- Single-column layout
- Compact controls
- Scrollable image viewer
- No requirement to support complex research workflows on mobile initially

Desktop is the primary platform.

# **16. Testing and acceptance criteria**

Har page ka implementation tabhi complete hoga jab its behaviour is verified.

## **16.1 Frontend test matrix**

| Area       | Test                            |
| ---------- | ------------------------------- |
| Navigation | All routes work                 |
| Theme      | Switching works and persists    |
| Upload     | Valid/invalid files handled     |
| Viewer     | Zoom, pan and reset work        |
| Overlay    | Classes can be toggled          |
| Opacity    | Changes are reflected           |
| Inference  | Job lifecycle is handled        |
| Errors     | API errors are displayed safely |
| Metrics    | Correct aggregation labels      |
| Charts     | Filters update visualizations   |
| Reports    | Correct artifact downloaded     |
| Responsive | Layout works at target sizes    |

## **16.2 Critical medical-image tests**

These are particularly important.

1. Predicted mask dimensions must match the intended MRI geometry.
2. MRI and mask must align correctly.
3. Class label mapping must remain consistent.
4. Overlay opacity must not alter the actual mask.
5. Viewer resizing must not change source pixels.
6. Slice navigation must use correct indices.
7. Voxel spacing must not be silently discarded.
8. Exported masks must preserve the correct geometry.
9. Missing ground truth must not produce fabricated metrics.
10. The same input and checkpoint should produce reproducible outputs within the defined execution tolerance.

## **16.3 Acceptance gates**

**Gate 1 — Foundation**

Required

- Application starts.
- Routes work.
- Theme works.
- TypeScript checks pass.
- Basic tests pass.

  **Gate 2 — Segmentation**
- Upload validation works.
- MRI viewer works.
- Inference integration works.
- Predicted mask alignment verified.
- Ground-truth comparison verified.

  **Gate 3 — Research experiments**
- Registered experiments load.
- Metrics match source artifacts.
- Patient-level comparisons are correct.
- Statistical labels are accurate.

  **Gate 4 — Complete application**
- Analytics work.
- Reports export.
- Errors are handled.
- Security checks pass.
- End-to-end tests pass.

No phase will be frozen just because its pages look complete.

# **17. Implementation roadmap**

Ab is specification ko actual development phases mein convert karte hain.

## **Development execution plan**

1. **Phase 0 — Repository and backend audit**

   Inspect:
   - Existing repository
   - Python source modules
   - Model checkpoints
   - Research output structure
   - Existing inference functions
   - Existing experiment manifests
   - Existing APIs, if any

   Deliverable: `frontend_backend_audit.md`
2. **Phase 1 — API and domain contract**
   - Freeze image input schema
   - Freeze inference request/response
   - Define job lifecycle
   - Define metric schema
   - Define artifact references
   - Define error contract

   Deliverable: `api-contract.yaml` and `domain-model.md`
3. **Phase 2 — Frontend foundation**
   - React + TypeScript
   - Routing
   - Design tokens
   - Adaptive theme
   - Application shell
   - Common components

   Deliverable: Working application foundation
4. **Phase 3 — Segmentation Studio**
   - MRI upload
   - Validation
   - Viewer
   - Overlay
   - Inference
   - Result retrieval

   Deliverable: End-to-end segmentation workflow
5. **Phase 4 — Research Dashboard**
   - Overview
   - Robustness Lab
   - Mitigation Lab

   Deliverable: Research exploration interface
6. **Phase 5 — Analytics and Reports**
   - Patient-wise results
   - Statistical analysis
   - Failure cases
   - PDF/CSV exports

   Deliverable: Complete research reporting
7. **Phase 6 — Final integration and QA**
   - Backend/frontend testing
   - Image geometry verification
   - Security
   - Performance
   - Responsive behaviour
   - Final demo verification

   Deliverable: Verified application

# **18. Documents to freeze before implementation**

We should turn this specification into a durable project documentation set.

| Document                   | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| `frontend-prd.md`          | Product requirements                          |
| `ui-ux-specification.md`   | Page-wise visual and interaction requirements |
| `page-design.md`           | Page layouts and component behaviour          |
| `frontend-architecture.md` | React architecture and state management       |
| `component-catalog.md`     | Reusable component definitions                |
| `design-system.md`         | Theme, colours, typography and spacing        |
| `domain-model.md`          | Core entities and relationships               |
| `api-contract.yaml`        | Backend API schemas                           |
| `backend-integration.md`   | API-to-research-engine mapping                |
| `data-contracts.md`        | Metrics and artifact formats                  |
| `security-requirements.md` | File handling and API security                |
| `testing-strategy.md`      | Testing and acceptance criteria               |
| `phases.md`                | Implementation roadmap                        |
| `Everything-Flow.md`       | Complete system flow                          |

The documentation should distinguish existing implementation, verified research evidence, proposed design and unresolved decisions.

# **19. Decisions still requiring repository inspection**

Before freezing the API contract, these are the questions we must resolve.

| Question                                                         | Why it matters                     | Current status       |
| ---------------------------------------------------------------- | ---------------------------------- | -------------------- |
| Is there an existing inference function?                         | Avoid duplicate ML implementation  | Unknown              |
| Can the model load from a checkpoint through a reusable service? | Model adapter                      | Unknown              |
| Does inference support a full NIfTI volume?                      | Viewer and upload workflow         | Needs inspection     |
| Is there an existing experiment registry?                        | Research pages                     | Unknown              |
| Are results stored as CSV, JSON or another format?               | Data adapter                       | Partially documented |
| Are original masks and prediction masks available?               | Overlay viewer                     | Needs inspection     |
| Can artifacts be accessed independently?                         | API design                         | Unknown              |
| Does the project already have a backend server?                  | Avoid unnecessary backend creation | Unknown              |
| Is there an existing test suite for inference?                   | Integration testing                | Needs inspection     |

**We should not start Phase 2 frontend coding until the repository audit and API contract are sufficiently complete.**

# **20. Final architectural decisions**

Here is the proposed decision ledger.

## **Frontend architecture decision ledger**

**Development approach**

Selected

Integrated frontend/backend development.

**Visual theme**

Selected

Adaptive light/dark/system.

**Frontend framework**

Proposed

React + TypeScript + Vite.

**Backend framework**

To verify

Reuse existing Python implementation; FastAPI if a new API layer is needed.

**Image viewer**

To evaluate

Lightweight 2D viewer initially, subject to NIfTI volume requirements.

**Research results**

Fixed principle

Backend/artifact-derived, evidence-backed and traceable.

**Authentication**

Deferred

Not required for the initial single-user academic workflow.

**Coding**

Not started

Repository inspection and specification freeze first.
