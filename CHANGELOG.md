# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-08-19

### Changed

- Renamed the `sim` package to `core`
- Moved domain independent base classes under `core`
- Moved `select_units` under `core`
- Made the preprocessor interface generic
- Removed all domain dependencies from `core`
- Changed classes to use `__slots__` where applicable

## [1.0.2] - 2026-08-18

### Changed

- An exception is now raised if no valid forest stand entries are parsed from source data

## [1.0.1] - 2026-08-17

### Changed

- Run mode validation now checks that resimulation is not run together with other modes

## [1.0.0] - 2026-08-12

### Changed

- Replaced dictionary based control structure with a typed, class based one

## [0.11.2] - 2026-08-12

### Added

- Lists of predetermined treatments are now cleared once the treatments are performed by updating

## [0.11.1] - 2026-08-12

### Removed

- Removed unused input slicing functions

## [0.11.0] - 2026-08-12

### Added

- Added resimulation run mode

## [0.10.0] - 2026-08-10

### Removed

- Removed the input slicing feature

## [0.9.0] - 2026-06-30

### Added

- Added `updating` run mode for updating units to given time point and performing predetermined treatments at specific time points along the way
- Added mechanism for parsing performed operations into predetermined treatments from FC source data

### Changed

- Removed the `leaf` column in the output database `nodes` table and added a more generic `node_type` column instead
- The treatment `update_to_year` now also performs the predetermined treatments just like the new run mode

## [0.8.9] - 2026-06-17

### Changed

- Reduced database output performance overhead

## [0.8.8] - 2026-06-08

### Fixed

- Fixed node id calcuation for nested `Alternatives` generators

## [0.8.7] - 2026-05-26

### Fixed

- Removed all contributions from strata in `update_aggregates`
- Added fallback to using all trees for dominant height calculation if all trees are retention trees

## [0.8.6] - 2026-05-25

### Changed

- Updated project dependencies

## [0.8.5] - 2026-05-22

### Fixed

- Event tags are now inserted to operation history in addition to default treatment tags

## [0.8.4] - 2026-05-19

### Fixed

- Modified last interval check in `distributions.weibull` so that it is no longer sensitive to floating point
imprecision

## [0.8.3] - 2026-05-15

### Fixed

- `koealan_kasittelyluokka` is now properly read from VMI10 source data

## [0.8.2] - 2026-05-12

### Fixed

- Added missing forest stand column headers `peatland_type`, `drained_peatland_type`, `under_storey` and `over_storey`
in CSV output

## [0.8.1] - 2026-05-12

### Added

- Added `First` and `Optional` generators

## [0.8.0] - 2026-05-11

### Changed

- Overhauled simulation algorithm with no pre-generation of event tree structures

## [0.7.11] - 2026-05-08

### Changed

- Refactored and cleaned up forest builder classes

## [0.7.10] - 2026-05-08

### Fixed

- Added examples directory to setuptools `package-dir` list for proper package discovery

## [0.7.9] - 2026-05-08

### Fixed

- Repeated transitions now properly get unique node identifiers with multiple `-T`

## [0.7.8] - 2026-04-24

### Changed

- Motti state copy logic relocated to `finalize`

## [0.7.7] - 2026-04-22

### Fixed

- Added missing tree types 'B' and 'C'

## [0.7.6] - 2026-04-22

### Fixed

- Added removal of extra whitespace in peatland forest type conversion

## [0.7.5] - 2026-04-22

### Added

- Added `update_to_year` treatment ("ajantasaistus")

## [0.7.4] - 2026-04-22

### Fixed

- Added missing `natural_process_transition` decorators to `grow_metsi_fn` and `grow_acta_fn`

## [0.7.3] - 2026-04-22

### Added

- Added proper docstrings for FDM attributes

## [0.7.2] - 2026-04-22

### Changed

- Transition leaf nodes now get the value `2` in the `leaf` column

## [0.7.1] - 2026-04-21

### Added

- Added support for variable-length time steps

## [0.7.0] - 2026-04-21

### Added

- Database output nodes from transitions
- `NaturalProcessInfo` collected data type
- `natural_process_transition` decorator for collecting `NaturalProcessInfo` from natural process transitions

## [0.6.14] - 2026-04-21

### Added

- Added option to read csv_exp data as input

## [0.6.13] - 2026-04-21

### Added

- Added progress logging to `simulate_alternatives`

## [0.6.12] - 2026-04-21

### Added

- Added better console logging for ForestBuilder errors

## [0.6.11] - 2026-04-21

### Fixed

- Motti init called only once, state is held in stand.

## [0.6.10] - 2026-04-16

### Added

- Added `breast_height_age` and `volume` columns to `removed_trees` table in output database

### Fixed

- Added proper type information for `removed_trees` table columns in output database

## [0.6.9] - 2026-04-15

### Fixed

- Changed RST C-variable total length field to integer type

## [0.6.8] - 2026-04-14

### Fixed

- Fixed `test_area_handling_class` checks in `determine_forest_management_category`

## [0.6.7] - 2026-04-14

### Fixed

- Fixed tree origin conversion for Mela RST output

## [0.6.6] - 2026-04-09

### Fixed

- Fixed issues with peatland forest type conversion

## [0.6.5] - 2026-04-09

### Fixed

- Added missing cutting methods for upper layer thinning variants

## [0.6.4] - 2026-04-07

### Added

- Added missing more generic damage types from NFI9 and NFI10

## [0.6.3] - 2026-04-07

### Fixed

- Ahvenanmaa logic bug fixed

## [0.6.2] - 2026-04-01

### Fixed

- Add and refactor fields in vFDM

## [0.6.1] - 2026-03-31

### Fixed

- Added stratum basal area condition for tree height distribution stems scaling

## [0.6.0] - 2026-03-30

### Added

- New enumerated types for FDM attributes:
    - `ForestStand`:
        - `DevelopmentClass`
        - `CuttingMethod`
        - `FraLandUseClass`
    - `ReferenceTrees`:
        - `TreeManagementCategory`
        - `TreeCategory`
        - `TreeType`
        - `DamageType`
        - `CrownClass`
    - `TreeStrata`:
        - `StratumRank`

### Changed

- Renamed FDM attributes:
    - `tuhon_ilmiasu` -> `damage_type`
    - `latvuskerros` -> `crown_class`
    - `asema` -> `stratum_rank`
- Removed `ForestStand` attribute `land_use_category_detail`

### Fixed

- Small fixes and enhancements to mapping NFI values to FDM

## [0.5.20] - 2026-03-26

### Fixed

- Mela enums are now IntEnums so that int conversions work implicitly

## [0.5.19] - 2026-03-26

### Changed

- Minor refactoring of the tree generation helper function `_finalize_trees`

## [0.5.18] - 2026-03-26

### Changed

- Stratum attribute `tree_number` renamed to `stratum_number`
- Reference trees are now linked to strata by `stratum_number` instead of `identifier`

### Fixed

- Retention trees and generated trees are now properly linked to strata

## [0.5.17] - 2026-03-23

### Fixed

- Added apply_conversions to vmi9 and 10

## [0.5.16] - 2026-03-23

### Added

- Added csv_exp and metadata.json #MELA2-175

## [0.5.15] - 2026-03-19

### Added

- Added optional basal area based `stems_per_ha` scaling for reference trees generated from height distribution

## [0.5.14] - 2026-03-19

### Fixed

- Fixed tree species mapping for age model

## [0.5.13] - 2026-03-18

### Fixed

- Value scaling fixed in pljak_osuuspaino.csv files for VMI9, VMI10 and VMI11

## [0.5.12] - 2026-03-18

### Changed

- Refactor model.ReferenceTree and model.TreeStratum away

### Added
- vector_model.update_many and csv export logic

## [0.5.11] - 2026-03-16

### Added

- Option to reduce db output fields by sqlite_decl list

## [0.5.10] - 2026-03-16

### Added

- Separate tree height models for different VMI iterations
- Height is now calculated for retention trees missing it

## [0.5.9] - 2026-03-16

### Changed

- Refactor ykjtm35 to python #MELA2-170

## [0.5.8] - 2026-03-16

### Added

- VMI12 robot test

## [0.5.7] - 2026-03-12

### Added

- `rho` parameter for reference tree generation (lm)
- stem parameters fetched for tree generation (lm) by stand and VMI iteration

## [0.5.6] - 2026-03-12

### Fixed

- Tree generation strategy selection now works for strata without mean height

## [0.5.5] - 2026-03-10

### Added

- Separate parameter files for different VMI iterations

## [0.5.4] - 2026-03-10

### Fixed
- Added development_class in VMI9-10

## [0.5.3] - 2026-03-05

### Fixed
- unified forest_management_category vmi 11-13

### Added
 - Same logic to vmi10

## [0.5.2] - 2026-03-04

### Added

- carrier return check to VMI input data parser

## [0.5.1] - 2026-03-03

### Added

- VMI9, VMI10 and VMI11 input support

## [0.5.0] - 2026-03-02

### Changed

- New model for reference tree generation, currently only for NFI13 source data
- Management category determination rules changed
- Tree-to-strata matching logic changed, now based on tree crown class and stratum position in tree storey class

### Added

- New preprocessing operations:
    - scale_trees_by_area_weight_factors
    - scale_basal_area_at_county_level
    - update_strata_to_match_trees
    - area_ha_to_1000ha

## [0.4.17] - 2026-02-24

### Fixed

- Volume is now calculated only for large trees (> 1.3 m)

## [0.4.16] - 2026-02-24

### Changed

- Added robot test update flag #MELA2-170

## [0.4.15] - 2026-02-24

### Changed

- Remove unused Pukkala codes #MELA2-169

## [0.4.14] - 2026-02-20

### Changed

- Remove unused fields in vFDM #MELA2-149

## [0.4.13] - 2026-02-18

### Fixed

- Refactored lookup table to avoid unnecessary IO transactions. #MELA2-153

## [0.4.12] - 2026-02-11

### Added

- Option to change output file name.
- Prompting confirmation if outputfiles alreade exists #MELA2-127

## [0.4.11] - 2026-01-28

### Changed

- Removed last object containers from ForestStand #MELA2-154

## [0.4.10] - 2026-01-21

### Changed

- Read source data straight to SoA format #MELA2-130

## [0.4.9] - 2026-01-16

### Changed

- Data conversion output options #MELA2-108

## [0.4.8] - 2026-01-13

### Added

- Added boolean parameter `db_output` for Events to toggle database output

## [0.4.7] - 2025-12-18

### Added

- Tree volume calculation based on the [variable form factor model](https://doi.org/10.1093/forestry/cpac038)

## [0.4.6] - 2025-12-16

### Added

- Added new dynamic parameters feature with lookup table  #MELA2-28

## [0.4.5] - 2025-12-16

### Fixed

- Reads now age and age13 values from Motti model.

## [0.4.4] - 2025-12-12

### Fixed

- Aggregate variables are now calculated for the initial state and updated after every transition

## [0.4.3] - 2025-12-12

### Added

- Added new ForestStand attributes:
    - `main_tree_species_dominant_storey` determined once from source data
    - `region` read from source data, currently only VMI12 and VMI13 formats
    - `dominant_height_dominant_storey`, aggregate variable updated after every treatment and transition

## [0.4.2] - 2025-12-10

### Added

- Treatment class for containing various treatment metadata: name, default tags and the types of collected data

## [0.4.1] - 2025-12-10

### Changed

- Made Condition less generic by assuming SimulationPayload

## [0.4.0] - 2025-11-28

### Added

- Support for event tags
- Condition template for checking previous tags

## [0.3.4] - 2025-11-28

### Changed

- Made stand list slicing functions more generic

## [0.3.3] - 2025-11-27

### Fixed

- Fixed handling of failing conditions in the case of multiple simulation instructions

## [0.3.2] - 2025-11-25

### Added

- Added relative time based attributes and conditions

## [0.3.1] - 2025-11-25

### Added

- Added `leaf` column to `nodes` table in the output database to indicate leaf nodes

## [0.3.0] - 2025-11-25

### Changed

- Fixed time point simulation replaced with configurable transition function and conditions

## [0.2.8] - 2025-11-19

### Added

- SelectionTarget and SelectionSet now have `__repr__` and `__str__`

### Changed

- Treatments utilizing select_units now receive their target and set parameters as SelectionTarget and SelectionSet
objects

## [0.2.7] - 2025-11-19

### Fixed

- Minor typing fixes

## [0.2.6] - 2025-11-19

### Added

- Added mark_trees treatment and MarkRetentionTrees event #MELA2-118

## [0.2.5] - 2025-11-17

### Added

- Added cutting treatment #MELA2-104


## [0.2.4] - 2025-11-10

### Changed

- Motti robot test using db files for output and ref
- Motti wrapper exits for land_use_category 3 or higher as non supported
- Motti wrapper exits early if trees are not found

## [0.2.3] - 2025-11-07

### Added

- Added regeneration treatment and planting event

## [0.2.2] - 2025-11-07

### Fixed

- The output database is now created and initialized only if the `simulate` step is in the control run modes

## [0.2.1] - 2025-11-06

### Fixed

- An exception is now raised if deleting an existing database file fails

## [0.2.0] - 2025-11-06

### Added

- Aggregate variable calculation after treatments are performed

## [0.1.4] - 2025-11-05

### Fixed

- `level` mode step calculation fixed for `select_units`

## [0.1.3] - 2025-10-31

### Added

- Added robot test with motti binaries #MELA2-81

## [0.1.2] - 2025-10-31

### Changed

- Output database columns are now properly typed

## [0.1.1] - 2025-10-27

### Added

- Added operation Soil preparation #MELA2-110

## [0.1.0] - 2025-10-17

### Changed

- Simulation results are now output into an SQLite database

## [0.0.6] - 2025-10-17

### Added

- Added select_units helper algorithm

## [0.0.5] - 2025-10-17

### Changed

- Removed evaluating strings in preproc_filter, predicates are now given as functions in control file

## [0.0.4] - 2025-10-16

### Fixed

- Fixed create method for VectorData when using multidimensional data

## [0.0.3] - 2025-10-09

### Changed

- Speed optimization for grow_acta

## [0.0.2] - 2025-10-09

### Fixed

- Restored export operations

## [0.0.1] - 2025-10-07

Project forked from Metsi version 4.0.0.

### Changed

- Removed all non-vectorized operations/treatments
- Removed LayeredObject mechanism
- Removed support for rsts output format
