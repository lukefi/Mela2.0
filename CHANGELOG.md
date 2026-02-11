# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
