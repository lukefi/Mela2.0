*** Settings ***
Library     ../CustomCompareLibrary.py
Resource    ../simulation.resource


*** Variables ***
${INPUT_DIR}        ${CURDIR}/input
${OUT_DIR}          ${CURDIR}/output/test
${EXAMPLES_DIR}     ${CURDIR}/../../../examples

*** Test Cases ***
Default control should run
    [Tags]    simulation    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/default_control
    ...    ${CURDIR}/../../../control.py

Conditions example should run
    [Tags]    simulation    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/conditions
    ...    ${EXAMPLES_DIR}/conditions.py

New events example should run
    [Tags]    simulation    smk
    Run Simulation
    ...    ${INPUT_DIR}/data.xml
    ...    ${OUT_DIR}/new_events
    ...    ${EXAMPLES_DIR}/control_new_events.py

New tree generation example should run
    [Tags]    preprocessing    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/new_tree_generation
    ...    ${EXAMPLES_DIR}/control_new_tree_generation.py

Updating example should run
    [Tags]    simulation    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/updating
    ...    ${EXAMPLES_DIR}/control_updating.py

Vector example should run
    [Tags]    simulation    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/vector
    ...    ${EXAMPLES_DIR}/control_vector.py

Declarative conversion example should run
    [Tags]    preprocessing    vmi13
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/declarative_conversion
    ...    ${EXAMPLES_DIR}/declarative_conversion.py

VMI12 tree generation example should run
    [Tags]    preprocessing    vmi12
    Run Simulation
    ...    ${INPUT_DIR}/VMI12_mini.dat
    ...    ${OUT_DIR}/vmi12_tree_generation
    ...    ${EXAMPLES_DIR}/vmi12_gen_trees.py

VMI13 tree generation example should run
    [Tags]    preprocessing    vmi12
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/vmi13_tree_generation
    ...    ${EXAMPLES_DIR}/vmi13_gen_trees.py

Resimulation example should run
    [Tags]    resimulation
    Run Simulation
    ...    ${INPUT_DIR}/simulation_results.db
    ...    ${OUT_DIR}/resimulation
    ...    ${EXAMPLES_DIR}/control_resimulation.py
