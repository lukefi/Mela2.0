*** Settings ***
Library     ../CustomCompareLibrary.py
Resource    ../simulation.resource


*** Variables ***
${INPUT_DIR}        ${CURDIR}/input
${OUT_DIR}          ${CURDIR}/output/test
${EXAMPLES_DIR}     ${CURDIR}/../../../examples

*** Test Cases ***
Conditions example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/conditions
    ...    ${EXAMPLES_DIR}/conditions.py

New events example should run
    Run Simulation
    ...    ${INPUT_DIR}/data.xml
    ...    ${OUT_DIR}/new_events
    ...    ${EXAMPLES_DIR}/control_new_events.py

New tree generation example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/new_tree_generation
    ...    ${EXAMPLES_DIR}/control_new_tree_generation.py

Vector example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/vector
    ...    ${EXAMPLES_DIR}/control_vector.py

Declarative conversion example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/declarative_conversion
    ...    ${EXAMPLES_DIR}/declarative_conversion.py

VMI12 tree generation example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI12_mini.dat
    ...    ${OUT_DIR}/vmi12_tree_generation
    ...    ${EXAMPLES_DIR}/vmi12_gen_trees.py

VMI13 tree generation example should run
    Run Simulation
    ...    ${INPUT_DIR}/VMI13_source_mini.dat
    ...    ${OUT_DIR}/vmi13_tree_generation
    ...    ${EXAMPLES_DIR}/vmi13_gen_trees.py
