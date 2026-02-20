*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           ${CURDIR}/../DatabaseCompareLibrary.py
Resource          ${CURDIR}/../simulation.resource
Suite Setup       Run Simulation    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/controlDynamicP.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        1e-4

*** Test Cases ***
Node Table Should Match Reference
    [Tags]    simulation
    Node Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}

Stand Table Should Match Reference
    [Tags]    simulation
    Stand Tables Should Be Equal     ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Stratum Table Should Match Reference
    [Tags]    simulation
    Stratum Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Tree Table Should Match Reference
    [Tags]    simulation
    Tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Removed_tree Table Should Match Reference
    [Tags]    simulation
    Removed_tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}