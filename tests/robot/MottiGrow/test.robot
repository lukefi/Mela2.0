*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           ${CURDIR}/../DatabaseCompareLibrary.py
Resource          ${CURDIR}/../simulation.resource
Suite Setup       Run Simulation    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/data.xml
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control_motti_vec_grow.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        1e-4

*** Test Cases ***
Node Table Should Match Reference
    [Tags]    simulation    motti
    Node Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}

Stand Table Should Match Reference
    [Tags]    simulation    motti
    Stand Tables Should Be Equal     ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Stratum Table Should Match Reference
    [Tags]    simulation    motti
    Stratum Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Tree Table Should Match Reference
    [Tags]    simulation    motti
    Tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
