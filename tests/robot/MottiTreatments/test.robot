*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           ${CURDIR}/../DatabaseCompareLibrary.py
Resource          ${CURDIR}/../simulation.resource
Suite Setup       Run Simulation Check Upd    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}    ${REFERENCE_DIR}

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/data.dat
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control_motti_treatments.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        1e-4

*** Test Cases ***
Node Table Should Match Reference
    [Tags]    simulation    motti
    Skip
    Node Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}

Stand Table Should Match Reference
    [Tags]    simulation    motti
    Skip
    Stand Tables Should Be Equal     ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Stratum Table Should Match Reference
    [Tags]    simulation    motti
    Skip
    Stratum Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}

Tree Table Should Match Reference
    [Tags]    simulation    motti
    Skip
    Tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
