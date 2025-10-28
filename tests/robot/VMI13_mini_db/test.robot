*** Settings ***
Library           ../DatabaseCompareLibrary.py
Resource          ../simulation.resource

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        0.0

*** Test Cases ***
Simulation Output Should Match Reference
    [Tags]    simulation

    Run Simulation    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}

    Log To Console    Simulation Succeeded. Verifying output files...

    Node Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}
    Stand Tables Should Be Equal     ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
    Stratum Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
    Tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
