*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           ${CURDIR}/../DatabaseCompareLibrary.py
Library           ${CURDIR}/../CustomCompareLibrary.py
Resource          ${CURDIR}/../simulation.resource
Suite Setup       Run Simulation Check Upd    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}    ${REFERENCE_DIR}
Test Tags         vmi13

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control_dynamic_p.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        1e-4
${RELATIVE_TOLERANCE}    1e-6

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
		
Preprocessed Data Exported As CSV Should Match Reference
    [Tags]    preprocessing
    Compare Files With Numeric In Text
    ...    ${OUTPUT_PATH}/preprocessing_result.csv
    ...    ${REFERENCE_DIR}/preprocessing_result.csv
    ...    ${TOLERANCE}
    ...    ${RELATIVE_TOLERANCE}

Preprocessed Data Exported As RST Should Match Reference
    [Tags]    preprocessing
    Compare Files With Numeric In Text
    ...    ${OUTPUT_PATH}/preprocessing_result.rst
    ...    ${REFERENCE_DIR}/preprocessing_result.rst
    ...    ${TOLERANCE}
    ...    ${RELATIVE_TOLERANCE}
    