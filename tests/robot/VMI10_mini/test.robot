*** Settings ***
Library           ../CustomCompareLibrary.py    WITH NAME    Files
Library           ../DatabaseCompareLibrary.py  WITH NAME    DB
Resource          ../simulation.resource
Suite Setup       Setup And Run Simulation
Test Tags         vmi10

*** Keywords ***
Setup And Run Simulation
    Run Simulation    ${INPUT_DATA}    ${OUTPUT_PATH}    ${CONTROL_SCRIPT}

*** Variables ***
${INPUT_DATA}       ${CURDIR}/input/VMI10_mini.dat
${OUTPUT_PATH}      ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_PATH}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${ABS_TOL}          1e-12
${REL_TOL}          1e-12
${DB_TOL}           1e-12

*** Test Cases ***
Preprocessed Data Exported As CSV Should Match Reference
    [Tags]    preprocessing
    Files.Compare Files With Numeric In Text
    ...    ${OUTPUT_PATH}/preprocessing_result.csv
    ...    ${REFERENCE_DIR}/preprocessing_result.csv
    ...    ${ABS_TOL}
    ...    ${REL_TOL}

Preprocessed Data Exported As RST Should Match Reference
    [Tags]    preprocessing
    Files.Compare Files With Numeric In Text
    ...    ${OUTPUT_PATH}/preprocessing_result.rst
    ...    ${REFERENCE_DIR}/preprocessing_result.rst
    ...    ${ABS_TOL}
    ...    ${REL_TOL}

Node Table Should Match Reference
    [Tags]    simulation
    DB.Node Tables Should Be Equal    ${REFERENCE_DB}    ${OUTPUT_DB}

Stand Table Should Match Reference
    [Tags]    simulation
    DB.Stand Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${DB_TOL}

Stratum Table Should Match Reference
    [Tags]    simulation
    DB.Stratum Tables Should Be Equal  ${REFERENCE_DB}    ${OUTPUT_DB}    ${DB_TOL}

Tree Table Should Match Reference
    [Tags]    simulation
    DB.Tree Tables Should Be Equal    ${REFERENCE_DB}    ${OUTPUT_DB}    ${DB_TOL}