*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           ../DatabaseCompareLibrary.py

*** Variables ***
${INPUT_JSON}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_DIR}       ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${OUTPUT_DB}        ${OUTPUT_DIR}/simulation_results.db
${REFERENCE_DB}     ${REFERENCE_DIR}/simulation_results.db
${TOLERANCE}        0.0000001  # Set your desired tolerance here
${REL_TOL}          1e-4


*** Keywords ***
Run Simulation
    Remove Directory    ${OUTPUT_DIR}    recursive=True
    Create Directory    ${OUTPUT_DIR}

    ${orig_env}=    Get Environment Variables
    Set To Dictionary    ${orig_env}    PYTHONPATH=${EXECDIR}
    ${result}=    Run Process    python
    ...           -m
    ...           lukefi.metsi.app.metsi
    ...           ${INPUT_JSON}
    ...           ${OUTPUT_DIR}
    ...           ${CONTROL_SCRIPT}
    ...           shell=True
    ...           env=${orig_env}

    Log    STDOUT:\n${result.stdout}
    Log    STDERR:\n${result.stderr}

    Should Be Equal As Integers    ${result.rc}    0    msg=Python script failed! See STDERR log for details.

*** Test Cases ***
Simulation Output Should Match Reference
    [Tags]    simulation

    Run Simulation

    Log To Console    Simulation Succeeded. Verifying output files...

    Node Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}
    Stand Tables Should Be Equal     ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
    Stratum Tables Should Be Equal   ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
    Tree Tables Should Be Equal      ${REFERENCE_DB}    ${OUTPUT_DB}    ${TOLERANCE}
