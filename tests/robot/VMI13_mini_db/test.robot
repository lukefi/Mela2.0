*** Settings ***
Library           OperatingSystem
Library           Process
Library           Collections
Library           String
Library           DatabaseLibrary
Library           ${CURDIR}/../CustomCompareLibrary.py

*** Variables ***
${SCRIPT}           -m
${MODULE}           lukefi.metsi.app.metsi
${INPUT_JSON}       ${CURDIR}/input/VMI13_source_mini.dat
${OUTPUT_DIR}       ${CURDIR}/output/test
${CONTROL_SCRIPT}   ${CURDIR}/input/control.py
${REFERENCE_DIR}    ${CURDIR}/output/ref
${TOLERANCE}        0.0000001  # Set your desired tolerance here
${REL_TOL}          1e-4

*** Keywords ***
Open Database Connections
    Connect To Database
    ...    sqlite3
    ...    db_name=${OUTPUT_DIR}/simulation_results.db
    ...    alias=sim

    Connect To Database
    ...    sqlite3
    ...    db_name=${REFERENCE_DIR}/simulation_results.db
    ...    alias=ref

*** Test Cases ***
Run Simulation And Compare Output Files
    [Tags]    simulation

    Remove Directory    ${OUTPUT_DIR}    recursive=True
    Create Directory    ${OUTPUT_DIR}


    ${orig_env}=    Get Environment Variables
    Set To Dictionary    ${orig_env}    PYTHONPATH=${EXECDIR}
    ${result}=    Run Process    python
    ...           ${SCRIPT}
    ...           ${MODULE}
    ...           ${INPUT_JSON}
    ...           ${OUTPUT_DIR}
    ...           ${CONTROL_SCRIPT}
    ...           shell=True
    ...           env=${orig_env}

    Log    STDOUT:\n${result.stdout}
    Log    STDERR:\n${result.stderr}

    Should Be Equal As Integers    ${result.rc}    0    msg=Python script failed! See STDERR log for details.

    Log To Console    Simulation Succeeded. Verifying output files...

    Open Database Connections
    